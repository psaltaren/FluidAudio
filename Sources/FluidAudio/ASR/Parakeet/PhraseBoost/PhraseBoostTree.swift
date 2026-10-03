import Foundation

/// Prefix tree with Aho–Corasick fail links for phrase boosting during greedy TDT decoding.
///
/// A port of NeMo's GPU-PB boosting tree (`ContextGraph` / `GPUBoostingTreeModel`, NeMo 2.7,
/// Apache 2.0; Andrusenko et al., "TurboBias", arXiv 2508.07014), following the single-hypothesis
/// semantics of Klang AI's `phrase_boost.py` for Pianissimo:
///
/// - An arc at depth 1 scores `contextScore`; deeper arcs score `contextScore * depthScaling +
///   ln(depth)`. A node carries the accumulated score of the path to it.
/// - From a state, a token that continues a phrase (directly or through fail links) earns that arc's
///   score. Every fail link taken on the way from a node that does not end a phrase takes back the
///   partially accrued bonus (`fail.nodeScore - node.nodeScore`). A token that starts no phrase
///   earns only the take-back (0 from the root).
///
/// States are node indices; 0 is the root.
public struct PhraseBoostTree: Sendable {

    public static let rootState = 0

    private var children: [[Int: Int]]
    private var tokenScore: [Double]
    private var nodeScore: [Double]
    private var isEnd: [Bool]
    private var fail: [Int]

    /// Number of nodes including the root.
    public var nodeCount: Int { children.count }

    /// - Parameters:
    ///   - phrases: Token sequences. Duplicates and empty sequences are ignored.
    ///   - contextScore: Score of the first arc (`c0`, NeMo `context_score`).
    ///   - depthScaling: Multiplier for deeper arcs (NeMo `depth_scaling`).
    public init(phrases: [[Int]], contextScore: Double = 1.0, depthScaling: Double = 2.0) {
        children = [[:]]
        tokenScore = [0]
        nodeScore = [0]
        isEnd = [false]
        fail = [0]

        var seen = Set<[Int]>()
        for phrase in phrases where !phrase.isEmpty && seen.insert(phrase).inserted {
            add(phrase, contextScore: contextScore, depthScaling: depthScaling)
        }
        fillFailLinks()
    }

    private mutating func add(_ tokens: [Int], contextScore: Double, depthScaling: Double) {
        var node = Self.rootState
        for (depth, token) in tokens.enumerated() {
            let last = depth == tokens.count - 1
            if let child = children[node][token] {
                tokenScore[child] = max(contextScore, tokenScore[child])
                nodeScore[child] = nodeScore[node] + tokenScore[child]
                isEnd[child] = last || isEnd[child]
                node = child
            } else {
                let score = depth > 0 ? contextScore * depthScaling + log(Double(depth + 1)) : contextScore
                let child = children.count
                children.append([:])
                tokenScore.append(score)
                nodeScore.append(nodeScore[node] + score)
                isEnd.append(last)
                fail.append(Self.rootState)
                children[node][token] = child
                node = child
            }
        }
    }

    private mutating func fillFailLinks() {
        var queue: [Int] = []
        var head = 0
        for child in orderedChildren(of: Self.rootState) {
            fail[child.node] = Self.rootState
            queue.append(child.node)
        }
        while head < queue.count {
            let node = queue[head]
            head += 1
            for (token, child) in orderedChildren(of: node) {
                var link = fail[node]
                while children[link][token] == nil && link != Self.rootState {
                    link = fail[link]
                }
                fail[child] = children[link][token] ?? Self.rootState
                queue.append(child)
            }
        }
    }

    /// Children in creation order, so traversal is deterministic (node ids grow with insertion).
    private func orderedChildren(of node: Int) -> [(token: Int, node: Int)] {
        children[node].map { (token: $0.key, node: $0.value) }.sorted { $0.node < $1.node }
    }

    /// Bonus for emitting `token` from `state`, and the state after it.
    public func step(from state: Int, token: Int) -> (bonus: Float, next: Int) {
        var node = state
        var takenBack = 0.0
        while true {
            if let child = children[node][token] {
                return (Float(takenBack + tokenScore[child]), child)
            }
            if node == Self.rootState {
                return (Float(takenBack), Self.rootState)
            }
            if !isEnd[node] {
                takenBack += nodeScore[fail[node]] - nodeScore[node]
            }
            node = fail[node]
        }
    }

    /// Tokens that continue a phrase from `state` (through fail links), with their bonus and next
    /// state, plus the bonus every other token gets (the full take-back to the root).
    public func transitions(from state: Int) -> (explicit: [Int: (bonus: Float, next: Int)], fallback: Float) {
        var result: [Int: (bonus: Float, next: Int)] = [:]
        var node = state
        var takenBack = 0.0
        while true {
            for (token, child) in children[node] where result[token] == nil {
                result[token] = (Float(takenBack + tokenScore[child]), child)
            }
            if node == Self.rootState { return (result, Float(takenBack)) }
            if !isEnd[node] {
                takenBack += nodeScore[fail[node]] - nodeScore[node]
            }
            node = fail[node]
        }
    }
}

/// Phrase boosting configuration for TDT greedy decoding (Parakeet v3 joint with top-K outputs).
///
/// At every step where the joint's argmax is a non-blank token, the decoder picks
/// `argmax(logit + alpha * bonus)` over the non-blank top-K candidates instead. Blank decisions are
/// never changed, so boosting can swap one word for another but cannot insert words into silence.
/// The tree state is carried in `TdtDecoderState`, so a phrase can span chunk boundaries.
public struct PhraseBoost: Sendable {
    public let tree: PhraseBoostTree
    public let alpha: Float

    public init(tree: PhraseBoostTree, alpha: Float = 1.0) {
        self.tree = tree
        self.alpha = alpha
    }

    /// Builds the tree from phrases as written (case-sensitive), tokenized with `encoder`.
    public init(
        phrases: [String],
        encoder: SentencePieceBPEEncoder,
        alpha: Float = 1.0,
        contextScore: Double = 1.0,
        depthScaling: Double = 2.0
    ) {
        self.init(
            tree: PhraseBoostTree(
                phrases: phrases.map { encoder.encode($0) },
                contextScore: contextScore,
                depthScaling: depthScaling),
            alpha: alpha)
    }

    /// The boosted choice among the top-K candidates that `isAllowed` accepts, or nil when boosting
    /// does not apply (blank argmax, no candidates). Ties go to the earlier candidate.
    func choose(
        label: Int,
        topKIds: [Int],
        topKLogits: [Float],
        state: Int,
        blankId: Int,
        isAllowed: (Int) -> Bool = { _ in true }
    ) -> (token: Int, logit: Float)? {
        guard label != blankId else { return nil }
        var best = -1
        var bestScore = -Float.infinity
        for i in 0..<min(topKIds.count, topKLogits.count) {
            let id = topKIds[i]
            guard id != blankId, isAllowed(id) else { continue }
            let score = topKLogits[i] + alpha * tree.step(from: state, token: id).bonus
            if best < 0 || score > bestScore {
                best = i
                bestScore = score
            }
        }
        guard best >= 0 else { return nil }
        return (topKIds[best], topKLogits[best])
    }
}
