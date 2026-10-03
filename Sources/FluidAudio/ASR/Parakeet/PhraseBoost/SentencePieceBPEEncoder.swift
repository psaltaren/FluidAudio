import Foundation

/// Text → token ids for a SentencePiece **BPE** model, matching `SentencePieceProcessor.encode`.
///
/// Parakeet's `parakeet_vocab.json` carries ids and piece strings but not the merge order, which only
/// lives in the scores of `tokenizer.model`. This encoder takes the pieces with their scores and
/// types (exported from `tokenizer.model`) and reproduces SentencePiece's encoding:
///
/// 1. `nmt_nfkc` normalization: NMT control-character removal and whitespace mapping, then NFKC.
/// 2. Spaces become `▁`, with a dummy `▁` prefix when `addDummyPrefix` is set. Extra whitespace is
///    kept when `removeExtraWhitespaces` is false (Pianissimo's setting).
/// 3. The text is split into code points; user-defined pieces (type 4, e.g. `<|nospeech|>`) are
///    matched whole and never merged.
/// 4. Adjacent symbols are merged greedily, highest piece score first, leftmost on ties, as long as
///    the merged string is a piece.
/// 5. Symbols that are not pieces map to the unknown id, and a run of unknowns becomes one token.
///
/// Used to turn boosting phrases into token sequences without shipping SentencePiece itself.
public struct SentencePieceBPEEncoder: Sendable {

    /// SentencePiece piece types, as stored in `ModelProto.SentencePiece.Type`.
    public enum PieceType: Int, Sendable {
        case normal = 1
        case unknown = 2
        case control = 3
        case userDefined = 4
        case unused = 5
        case byte = 6
    }

    public struct Piece: Sendable {
        public let piece: String
        public let score: Float
        public let type: PieceType

        public init(piece: String, score: Float, type: PieceType) {
            self.piece = piece
            self.score = score
            self.type = type
        }
    }

    public enum LoadError: Error, LocalizedError {
        case invalidFormat(String)
        case unsupported(String)

        public var errorDescription: String? {
            switch self {
            case .invalidFormat(let detail): return "Invalid SentencePiece export: \(detail)"
            case .unsupported(let detail): return "Unsupported SentencePiece model: \(detail)"
            }
        }
    }

    static let spaceSymbol: Character = "\u{2581}"

    public let unknownId: Int
    public let addDummyPrefix: Bool
    public let removeExtraWhitespaces: Bool

    /// Pieces SentencePiece may produce by merging or matching (normal, user-defined, unused).
    private let pieceIds: [String: Int]
    /// Unknown and control pieces: looked up first, like SentencePiece's `reserved_id_map_`.
    private let reservedIds: [String: Int]
    private let scores: [Float]
    /// User-defined pieces as scalar arrays, longest first, for whole-symbol prefix matching.
    private let userDefined: [[Unicode.Scalar]]

    public init(
        pieces: [Piece],
        unknownId: Int = 0,
        addDummyPrefix: Bool = true,
        removeExtraWhitespaces: Bool = false
    ) throws {
        var pieceIds: [String: Int] = [:]
        var reservedIds: [String: Int] = [:]
        var userDefined: [[Unicode.Scalar]] = []
        for (id, piece) in pieces.enumerated() {
            switch piece.type {
            case .normal, .userDefined, .unused:
                if pieceIds[piece.piece] == nil { pieceIds[piece.piece] = id }
                if piece.type == .userDefined { userDefined.append(Array(piece.piece.unicodeScalars)) }
            case .unknown, .control:
                if reservedIds[piece.piece] == nil { reservedIds[piece.piece] = id }
            case .byte:
                throw LoadError.unsupported("byte-fallback pieces")
            }
            if piece.type == .unused {
                // SentencePiece re-segments merges that land on unused pieces. Parakeet's
                // tokenizers have none, so that path is not implemented.
                throw LoadError.unsupported("unused pieces")
            }
        }
        guard unknownId >= 0, unknownId < pieces.count else {
            throw LoadError.invalidFormat("unknown id \(unknownId) out of range")
        }
        self.pieceIds = pieceIds
        self.reservedIds = reservedIds
        self.scores = pieces.map(\.score)
        self.userDefined = userDefined.sorted { $0.count > $1.count }
        self.unknownId = unknownId
        self.addDummyPrefix = addDummyPrefix
        self.removeExtraWhitespaces = removeExtraWhitespaces
    }

    /// Loads the JSON export of `tokenizer.model`:
    /// `{"type": "BPE", "normalizer": "nmt_nfkc", "add_dummy_prefix": true,
    ///   "remove_extra_whitespaces": false, "unk_id": 0, "pieces": [[piece, score, type], ...]}`.
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoadError.invalidFormat("root is not an object")
        }
        if let type = root["type"] as? String, type != "BPE" {
            throw LoadError.unsupported("model type \(type)")
        }
        if let normalizer = root["normalizer"] as? String, normalizer != "nmt_nfkc" {
            throw LoadError.unsupported("normalizer \(normalizer)")
        }
        guard let rawPieces = root["pieces"] as? [[Any]] else {
            throw LoadError.invalidFormat("missing pieces")
        }
        var pieces: [Piece] = []
        pieces.reserveCapacity(rawPieces.count)
        for (index, entry) in rawPieces.enumerated() {
            guard entry.count == 3,
                let text = entry[0] as? String,
                let score = (entry[1] as? NSNumber)?.floatValue,
                let rawType = (entry[2] as? NSNumber)?.intValue,
                let type = PieceType(rawValue: rawType)
            else {
                throw LoadError.invalidFormat("piece \(index)")
            }
            pieces.append(Piece(piece: text, score: score, type: type))
        }
        try self.init(
            pieces: pieces,
            unknownId: (root["unk_id"] as? NSNumber)?.intValue ?? 0,
            addDummyPrefix: (root["add_dummy_prefix"] as? Bool) ?? true,
            removeExtraWhitespaces: (root["remove_extra_whitespaces"] as? Bool) ?? false
        )
    }

    /// Token ids for `text`, identical to `SentencePieceProcessor.encode(text)`.
    public func encode(_ text: String) -> [Int] {
        encodeWithPieces(text).map(\.id)
    }

    /// Token ids with their piece strings (a run of unknown characters is one piece).
    public func encodeWithPieces(_ text: String) -> [(piece: String, id: Int)] {
        let normalized = normalize(text)
        guard !normalized.isEmpty else { return [] }

        // Split into symbols: a whole user-defined piece (frozen), otherwise one code point.
        var symbols: [[Unicode.Scalar]] = []
        var frozen: [Bool] = []
        var index = 0
        while index < normalized.count {
            if let match = userDefinedMatch(in: normalized, at: index) {
                symbols.append(Array(normalized[index..<index + match]))
                frozen.append(true)
                index += match
            } else {
                symbols.append([normalized[index]])
                frozen.append(false)
                index += 1
            }
        }

        // Greedy BPE. SentencePiece keeps a lazy priority queue ordered by (score desc, left index
        // asc); taking the best currently adjacent mergeable pair each round is the same order.
        var next = Array(1...symbols.count).map { $0 == symbols.count ? -1 : $0 }
        var prev = Array(-1..<symbols.count - 1)
        var alive = [Bool](repeating: true, count: symbols.count)
        var pairScore = [Float?](repeating: nil, count: symbols.count)

        func refresh(_ left: Int) {
            guard left >= 0, alive[left] else { return }
            let right = next[left]
            guard right >= 0, !frozen[left], !frozen[right] else {
                pairScore[left] = nil
                return
            }
            let merged = String(String.UnicodeScalarView(symbols[left] + symbols[right]))
            if let id = pieceIds[merged] {
                pairScore[left] = scores[id]
            } else {
                pairScore[left] = nil
            }
        }

        for left in 0..<symbols.count { refresh(left) }

        while true {
            var best = -1
            var bestScore = -Float.infinity
            var cursor = 0
            while cursor >= 0 {
                if let score = pairScore[cursor], best < 0 || score > bestScore {
                    best = cursor
                    bestScore = score
                }
                cursor = next[cursor]
            }
            guard best >= 0 else { break }

            let right = next[best]
            symbols[best] += symbols[right]
            alive[right] = false
            pairScore[right] = nil
            next[best] = next[right]
            if next[right] >= 0 { prev[next[right]] = best }
            refresh(best)
            refresh(prev[best])
        }

        var output: [(piece: String, id: Int)] = []
        var cursor = 0
        while cursor >= 0 {
            let piece = String(String.UnicodeScalarView(symbols[cursor]))
            let id = pieceId(piece)
            if id == unknownId, let last = output.last, last.id == unknownId {
                output[output.count - 1] = (last.piece + piece, unknownId)
            } else {
                output.append((piece, id))
            }
            cursor = next[cursor]
        }
        return output
    }

    private func pieceId(_ piece: String) -> Int {
        if let id = reservedIds[piece] { return id }
        if let id = pieceIds[piece] { return id }
        return unknownId
    }

    private func userDefinedMatch(in scalars: [Unicode.Scalar], at index: Int) -> Int? {
        for candidate in userDefined where index + candidate.count <= scalars.count {
            var matches = true
            for (offset, scalar) in candidate.enumerated() where scalars[index + offset] != scalar {
                matches = false
                break
            }
            if matches { return candidate.count }
        }
        return nil
    }

    /// `nmt_nfkc` plus SentencePiece's whitespace escaping, as code points.
    func normalize(_ text: String) -> [Unicode.Scalar] {
        var mapped = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            let value = scalar.value
            if Self.nmtRemoved(value) { continue }
            mapped.append(Self.nmtWhitespace(value) ? " " : scalar)
        }
        let nfkc = String(mapped).precomposedStringWithCompatibilityMapping
        var scalars = Array(nfkc.unicodeScalars)
        guard !scalars.isEmpty else { return [] }

        if removeExtraWhitespaces {
            var collapsed: [Unicode.Scalar] = []
            for scalar in scalars {
                if scalar == " " && (collapsed.isEmpty || collapsed.last == " ") { continue }
                collapsed.append(scalar)
            }
            if collapsed.last == " " { collapsed.removeLast() }
            scalars = collapsed
            guard !scalars.isEmpty else { return [] }
        }

        let space = Self.spaceSymbol.unicodeScalars.first!
        var output = scalars.map { $0 == " " ? space : $0 }
        if addDummyPrefix { output.insert(space, at: 0) }
        return output
    }

    private static func nmtRemoved(_ value: UInt32) -> Bool {
        switch value {
        case 0x0001...0x0008, 0x000B, 0x000E...0x001F, 0x007F, 0x008F, 0x009F: return true
        default: return false
        }
    }

    private static func nmtWhitespace(_ value: UInt32) -> Bool {
        switch value {
        case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B, 0x200C, 0x200E, 0x200F, 0x2028, 0x2029,
            0x2581, 0xFEFF, 0xFFFD:
            return true
        default: return false
        }
    }
}
