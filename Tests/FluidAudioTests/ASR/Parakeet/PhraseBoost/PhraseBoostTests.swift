import Foundation
import XCTest

@testable import FluidAudio

/// Parity of the Swift phrase-boosting pieces with their Python references (see PhraseBoostFixtures/README.md).
final class PhraseBoostTests: XCTestCase {

    private static func fixtureURL(_ name: String) throws -> URL {
        let url = Bundle.module.url(forResource: "PhraseBoostFixtures", withExtension: nil)?
            .appendingPathComponent(name)
        guard let url, FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Missing fixture \(name)")
        }
        return url
    }

    private static func encoder() throws -> SentencePieceBPEEncoder {
        try SentencePieceBPEEncoder(contentsOf: fixtureURL("pianissimo-sv-tokenizer.json"))
    }

    // MARK: - Encoder

    func testEncoderMatchesSentencePieceOnAllFixtures() throws {
        let encoder = try Self.encoder()
        let data = try Data(contentsOf: Self.fixtureURL("encode_fixtures.json"))
        let fixtures = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        XCTAssertGreaterThan(fixtures.count, 9000)

        var mismatches: [String] = []
        for fixture in fixtures {
            let text = try XCTUnwrap(fixture["text"] as? String)
            let ids = try XCTUnwrap(fixture["ids"] as? [Int])
            let pieces = try XCTUnwrap(fixture["pieces"] as? [String])
            let encoded = encoder.encodeWithPieces(text)
            if encoded.map(\.id) != ids || encoded.map(\.piece) != pieces {
                mismatches.append("\(text.debugDescription): \(encoded.map(\.piece)) != \(pieces)")
            }
        }
        XCTAssertEqual(mismatches.count, 0, mismatches.prefix(10).joined(separator: "\n"))
    }

    func testEncoderKnownCases() throws {
        let encoder = try Self.encoder()
        XCTAssertEqual(encoder.encodeWithPieces("Sæga").map(\.piece), ["\u{2581}S", "æ", "ga"])
        XCTAssertEqual(encoder.encodeWithPieces("Säga").map(\.piece), ["\u{2581}S", "ä", "ga"])
        XCTAssertEqual(encoder.encodeWithPieces("ﬁnal").map(\.piece), ["\u{2581}final"])
        // Equal-score merges go to the leftmost pair, as in SentencePiece (checked against Python).
        XCTAssertEqual(encoder.encodeWithPieces("aaaa").map(\.piece), ["\u{2581}a", "aa", "a"])
        XCTAssertEqual(encoder.encodeWithPieces("eaaaaa").map(\.piece), ["\u{2581}e", "aa", "aa", "a"])
        XCTAssertEqual(encoder.encode("<|nospeech|>"), [7863, 1])
        XCTAssertEqual(encoder.encode(""), [])
    }

    // MARK: - Tree

    func testTreeMatchesReferenceForEveryState() throws {
        let data = try Data(contentsOf: Self.fixtureURL("tree_fixtures.json"))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let contextScore = try XCTUnwrap(root["context_score"] as? Double)
        let depthScaling = try XCTUnwrap(root["depth_scaling"] as? Double)
        let sets = try XCTUnwrap(root["sets"] as? [[String: Any]])
        let encoder = try Self.encoder()
        XCTAssertEqual(sets.count, 2)

        for set in sets {
            let name = set["name"] as? String ?? "?"
            let phrases = try XCTUnwrap(set["phrases"] as? [String])
            let tokens = try XCTUnwrap(set["tokens"] as? [[Int]])
            XCTAssertEqual(phrases.map { encoder.encode($0) }, tokens, "\(name): phrase encoding")

            let tree = PhraseBoostTree(phrases: tokens, contextScore: contextScore, depthScaling: depthScaling)
            let states = try XCTUnwrap(set["states"] as? [[Any]])
            XCTAssertEqual(tree.nodeCount, states.count, "\(name): node count")

            let rootArcs = try XCTUnwrap(set["root"] as? [[NSNumber]]).map {
                (token: $0[0].intValue, score: $0[1].floatValue, next: $0[2].intValue)
            }
            let unusedToken = 8191  // a piece no phrase in either set starts with

            for (state, entry) in states.enumerated() {
                let fallback = try XCTUnwrap(entry[0] as? NSNumber).floatValue
                var expected: [Int: (bonus: Float, next: Int)] = [:]
                for arc in rootArcs { expected[arc.token] = (fallback + arc.score, arc.next) }
                for arc in try XCTUnwrap(entry[1] as? [[NSNumber]]) {
                    expected[arc[0].intValue] = (arc[1].floatValue, arc[2].intValue)
                }
                XCTAssertNil(expected[unusedToken])

                let (explicit, swiftFallback) = tree.transitions(from: state)
                XCTAssertEqual(swiftFallback, fallback, accuracy: 1e-5, "\(name) state \(state): fallback")
                XCTAssertEqual(Set(explicit.keys), Set(expected.keys), "\(name) state \(state): tokens")
                for (token, want) in expected {
                    let got = tree.step(from: state, token: token)
                    XCTAssertEqual(got.next, want.next, "\(name) state \(state) token \(token): next")
                    XCTAssertEqual(got.bonus, want.bonus, accuracy: 1e-5, "\(name) state \(state) token \(token)")
                }
                let other = tree.step(from: state, token: unusedToken)
                XCTAssertEqual(other.next, PhraseBoostTree.rootState)
                XCTAssertEqual(other.bonus, fallback, accuracy: 1e-5)
            }
        }
    }

    func testPhraseBonusesFollowDepthScoring() throws {
        let encoder = try Self.encoder()
        let tree = PhraseBoostTree(phrases: [encoder.encode("Sæga")])
        let ids = encoder.encode("Sæga")  // ▁S æ ga
        var state = PhraseBoostTree.rootState
        var bonuses: [Float] = []
        for id in ids {
            let step = tree.step(from: state, token: id)
            bonuses.append(step.bonus)
            state = step.next
        }
        XCTAssertEqual(bonuses[0], 1, accuracy: 1e-6)
        XCTAssertEqual(bonuses[1], 2 + Float(log(2.0)), accuracy: 1e-6)
        XCTAssertEqual(bonuses[2], 2 + Float(log(3.0)), accuracy: 1e-6)

        // Falling out after "▁S æ" takes the accrued 1 + 2.69 back.
        let afterTwo = tree.step(from: tree.step(from: 0, token: ids[0]).next, token: ids[1]).next
        let out = tree.step(from: afterTwo, token: encoder.encode("x").last!)
        XCTAssertEqual(out.next, PhraseBoostTree.rootState)
        XCTAssertEqual(out.bonus, -(1 + 2 + Float(log(2.0))), accuracy: 1e-5)

        // A finished phrase takes nothing back.
        let done = tree.step(from: state, token: encoder.encode("x").last!)
        XCTAssertEqual(done.bonus, 0, accuracy: 1e-6)
    }

    // MARK: - Decoder hook

    func testHookLeavesBlankAndUnboostedStepsAlone() throws {
        let encoder = try Self.encoder()
        let boost = PhraseBoost(phrases: ["Sæga"], encoder: encoder, alpha: 1.0)
        let ids = encoder.encode("Sæga")
        let blank = 8192
        let saId = encoder.encodeWithPieces("Säga")[1].id  // ä after ▁S
        let state = boost.tree.step(from: 0, token: ids[0]).next

        // Blank argmax: never replaced, even with a boosted candidate in top-K.
        var label = blank
        var score: Float = 0.9
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [blank, ids[1]], topKLogits: [10, 9.9],
            state: state, blankId: blank)
        XCTAssertEqual(label, blank)
        XCTAssertEqual(score, 0.9)

        // No boost: unchanged.
        label = saId
        TdtDecoderV3.applyPhraseBoost(
            nil, label: &label, score: &score, topKIds: [saId, ids[1]], topKLogits: [10, 8],
            state: state, blankId: blank)
        XCTAssertEqual(label, saId)

        // After ▁S, æ gains 2.69 and ä (no phrase) loses the accrued 1: a lead of 2 flips.
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [saId, ids[1], blank], topKLogits: [10, 8, 7],
            state: state, blankId: blank)
        XCTAssertEqual(label, ids[1])
        XCTAssertLessThan(score, 0.5)

        // A lead of 4 does not (9 against 8.69).
        label = saId
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [saId, ids[1]], topKLogits: [10, 6],
            state: state, blankId: blank)
        XCTAssertEqual(label, saId)

        // A lead of 2 at alpha 0.5 does not either (9.5 against 9.35).
        let weak = PhraseBoost(tree: boost.tree, alpha: 0.5)
        TdtDecoderV3.applyPhraseBoost(
            weak, label: &label, score: &score, topKIds: [saId, ids[1]], topKLogits: [10, 8],
            state: state, blankId: blank)
        XCTAssertEqual(label, saId)
    }

    func testBoostNeverBringsBackATokenTheScriptFilterRemoved() throws {
        let encoder = try Self.encoder()
        let boost = PhraseBoost(phrases: ["Sæga"], encoder: encoder, alpha: 1.0)
        let blank = 8192
        let cyrillic = 9001
        let latin = 9002
        let vocabulary = [cyrillic: "\u{2581}при", latin: "\u{2581}pri"]

        // The joint's top-1 is Cyrillic; the Polish script filter has already picked the Latin
        // runner-up. With no phrase in play, boosting must keep the filter's choice.
        var label = latin
        var score: Float = 0.3
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [cyrillic, latin, blank], topKLogits: [10, 8, 7],
            state: PhraseBoostTree.rootState, blankId: blank, language: .polish, vocabulary: vocabulary)
        XCTAssertEqual(label, latin)
        XCTAssertEqual(score, 0.3)

        // A boosted phrase token that passes the filter still wins over the filtered label.
        let sId = encoder.encode("Sæga")[0]
        var withPhrase = vocabulary
        withPhrase[sId] = "\u{2581}S"
        label = latin
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [cyrillic, latin, sId], topKLogits: [10, 8, 7.5],
            state: PhraseBoostTree.rootState, blankId: blank, language: .polish, vocabulary: withPhrase)
        XCTAssertEqual(label, sId)
    }

    func testBoostNeverBringsBackAFrenchBlocklistedToken() throws {
        let encoder = try Self.encoder()
        let boost = PhraseBoost(phrases: ["Sæga"], encoder: encoder, alpha: 1.0)
        let blank = 8192
        let the = 506  // ' the', in TdtDecoderV3.englishBlocklistIds
        let le = 9003
        let vocabulary = [the: "\u{2581}the", le: "\u{2581}le"]

        var label = le
        var score: Float = 0.2
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [the, le], topKLogits: [10, 8],
            state: PhraseBoostTree.rootState, blankId: blank, language: .french, vocabulary: vocabulary)
        XCTAssertEqual(label, le)

        // Outside French the same id is ordinary vocabulary and stays eligible.
        TdtDecoderV3.applyPhraseBoost(
            boost, label: &label, score: &score, topKIds: [the, le], topKLogits: [10, 8],
            state: PhraseBoostTree.rootState, blankId: blank, language: .english, vocabulary: vocabulary)
        XCTAssertEqual(label, the)
    }

    func testDecoderStateCarriesAndResetsTreeState() throws {
        var state = try TdtDecoderState()
        XCTAssertEqual(state.phraseBoostState, PhraseBoostTree.rootState)
        state.phraseBoostState = 7
        let copy = try TdtDecoderState(from: state)
        XCTAssertEqual(copy.phraseBoostState, 7)
        state.reset()
        XCTAssertEqual(state.phraseBoostState, PhraseBoostTree.rootState)
    }
}
