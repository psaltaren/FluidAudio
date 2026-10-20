import XCTest

@testable import FluidAudio

final class ChunkSpeakerWeightsTests: XCTestCase {

    func testReadsBackEveryChunkAsStored() {
        let chunks: [[[Float]]] = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[1, 0, 0]],
            [],
        ]
        let weights = ChunkSpeakerWeights(chunks)
        XCTAssertEqual(weights.count, 3)
        XCTAssertEqual(Array(weights), chunks)
    }

    func testFramesWithoutSpeakersKeepTheirFrameCount() {
        let weights: ChunkSpeakerWeights = [[[], [], []]]
        XCTAssertEqual(weights[0], [[], [], []])
    }

    func testSegmentationOutputKeepsNoLogProbabilitiesButEveryWeight() {
        let output = SegmentationOutput(
            logProbs: [], speakerWeights: [[[0.5, 0.5]], [[1, 0]]],
            numChunks: 2, numFrames: 1, numSpeakers: 2)
        XCTAssertEqual(output.speakerWeights[1], [[1, 0]])
    }
}
