import Foundation
import Metal

/// Manages neural network weight buffers on the GPU.
/// Handles initialization with Xavier-like random weights and provides
/// both mutable (current) and immutable (initial) copies for decay.
final class NeuralWeights {
    let terrainWeights: MTLBuffer
    let terrainInitialWeights: MTLBuffer
    let colorWeights: MTLBuffer
    let colorInitialWeights: MTLBuffer

    static let terrainCount = 1348  // Matches TERRAIN_WEIGHT_COUNT in ShaderTypes.h
    static let colorCount = 499     // Matches COLOR_WEIGHT_COUNT in ShaderTypes.h

    init(device: MTLDevice) {
        let terrainByteSize = NeuralWeights.terrainCount * MemoryLayout<Float>.size
        let colorByteSize = NeuralWeights.colorCount * MemoryLayout<Float>.size

        // Generate initial weights using Xavier-like initialization
        var terrainData = NeuralWeights.generateInitialWeights(
            count: NeuralWeights.terrainCount,
            layerSizes: [(4, 32), (32, 32), (32, 4)]
        )
        var colorData = NeuralWeights.generateInitialWeights(
            count: NeuralWeights.colorCount,
            layerSizes: [(10, 16), (16, 16), (16, 3)]
        )

        // Create GPU buffers
        terrainWeights = device.makeBuffer(
            bytes: &terrainData,
            length: terrainByteSize,
            options: .storageModeShared
        )!
        terrainWeights.label = "Terrain Neural Weights"

        terrainInitialWeights = device.makeBuffer(
            bytes: &terrainData,
            length: terrainByteSize,
            options: .storageModeShared
        )!
        terrainInitialWeights.label = "Terrain Initial Weights"

        colorWeights = device.makeBuffer(
            bytes: &colorData,
            length: colorByteSize,
            options: .storageModeShared
        )!
        colorWeights.label = "Color Neural Weights"

        colorInitialWeights = device.makeBuffer(
            bytes: &colorData,
            length: colorByteSize,
            options: .storageModeShared
        )!
        colorInitialWeights.label = "Color Initial Weights"
    }

    /// Xavier initialization: weights ~ N(0, sqrt(2 / (fanIn + fanOut)))
    /// Biases initialized to small positive values for ReLU networks
    private static func generateInitialWeights(
        count: Int,
        layerSizes: [(Int, Int)]
    ) -> [Float] {
        var weights = [Float](repeating: 0, count: count)
        var offset = 0

        for (fanIn, fanOut) in layerSizes {
            let scale = sqrt(2.0 / Float(fanIn + fanOut))

            // Weights
            for i in 0..<(fanIn * fanOut) {
                weights[offset + i] = NeuralWeights.gaussianRandom() * scale
            }
            offset += fanIn * fanOut

            // Biases (small positive for ReLU, zero for tanh/linear)
            for i in 0..<fanOut {
                weights[offset + i] = 0.01
            }
            offset += fanOut
        }

        return weights
    }

    /// Box-Muller transform for Gaussian random numbers
    private static func gaussianRandom() -> Float {
        let u1 = Float.random(in: 0.0001...1.0)
        let u2 = Float.random(in: 0.0001...1.0)
        return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }

    /// Reset weights to initial state
    func reset() {
        let terrainByteSize = NeuralWeights.terrainCount * MemoryLayout<Float>.size
        let colorByteSize = NeuralWeights.colorCount * MemoryLayout<Float>.size

        memcpy(terrainWeights.contents(),
               terrainInitialWeights.contents(),
               terrainByteSize)
        memcpy(colorWeights.contents(),
               colorInitialWeights.contents(),
               colorByteSize)
    }
}
