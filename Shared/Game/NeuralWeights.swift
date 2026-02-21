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

    static let terrainCount = 1732  // Matches TERRAIN_WEIGHT_COUNT in ShaderTypes.h
    static let colorCount = 1371    // Matches COLOR_WEIGHT_COUNT in ShaderTypes.h

    init(device: MTLDevice) {
        let terrainByteSize = NeuralWeights.terrainCount * MemoryLayout<Float>.size
        let colorByteSize = NeuralWeights.colorCount * MemoryLayout<Float>.size

        // Generate initial weights — boosted scale for dramatic terrain
        var terrainData = NeuralWeights.generateInitialWeights(
            count: NeuralWeights.terrainCount,
            layerSizes: [(16, 32), (32, 32), (32, 4)],
            scale: 2.0  // Large scale → dramatic ridges and valleys
        )
        // Bias the output layer toward interesting height range
        let terrainOutputBiasStart = 16*32+32 + 32*32+32 + 32*4 // Start of output biases
        terrainData[terrainOutputBiasStart] = 0.5     // Height bias: slightly above zero

        var colorData = NeuralWeights.generateInitialWeights(
            count: NeuralWeights.colorCount,
            layerSizes: [(28, 24), (24, 24), (24, 3)],
            scale: 1.5  // Vivid initial colors
        )
        // Bias color output for warm earth tones
        let colorOutputBiasStart = 28*24+24 + 24*24+24 + 24*3
        colorData[colorOutputBiasStart]     =  0.3   // R bias
        colorData[colorOutputBiasStart + 1] = -0.2   // G bias (will produce warm greens via sigmoid)
        colorData[colorOutputBiasStart + 2] = -0.5   // B bias (less blue → earthy)

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
        layerSizes: [(Int, Int)],
        scale: Float = 1.0
    ) -> [Float] {
        var weights = [Float](repeating: 0, count: count)
        var offset = 0

        for (fanIn, fanOut) in layerSizes {
            // He initialization (good for ReLU): sqrt(2/fanIn) * scale
            let layerScale = sqrt(2.0 / Float(fanIn)) * scale

            // Weights
            for i in 0..<(fanIn * fanOut) {
                weights[offset + i] = NeuralWeights.gaussianRandom() * layerScale
            }
            offset += fanIn * fanOut

            // Biases: small random to break symmetry
            for i in 0..<fanOut {
                weights[offset + i] = Float.random(in: -0.05...0.05)
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
