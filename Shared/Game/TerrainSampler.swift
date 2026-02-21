import Foundation
import Metal
import simd

/// CPU-side neural network evaluator that mirrors the GPU terrain shader.
/// Used to sample terrain height at arbitrary positions for camera following,
/// collision, and gameplay logic without GPU readback.
final class TerrainSampler {
    private let weightsBuffer: MTLBuffer
    private let weightCount: Int

    // Network dimensions (must match ShaderTypes.h)
    private let inputSize = 16
    private let hidden1Size = 32
    private let hidden2Size = 32
    private let outputSize = 4
    private let posEncodeBands = 3
    private let posEncodePerCoord = 7  // raw + 2*bands

    init(weightsBuffer: MTLBuffer) {
        self.weightsBuffer = weightsBuffer
        self.weightCount = 1732
    }

    /// Sample terrain height at a world (x, z) position.
    /// Replicates the GPU's evaluateTerrainHeight function exactly.
    func heightAt(x: Float, z: Float, time: Float, playerInfluence: Float = 0) -> Float {
        let weights = weightsBuffer.contents().bindMemory(to: Float.self, capacity: weightCount)

        // Build input with positional encoding + time-varying phase shift
        var input = [Float](repeating: 0, count: inputSize)
        let phase = time * 0.15

        var idx = 0
        idx = positionalEncode(coord: x * 0.15 + sin(phase) * 0.3, output: &input, startIdx: idx)
        idx = positionalEncode(coord: z * 0.15 + cos(phase * 0.7) * 0.3, output: &input, startIdx: idx)
        input[idx] = sin(time * 0.4) * cos(time * 0.17)
        idx += 1
        input[idx] = playerInfluence

        // Hidden layer 1: 16 -> 32 (ReLU)
        var hidden1 = [Float](repeating: 0, count: hidden1Size)
        var offset = denseLayerReLU(input: input, inputSize: inputSize,
                                     outputSize: hidden1Size, weights: weights,
                                     offset: 0, output: &hidden1)

        // Hidden layer 2: 32 -> 32 (ReLU)
        var hidden2 = [Float](repeating: 0, count: hidden2Size)
        offset = denseLayerReLU(input: hidden1, inputSize: hidden1Size,
                                 outputSize: hidden2Size, weights: weights,
                                 offset: offset, output: &hidden2)

        // Output layer: 32 -> 4 (Linear)
        var result = [Float](repeating: 0, count: outputSize)
        _ = denseLayerLinear(input: hidden2, inputSize: hidden2Size,
                             outputSize: outputSize, weights: weights,
                             offset: offset, output: &result)

        return tanh(result[0]) * 4.0  // Match GPU: tanh keeps range [-4, 4]
    }

    /// Sample height with smoothing (average of nearby points for stable camera)
    func smoothHeightAt(x: Float, z: Float, time: Float, radius: Float = 0.5) -> Float {
        let h0 = heightAt(x: x, z: z, time: time)
        let h1 = heightAt(x: x + radius, z: z, time: time)
        let h2 = heightAt(x: x - radius, z: z, time: time)
        let h3 = heightAt(x: x, z: z + radius, time: time)
        let h4 = heightAt(x: x, z: z - radius, time: time)
        return (h0 * 2.0 + h1 + h2 + h3 + h4) / 6.0
    }

    // MARK: - Private neural network operations

    private func positionalEncode(coord: Float, output: inout [Float], startIdx: Int) -> Int {
        output[startIdx] = coord
        var idx = startIdx + 1
        var freq: Float = 1.0
        for _ in 0..<posEncodeBands {
            output[idx]     = sin(freq * coord)
            output[idx + 1] = cos(freq * coord)
            idx += 2
            freq *= 2.0
        }
        return idx
    }

    private func denseLayerReLU(input: [Float], inputSize: Int, outputSize: Int,
                                 weights: UnsafePointer<Float>, offset: Int,
                                 output: inout [Float]) -> Int {
        for o in 0..<outputSize {
            var sum: Float = 0
            for i in 0..<inputSize {
                sum += input[i] * weights[offset + o * inputSize + i]
            }
            sum += weights[offset + outputSize * inputSize + o]  // bias
            output[o] = max(0, sum)  // ReLU
        }
        return offset + outputSize * inputSize + outputSize
    }

    private func denseLayerLinear(input: [Float], inputSize: Int, outputSize: Int,
                                   weights: UnsafePointer<Float>, offset: Int,
                                   output: inout [Float]) -> Int {
        for o in 0..<outputSize {
            var sum: Float = 0
            for i in 0..<inputSize {
                sum += input[i] * weights[offset + o * inputSize + i]
            }
            sum += weights[offset + outputSize * inputSize + o]  // bias
            output[o] = sum
        }
        return offset + outputSize * inputSize + outputSize
    }
}
