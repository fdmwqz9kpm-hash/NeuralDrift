import Foundation
import simd

/// A single creature with a tiny neural network brain.
/// Brain inputs: [terrain slope, nearby density, player distance, energy level]
/// Brain outputs: [turn angle, speed multiplier]
struct Creature {
    var position: SIMD2<Float>       // XZ world position
    var heading: Float               // Radians
    var speed: Float = 0.5
    var energy: Float = 1.0          // 0 = dead, reproduces above threshold
    var age: Float = 0
    var generation: Int = 0
    var species: Int = 0             // Species ID (derived from brain hash)
    var speciesColor: SIMD3<Float>   // Visual color
    var brain: CreatureBrain

    var isAlive: Bool { energy > 0 }

    /// Advance one simulation step
    mutating func step(inputs: BrainInputs, deltaTime: Float) {
        age += deltaTime

        // Run neural brain
        let outputs = brain.evaluate(inputs: inputs)
        var turnRate = outputs.turn * 3.0   // Max ±3 rad/sec
        var speedMult = 0.3 + outputs.speed * 0.7  // 0.3 to 1.0

        // Flocking: gently steer toward nearby same-species heading
        turnRate += inputs.flockHeading * 1.5

        // Panic: scatter away from mutation shockwave
        if inputs.panic > 0.1 {
            speedMult = min(speedMult + inputs.panic * 0.8, 1.5)
            turnRate += (Float.random(in: -1...1)) * inputs.panic * 5.0
        }

        heading += turnRate * deltaTime
        speed = speedMult * 2.5

        // Move
        let dx = cos(heading) * speed * deltaTime
        let dz = sin(heading) * speed * deltaTime
        position.x += dx
        position.y += dz

        // Energy cost: movement + base metabolism
        energy -= (0.025 + speed * 0.008) * deltaTime

        // Energy from terrain (food patches at certain heights)
        energy += inputs.foodValue * 0.15 * deltaTime

        energy = min(energy, 2.0)
    }
}

/// Inputs to the creature's neural brain
struct BrainInputs {
    var terrainSlope: Float      // -1 to 1, steepness in heading direction
    var nearbyDensity: Float     // 0 to 1, how crowded nearby
    var playerDistance: Float     // 0 to 1, normalized (0=close, 1=far)
    var energyLevel: Float       // 0 to 2, creature's current energy
    var foodValue: Float         // 0 to 1, food availability at position
    var nearestCreatureAngle: Float  // -1 to 1, relative angle to nearest
    var flockHeading: Float      // -1 to 1, average heading of nearby same-species
    var panic: Float             // 0 to 1, mutation shockwave nearby
}

/// Brain outputs
struct BrainOutputs {
    var turn: Float   // -1 to 1
    var speed: Float  // 0 to 1
}

/// Tiny neural network: 8 inputs → 8 hidden (tanh) → 2 outputs (tanh)
struct CreatureBrain {
    // Layer 1: 8×8 weights + 8 biases = 72
    var w1: [Float]  // 64 weights
    var b1: [Float]  // 8 biases
    // Layer 2: 8×2 weights + 2 biases = 18
    var w2: [Float]  // 16 weights
    var b2: [Float]  // 2 biases

    static let inputSize = 8
    static let hiddenSize = 8
    static let outputSize = 2
    static let totalWeights = 8*8 + 8 + 8*2 + 2  // 90

    /// Random initialization
    static func random() -> CreatureBrain {
        CreatureBrain(
            w1: (0..<64).map { _ in Float.random(in: -1...1) },
            b1: (0..<8).map { _ in Float.random(in: -0.3...0.3) },
            w2: (0..<16).map { _ in Float.random(in: -1...1) },
            b2: (0..<2).map { _ in Float.random(in: -0.2...0.2) }
        )
    }

    /// Forward pass
    func evaluate(inputs: BrainInputs) -> BrainOutputs {
        let x: [Float] = [
            inputs.terrainSlope,
            inputs.nearbyDensity,
            inputs.playerDistance,
            inputs.energyLevel,
            inputs.foodValue,
            inputs.nearestCreatureAngle,
            inputs.flockHeading,
            inputs.panic
        ]

        // Hidden layer (tanh)
        var hidden = [Float](repeating: 0, count: CreatureBrain.hiddenSize)
        for h in 0..<CreatureBrain.hiddenSize {
            var sum = b1[h]
            for i in 0..<CreatureBrain.inputSize {
                sum += x[i] * w1[h * CreatureBrain.inputSize + i]
            }
            hidden[h] = tanh(sum)
        }

        // Output layer (tanh)
        var output = [Float](repeating: 0, count: CreatureBrain.outputSize)
        for o in 0..<CreatureBrain.outputSize {
            var sum = b2[o]
            for h in 0..<CreatureBrain.hiddenSize {
                sum += hidden[h] * w2[o * CreatureBrain.hiddenSize + h]
            }
            output[o] = tanh(sum)
        }

        return BrainOutputs(turn: output[0], speed: (output[1] + 1.0) * 0.5)
    }

    /// Create offspring with mutated weights
    func mutated(rate: Float = 0.1) -> CreatureBrain {
        func mutate(_ arr: [Float]) -> [Float] {
            arr.map { w in
                if Float.random(in: 0...1) < rate {
                    return w + Float.random(in: -0.5...0.5)
                }
                return w
            }
        }
        return CreatureBrain(
            w1: mutate(w1), b1: mutate(b1),
            w2: mutate(w2), b2: mutate(b2)
        )
    }

    /// Hash brain weights to determine species
    var speciesHash: Int {
        var hash = 0
        // Sample a few weights to determine species identity
        for i in stride(from: 0, to: w1.count, by: 6) {
            hash ^= Int(w1[i] * 1000) &* 2654435761
        }
        return abs(hash) % 8  // 8 possible species
    }
}
