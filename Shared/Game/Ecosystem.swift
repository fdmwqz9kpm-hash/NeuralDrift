import Foundation
import simd

/// Simulates a living ecosystem of neural-network-driven creatures.
/// Handles spawning, brain evaluation, reproduction, death, and species tracking.
final class Ecosystem: ObservableObject {
    @Published var creatures: [Creature] = []
    @Published var generation: Int = 0
    @Published var totalBorn: Int = 0
    @Published var totalDied: Int = 0
    @Published var speciesCounts: [Int: Int] = [:]  // species ID → count

    let maxCreatures = 80
    let initialCount = 20
    let reproductionThreshold: Float = 1.5
    let gridSpacing: Float
    let gridSize: Int
    private var tickAccumulator: Float = 0
    private let tickInterval: Float = 1.0 / 20.0  // 20 Hz simulation

    // Species colors (8 possible species)
    static let speciesColors: [SIMD3<Float>] = [
        SIMD3<Float>(0.2, 0.8, 1.0),   // Cyan
        SIMD3<Float>(1.0, 0.4, 0.2),   // Orange
        SIMD3<Float>(0.3, 1.0, 0.3),   // Green
        SIMD3<Float>(1.0, 0.2, 0.8),   // Pink
        SIMD3<Float>(1.0, 1.0, 0.2),   // Yellow
        SIMD3<Float>(0.5, 0.3, 1.0),   // Purple
        SIMD3<Float>(0.2, 1.0, 0.7),   // Mint
        SIMD3<Float>(1.0, 0.6, 0.4),   // Peach
    ]

    private var terrainSampler: TerrainSampler?

    init(gridSize: Int, gridSpacing: Float) {
        self.gridSize = gridSize
        self.gridSpacing = gridSpacing
    }

    func setTerrainSampler(_ sampler: TerrainSampler) {
        self.terrainSampler = sampler
    }

    /// Spawn the initial population
    func spawn() {
        let halfExtent = Float(gridSize) * gridSpacing * 0.35
        creatures.removeAll()

        for _ in 0..<initialCount {
            let brain = CreatureBrain.random()
            let species = brain.speciesHash
            let pos = SIMD2<Float>(
                Float.random(in: -halfExtent...halfExtent),
                Float.random(in: -halfExtent...halfExtent)
            )
            let creature = Creature(
                position: pos,
                heading: Float.random(in: 0...(2 * .pi)),
                species: species,
                speciesColor: Ecosystem.speciesColors[species % Ecosystem.speciesColors.count],
                brain: brain
            )
            creatures.append(creature)
            totalBorn += 1
        }
        updateSpeciesCounts()
    }

    /// Run one simulation tick (throttled to 20Hz)
    func update(playerPosition: SIMD3<Float>, isInteracting: Bool, time: Float, deltaTime: Float) {
        guard deltaTime > 0, deltaTime < 0.1 else { return }

        tickAccumulator += deltaTime
        guard tickAccumulator >= tickInterval else { return }
        let simDT = tickAccumulator
        tickAccumulator = 0

        let halfExtent = Float(gridSize) * gridSpacing * 0.4
        var newCreatures: [Creature] = []

        // Precompute spatial data for nearby creature detection
        let positions = creatures.map { $0.position }

        for i in 0..<creatures.count {
            guard creatures[i].isAlive else { continue }

            // Compute brain inputs
            let pos = creatures[i].position
            let heading = creatures[i].heading

            // Approximate slope from position hash (avoids CPU neural net eval)
            let slope = sin(pos.x * 0.5 + time * 0.1) * cos(pos.y * 0.5) * 0.3

            // Nearby creature density (within radius 3)
            var nearCount: Float = 0
            var nearestDist: Float = 999
            var nearestAngle: Float = 0
            for j in 0..<positions.count where j != i {
                let d = simd_length(positions[j] - pos)
                if d < 3.0 {
                    nearCount += 1
                    if d < nearestDist {
                        nearestDist = d
                        let toNearest = positions[j] - pos
                        let angle = atan2(toNearest.y, toNearest.x) - heading
                        nearestAngle = sin(angle) // -1 to 1
                    }
                }
            }
            let density = min(nearCount / 8.0, 1.0)

            // Player distance
            let playerDist2D = simd_length(SIMD2<Float>(playerPosition.x, playerPosition.z) - pos)
            let playerNorm = min(playerDist2D / 20.0, 1.0)

            // Food: procedural based on position (avoids CPU neural net eval)
            let foodNoise = sin(pos.x * 0.3 + time * 0.05) * cos(pos.y * 0.4 - time * 0.03)
            var foodValue: Float = max(0, foodNoise * 0.5 + 0.3)

            // Player interaction creates food burst nearby
            if isInteracting && playerDist2D < 8.0 {
                foodValue += (1.0 - playerDist2D / 8.0) * 0.5
            }

            let inputs = BrainInputs(
                terrainSlope: max(-1, min(1, slope)),
                nearbyDensity: density,
                playerDistance: playerNorm,
                energyLevel: creatures[i].energy,
                foodValue: foodValue,
                nearestCreatureAngle: nearestAngle
            )

            creatures[i].step(inputs: inputs, deltaTime: simDT)

            // Boundary wrapping
            if creatures[i].position.x > halfExtent { creatures[i].position.x = -halfExtent }
            if creatures[i].position.x < -halfExtent { creatures[i].position.x = halfExtent }
            if creatures[i].position.y > halfExtent { creatures[i].position.y = -halfExtent }
            if creatures[i].position.y < -halfExtent { creatures[i].position.y = halfExtent }

            // Reproduction
            if creatures[i].energy > reproductionThreshold && creatures.count + newCreatures.count < maxCreatures {
                creatures[i].energy -= 0.8  // Cost of reproduction

                let childBrain = creatures[i].brain.mutated(rate: 0.15)
                let childSpecies = childBrain.speciesHash
                let offset = SIMD2<Float>(Float.random(in: -0.5...0.5),
                                           Float.random(in: -0.5...0.5))
                let child = Creature(
                    position: pos + offset,
                    heading: Float.random(in: 0...(2 * .pi)),
                    generation: creatures[i].generation + 1,
                    species: childSpecies,
                    speciesColor: Ecosystem.speciesColors[childSpecies % Ecosystem.speciesColors.count],
                    brain: childBrain
                )
                newCreatures.append(child)
                totalBorn += 1
                generation = max(generation, child.generation)
            }
        }

        // Remove dead creatures
        let beforeCount = creatures.count
        creatures.removeAll { !$0.isAlive }
        totalDied += beforeCount - creatures.count

        // Add newborns
        creatures.append(contentsOf: newCreatures)

        // Minimum population maintenance (prevent extinction)
        if creatures.count < 10 {
            for _ in 0..<5 {
                let brain = CreatureBrain.random()
                let species = brain.speciesHash
                let pos = SIMD2<Float>(
                    Float.random(in: -halfExtent...halfExtent),
                    Float.random(in: -halfExtent...halfExtent)
                )
                creatures.append(Creature(
                    position: pos,
                    heading: Float.random(in: 0...(2 * .pi)),
                    energy: 1.0,
                    generation: generation,
                    species: species,
                    speciesColor: Ecosystem.speciesColors[species % Ecosystem.speciesColors.count],
                    brain: brain
                ))
                totalBorn += 1
            }
        }

        updateSpeciesCounts()
    }

    private func updateSpeciesCounts() {
        var counts: [Int: Int] = [:]
        for c in creatures where c.isAlive {
            counts[c.species, default: 0] += 1
        }
        speciesCounts = counts
    }

    /// Biodiversity score: Shannon entropy of species distribution
    var biodiversityScore: Float {
        let total = Float(creatures.count)
        guard total > 0 else { return 0 }
        var entropy: Float = 0
        for (_, count) in speciesCounts {
            let p = Float(count) / total
            if p > 0 {
                entropy -= p * log2(p)
            }
        }
        return entropy
    }
}
