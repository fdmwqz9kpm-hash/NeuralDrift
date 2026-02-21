import Foundation
import simd

/// Detects "resonance points" — locations where neural weight patterns create
/// aesthetically interesting, stable terrain features. These become collectible orbs.
final class ResonanceDetector: ObservableObject {
    @Published var orbs: [ResonanceOrb] = []
    @Published var capturedCount: Int = 0
    @Published var totalScore: Int = 0
    @Published var nearbyOrb: ResonanceOrb? = nil

    private let maxOrbs = 5
    private let captureRadius: Float = 3.0
    private let detectionInterval: TimeInterval = 4.0
    private let orbLifetime: Float = 30.0
    private var lastDetectionTime: CFAbsoluteTime = 0
    private var weightHistory: [Float] = []
    private let historyLength = 8

    /// Sample terrain weights and detect resonance patterns.
    /// Call from the game loop with current weight buffer contents.
    func update(terrainWeights: UnsafePointer<Float>,
                weightCount: Int,
                playerPosition: SIMD3<Float>,
                totalTime: Float,
                gridSpacing: Float,
                gridSize: Int) {

        let now = CFAbsoluteTimeGetCurrent()

        // Age and remove expired orbs
        orbs.removeAll { $0.spawnTime + orbLifetime < totalTime }

        // Check proximity to existing orbs
        nearbyOrb = nil
        var closestDist: Float = .infinity
        for orb in orbs where !orb.captured {
            let dist = simd_length(playerPosition - orb.position)
            if dist < captureRadius && dist < closestDist {
                closestDist = dist
                nearbyOrb = orb
            }
        }

        // Detect new resonance points periodically
        if now - lastDetectionTime > detectionInterval && orbs.count < maxOrbs {
            lastDetectionTime = now

            // Compute weight statistics for resonance detection
            let stats = analyzeWeights(terrainWeights, count: weightCount)
            weightHistory.append(stats.variance)
            if weightHistory.count > historyLength {
                weightHistory.removeFirst()
            }

            // Resonance = weights have stabilized into a non-trivial pattern
            // (low recent variance change + interesting spatial structure)
            if weightHistory.count >= 4 {
                let recentChange = abs(weightHistory.last! - weightHistory[weightHistory.count - 3])
                let isStable = recentChange < stats.variance * 0.15
                let isInteresting = stats.variance > 0.1 && stats.peakSpread > 0.3

                if isStable && isInteresting {
                    spawnOrb(stats: stats, playerPos: playerPosition, time: totalTime,
                             gridSpacing: gridSpacing, gridSize: gridSize,
                             weights: terrainWeights, weightCount: weightCount)
                }
            }
        }
    }

    /// Capture the nearest orb if in range. Returns the score earned.
    @discardableResult
    func captureNearestOrb() -> Int {
        guard let orb = nearbyOrb, let idx = orbs.firstIndex(where: { $0.id == orb.id && !$0.captured }) else {
            return 0
        }
        orbs[idx].captured = true
        capturedCount += 1

        // Score based on uniqueness (hash of orb properties)
        let score = computeScore(for: orbs[idx])
        totalScore += score
        nearbyOrb = nil
        return score
    }

    // MARK: - Private

    private struct WeightStats {
        var mean: Float
        var variance: Float
        var peakSpread: Float  // Max - min across sampled weights
        var spectralEnergy: Float  // Low-frequency energy (smoothness indicator)
    }

    private func analyzeWeights(_ weights: UnsafePointer<Float>, count: Int) -> WeightStats {
        let sampleStride = max(1, count / 128) // Sample ~128 weights
        var sum: Float = 0
        var sumSq: Float = 0
        var minW: Float = .infinity
        var maxW: Float = -.infinity
        var n: Float = 0

        // Low-frequency energy: compare adjacent sampled weights
        var smoothSum: Float = 0
        var prevW: Float = weights[0]

        for i in stride(from: 0, to: count, by: sampleStride) {
            let w = weights[i]
            sum += w
            sumSq += w * w
            minW = min(minW, w)
            maxW = max(maxW, w)
            smoothSum += abs(w - prevW)
            prevW = w
            n += 1
        }

        let mean = sum / max(n, 1)
        let variance = (sumSq / max(n, 1)) - mean * mean
        let spectralEnergy = 1.0 / (smoothSum / max(n, 1) + 0.01)

        return WeightStats(mean: mean, variance: max(0, variance),
                           peakSpread: maxW - minW,
                           spectralEnergy: spectralEnergy)
    }

    private func spawnOrb(stats: WeightStats, playerPos: SIMD3<Float>, time: Float,
                          gridSpacing: Float, gridSize: Int,
                          weights: UnsafePointer<Float>, weightCount: Int) {
        let halfExtent = Float(gridSize) * gridSpacing * 0.5

        // Place orb at a position derived from weight pattern — not random,
        // but deterministic from the neural state
        let wx = weights[min(42, weightCount - 1)]
        let wz = weights[min(137, weightCount - 1)]
        let angle = atan2(wx, wz + 0.001)
        let radius = (stats.variance * 15.0 + 5.0)

        var orbX = playerPos.x + cos(angle) * radius
        var orbZ = playerPos.z + sin(angle) * radius
        orbX = max(-halfExtent + 2, min(halfExtent - 2, orbX))
        orbZ = max(-halfExtent + 2, min(halfExtent - 2, orbZ))

        // Don't spawn too close to existing orbs
        let newPos = SIMD3<Float>(orbX, 2.0, orbZ) // Y will be set by shader
        for existing in orbs where !existing.captured {
            if simd_length(existing.position - newPos) < 8.0 { return }
        }

        let orb = ResonanceOrb(
            position: newPos,
            color: colorFromStats(stats),
            intensity: min(stats.spectralEnergy * 0.3, 2.0),
            spawnTime: time,
            worldHash: hashWorld(weights: weights, count: weightCount)
        )
        orbs.append(orb)
    }

    private func colorFromStats(_ stats: WeightStats) -> SIMD3<Float> {
        // Map weight statistics to a distinctive orb color
        let hue = fmod(stats.mean * 3.0 + stats.variance * 7.0, 1.0)
        let sat: Float = 0.7
        let val: Float = 1.0
        return hsvToRGB(h: abs(hue), s: sat, v: val)
    }

    private func hsvToRGB(h: Float, s: Float, v: Float) -> SIMD3<Float> {
        let c = v * s
        let x = c * (1 - abs(fmod(h * 6, 2) - 1))
        let m = v - c
        var r: Float = 0, g: Float = 0, b: Float = 0
        let sector = Int(h * 6) % 6
        switch sector {
        case 0: r = c; g = x; b = 0
        case 1: r = x; g = c; b = 0
        case 2: r = 0; g = c; b = x
        case 3: r = 0; g = x; b = c
        case 4: r = x; g = 0; b = c
        default: r = c; g = 0; b = x
        }
        return SIMD3<Float>(r + m, g + m, b + m)
    }

    private func computeScore(for orb: ResonanceOrb) -> Int {
        // Score based on world hash uniqueness + orb properties
        var hashValue = orb.worldHash
        hashValue ^= UInt64(orb.intensity * 1000)
        hashValue = hashValue &* 6364136223846793005 &+ 1442695040888963407
        return Int(100 + (hashValue % 900)) // 100-999 points per capture
    }

    private func hashWorld(weights: UnsafePointer<Float>, count: Int) -> UInt64 {
        // FNV-1a hash of sampled weights for world fingerprinting
        var hash: UInt64 = 14695981039346656037
        let stride = max(1, count / 64)
        for i in Swift.stride(from: 0, to: count, by: stride) {
            let bits = weights[i].bitPattern
            hash ^= UInt64(bits)
            hash = hash &* 1099511628211
        }
        return hash
    }
}

/// A collectible resonance orb that appears at aesthetically interesting terrain points.
struct ResonanceOrb: Identifiable {
    let id = UUID()
    var position: SIMD3<Float>
    var color: SIMD3<Float>
    var intensity: Float
    var spawnTime: Float
    var worldHash: UInt64
    var captured: Bool = false
}
