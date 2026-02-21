import Foundation
import simd

/// Central game state — owns the camera, player position, and interaction state.
/// Platform-agnostic; input handlers update this via the InputAction protocol.
final class GameState: ObservableObject {
    // Camera — first-person, terrain-following
    @Published var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 2, 5)
    @Published var cameraYaw: Float = 0             // Radians, horizontal rotation
    @Published var cameraPitch: Float = -0.05        // Slightly looking down

    // Movement
    var moveForward: Float = 0   // -1 to 1
    var moveRight: Float = 0     // -1 to 1
    let moveSpeed: Float = 6.0

    // Interaction
    @Published var isInteracting: Bool = false
    var interactionStrength: Float = 1.0
    let influenceRadius: Float = 5.0

    // Time
    var totalTime: Float = 0
    var deltaTime: Float = 0
    private var lastUpdateTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    // Grid configuration
    let gridSize: Int = 256
    let gridSpacing: Float = 0.25

    // Camera constraints
    let pitchMin: Float = -.pi / 2.2   // Can look almost straight down
    let pitchMax: Float = .pi / 2.2    // Can look almost straight up
    let eyeHeight: Float = 2.0         // Height above terrain surface

    // Terrain following
    var terrainSampler: TerrainSampler?
    private var smoothedTerrainHeight: Float = 0
    private let heightSmoothFactor: Float = 6.0  // How fast camera follows terrain

    // Head bob
    private var walkCycle: Float = 0
    private let bobAmplitude: Float = 0.06
    private let bobFrequency: Float = 8.0

    // Camera shake (for mutation feedback)
    private(set) var shakeOffset: SIMD3<Float> = .zero
    private var shakeIntensity: Float = 0
    private let shakeDecay: Float = 8.0

    func update() {
        let now = CFAbsoluteTimeGetCurrent()
        deltaTime = Float(now - lastUpdateTime)
        deltaTime = min(deltaTime, 1.0 / 30.0)
        lastUpdateTime = now
        totalTime += deltaTime

        // Calculate forward and right vectors from yaw (horizontal only)
        let forward = SIMD3<Float>(sin(cameraYaw), 0, -cos(cameraYaw))
        let right = SIMD3<Float>(cos(cameraYaw), 0, sin(cameraYaw))

        // Apply movement
        let speed = moveSpeed * deltaTime
        let velocity = (forward * moveForward + right * moveRight) * speed
        cameraPosition += velocity

        // Clamp to grid bounds
        let halfExtent = Float(gridSize) * gridSpacing * 0.45
        cameraPosition.x = max(-halfExtent, min(halfExtent, cameraPosition.x))
        cameraPosition.z = max(-halfExtent, min(halfExtent, cameraPosition.z))

        // Terrain following — sample height at camera XZ and smoothly track it
        if let sampler = terrainSampler {
            let targetHeight = sampler.smoothHeightAt(
                x: cameraPosition.x, z: cameraPosition.z, time: totalTime)
            smoothedTerrainHeight += (targetHeight - smoothedTerrainHeight)
                * min(1.0, heightSmoothFactor * deltaTime)
        }

        // Head bob during movement
        let isMoving = abs(moveForward) > 0.1 || abs(moveRight) > 0.1
        if isMoving {
            walkCycle += deltaTime * bobFrequency
        } else {
            walkCycle *= 0.9  // Fade out bob
        }
        let bob = sin(walkCycle) * bobAmplitude * (isMoving ? 1.0 : 0.0)

        // Final camera Y = terrain + eye height + bob
        cameraPosition.y = smoothedTerrainHeight + eyeHeight + bob

        // Camera shake decay
        if shakeIntensity > 0.001 {
            shakeIntensity *= exp(-shakeDecay * deltaTime)
            shakeOffset = SIMD3<Float>(
                (Float.random(in: -1...1)) * shakeIntensity,
                (Float.random(in: -1...1)) * shakeIntensity * 0.5,
                (Float.random(in: -1...1)) * shakeIntensity
            )
        } else {
            shakeOffset = .zero
            shakeIntensity = 0
        }
    }

    /// Trigger camera shake (called on mutation)
    func triggerShake(intensity: Float = 0.15) {
        shakeIntensity = max(shakeIntensity, intensity)
    }

    /// View matrix from current camera state (includes shake)
    var viewMatrix: simd_float4x4 {
        let eye = cameraPosition + shakeOffset
        let direction = SIMD3<Float>(
            sin(cameraYaw) * cos(cameraPitch),
            sin(cameraPitch),
            -cos(cameraYaw) * cos(cameraPitch)
        )
        let target = eye + direction
        return GameState.lookAt(eye: eye, center: target, up: SIMD3<Float>(0, 1, 0))
    }

    /// Projection matrix — wider FOV for immersion
    func projectionMatrix(aspectRatio: Float) -> simd_float4x4 {
        return GameState.perspective(fovY: Float.pi / 2.8,
                                     aspectRatio: aspectRatio,
                                     nearZ: 0.1,
                                     farZ: 200.0)
    }

    // MARK: - Matrix Helpers

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(center - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)

        var result = matrix_identity_float4x4
        result[0][0] = s.x
        result[1][0] = s.y
        result[2][0] = s.z
        result[0][1] = u.x
        result[1][1] = u.y
        result[2][1] = u.z
        result[0][2] = -f.x
        result[1][2] = -f.y
        result[2][2] = -f.z
        result[3][0] = -dot(s, eye)
        result[3][1] = -dot(u, eye)
        result[3][2] = dot(f, eye)
        return result
    }

    static func perspective(fovY: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let yScale = 1.0 / tan(fovY * 0.5)
        let xScale = yScale / aspectRatio
        let zRange = farZ - nearZ

        var result = simd_float4x4(0)
        result[0][0] = xScale
        result[1][1] = yScale
        result[2][2] = -(farZ + nearZ) / zRange
        result[2][3] = -1.0
        result[3][2] = -2.0 * farZ * nearZ / zRange
        return result
    }
}
