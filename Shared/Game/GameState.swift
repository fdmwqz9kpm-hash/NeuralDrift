import Foundation
import simd

/// Central game state — owns the camera, player position, and interaction state.
/// Platform-agnostic; input handlers update this via the InputAction protocol.
final class GameState: ObservableObject {
    // Camera
    @Published var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 5, 10)
    @Published var cameraYaw: Float = 0       // Radians, horizontal rotation
    @Published var cameraPitch: Float = -0.3   // Radians, vertical rotation (looking slightly down)

    // Movement
    var moveForward: Float = 0   // -1 to 1
    var moveRight: Float = 0     // -1 to 1
    let moveSpeed: Float = 8.0

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
    let pitchMin: Float = -.pi / 2.5
    let pitchMax: Float = .pi / 4.0
    let cameraHeight: Float = 3.0

    func update() {
        let now = CFAbsoluteTimeGetCurrent()
        deltaTime = Float(now - lastUpdateTime)
        deltaTime = min(deltaTime, 1.0 / 30.0) // Cap to prevent huge jumps
        lastUpdateTime = now
        totalTime += deltaTime

        // Calculate forward and right vectors from yaw
        let forward = SIMD3<Float>(
            sin(cameraYaw),
            0,
            -cos(cameraYaw)
        )
        let right = SIMD3<Float>(
            cos(cameraYaw),
            0,
            sin(cameraYaw)
        )

        // Apply movement
        let velocity = (forward * moveForward + right * moveRight) * moveSpeed * deltaTime
        cameraPosition += velocity
        cameraPosition.y = cameraHeight // Lock height for now (Phase 2: terrain-following)
    }

    /// View matrix from current camera state
    var viewMatrix: simd_float4x4 {
        let direction = SIMD3<Float>(
            sin(cameraYaw) * cos(cameraPitch),
            sin(cameraPitch),
            -cos(cameraYaw) * cos(cameraPitch)
        )
        let target = cameraPosition + direction
        return GameState.lookAt(eye: cameraPosition, center: target, up: SIMD3<Float>(0, 1, 0))
    }

    /// Projection matrix
    func projectionMatrix(aspectRatio: Float) -> simd_float4x4 {
        return GameState.perspective(fovY: Float.pi / 3.0,
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
