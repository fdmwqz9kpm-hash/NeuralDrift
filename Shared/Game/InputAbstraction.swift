import Foundation

/// Platform-agnostic input actions that modify GameState.
/// Each platform (macOS, iPadOS) implements its own input handler
/// that translates raw events into these actions.
enum InputAction {
    case moveForward(Float)    // -1 to 1
    case moveRight(Float)      // -1 to 1
    case look(deltaX: Float, deltaY: Float)
    case interact(Bool)        // true = started, false = ended
    case resetWorld
}

/// Protocol that platform-specific input handlers conform to.
protocol InputHandler {
    var gameState: GameState { get }
    func handleAction(_ action: InputAction)
}

extension InputHandler {
    func handleAction(_ action: InputAction) {
        switch action {
        case .moveForward(let value):
            gameState.moveForward = value
        case .moveRight(let value):
            gameState.moveRight = value
        case .look(let deltaX, let deltaY):
            gameState.cameraYaw += deltaX
            gameState.cameraPitch = max(gameState.pitchMin,
                                        min(gameState.pitchMax,
                                            gameState.cameraPitch + deltaY))
        case .interact(let active):
            gameState.isInteracting = active
        case .resetWorld:
            break // Handled at renderer level
        }
    }
}
