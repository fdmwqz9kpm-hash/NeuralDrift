import SwiftUI
import MetalKit

#if os(iOS)

/// UIViewRepresentable wrapper for MTKView on iPadOS.
/// Hosts the Metal renderer and forwards touch gesture events.
struct MetalViewRepresentable: UIViewRepresentable {
    let renderer: Renderer

    func makeUIView(context: Context) -> GameMTKView {
        let mtkView = GameMTKView()
        mtkView.device = renderer.device
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        mtkView.delegate = renderer
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false

        mtkView.gameState = renderer.gameState
        mtkView.onResetWorld = { [weak renderer] in renderer?.resetWorld() }
        mtkView.isMultipleTouchEnabled = true

        return mtkView
    }

    func updateUIView(_ uiView: GameMTKView, context: Context) {}
}

/// Custom MTKView subclass that handles touch input on iPadOS.
class GameMTKView: MTKView {
    var gameState: GameState?
    var onResetWorld: (() -> Void)?

    private let lookSensitivity: Float = 0.005
    private let moveSensitivity: Float = 0.02

    // Track touches by phase
    private var moveTouchID: UITouch?
    private var lookTouchID: UITouch?
    private var moveTouchStart: CGPoint = .zero
    private var lookTouchPrevious: CGPoint = .zero

    // MARK: - Touch Handling

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            let location = touch.location(in: self)

            // Left half of screen = move, right half = look
            if location.x < bounds.midX {
                if moveTouchID == nil {
                    moveTouchID = touch
                    moveTouchStart = location
                }
            } else {
                if lookTouchID == nil {
                    lookTouchID = touch
                    lookTouchPrevious = location
                }
            }
        }

        // Double-tap to interact
        for touch in touches where touch.tapCount == 2 {
            gameState?.isInteracting = true
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let gs = gameState else { return }

        for touch in touches {
            let location = touch.location(in: self)

            if touch === moveTouchID {
                // Virtual joystick: offset from start point
                let dx = Float(location.x - moveTouchStart.x) * moveSensitivity
                let dy = Float(location.y - moveTouchStart.y) * moveSensitivity
                gs.moveRight = max(-1, min(1, dx))
                gs.moveForward = max(-1, min(1, -dy)) // Invert Y
            }

            if touch === lookTouchID {
                // Look: delta from previous position
                let dx = Float(location.x - lookTouchPrevious.x) * lookSensitivity
                let dy = Float(location.y - lookTouchPrevious.y) * lookSensitivity
                gs.cameraYaw += dx
                gs.cameraPitch = max(gs.pitchMin, min(gs.pitchMax, gs.cameraPitch - dy))
                lookTouchPrevious = location
            }
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            if touch === moveTouchID {
                moveTouchID = nil
                gameState?.moveForward = 0
                gameState?.moveRight = 0
            }
            if touch === lookTouchID {
                lookTouchID = nil
            }
        }

        gameState?.isInteracting = false
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        touchesEnded(touches, with: event)
    }
}

#endif
