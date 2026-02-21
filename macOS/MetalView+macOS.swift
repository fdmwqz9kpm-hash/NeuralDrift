import SwiftUI
import MetalKit

#if os(macOS)

/// NSViewRepresentable wrapper for MTKView on macOS.
/// Hosts the Metal renderer and forwards keyboard/mouse events.
struct MetalViewRepresentable: NSViewRepresentable {
    let renderer: Renderer

    func makeNSView(context: Context) -> MTKView {
        let mtkView = GameMTKView()
        mtkView.device = renderer.device
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        mtkView.delegate = renderer
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false

        // Enable keyboard/mouse input
        mtkView.becomeFirstResponder()
        mtkView.gameState = renderer.gameState
        mtkView.onResetWorld = { [weak renderer] in renderer?.resetWorld() }

        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}

/// Custom MTKView subclass that handles keyboard and mouse input on macOS.
class GameMTKView: MTKView {
    var gameState: GameState?
    var onResetWorld: (() -> Void)?

    private var keysPressed: Set<UInt16> = []
    private let lookSensitivity: Float = 0.003

    override var acceptsFirstResponder: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    // MARK: - Keyboard

    override func keyDown(with event: NSEvent) {
        keysPressed.insert(event.keyCode)
        updateMovement()

        // R key = reset world
        if event.keyCode == 15 {
            onResetWorld?()
        }
    }

    override func keyUp(with event: NSEvent) {
        keysPressed.remove(event.keyCode)
        updateMovement()
    }

    private func updateMovement() {
        guard let gs = gameState else { return }

        var forward: Float = 0
        var right: Float = 0

        if keysPressed.contains(13) || keysPressed.contains(126) { forward += 1 }  // W or Up
        if keysPressed.contains(1) || keysPressed.contains(125) { forward -= 1 }   // S or Down
        if keysPressed.contains(0) || keysPressed.contains(123) { right -= 1 }     // A or Left
        if keysPressed.contains(2) || keysPressed.contains(124) { right += 1 }     // D or Right

        gs.moveForward = forward
        gs.moveRight = right
    }

    // MARK: - Mouse

    override func mouseMoved(with event: NSEvent) {
        guard let gs = gameState else { return }
        let dx = Float(event.deltaX) * lookSensitivity
        let dy = Float(event.deltaY) * lookSensitivity
        gs.cameraYaw += dx
        gs.cameraPitch = max(gs.pitchMin, min(gs.pitchMax, gs.cameraPitch - dy))
    }

    override func mouseDragged(with event: NSEvent) {
        mouseMoved(with: event)
    }

    override func rightMouseDragged(with event: NSEvent) {
        mouseMoved(with: event)
    }

    override func mouseDown(with event: NSEvent) {
        gameState?.isInteracting = true
    }

    override func mouseUp(with event: NSEvent) {
        gameState?.isInteracting = false
    }

    // Capture mouse for FPS-style look
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas {
            removeTrackingArea(area)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseMoved, .activeInKeyWindow, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(area)
    }
}

#endif
