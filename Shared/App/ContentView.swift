import SwiftUI
import MetalKit
import GameKit

struct ContentView: View {
    @StateObject private var gameState = GameState()
    @StateObject private var gameCenterManager = GameCenterManager()
    @State private var renderer: Renderer?

    var body: some View {
        ZStack {
            // Metal rendering view
            MetalViewContainer(renderer: $renderer, gameState: gameState)
                .ignoresSafeArea()

            // HUD overlay
            VStack {
                HStack(alignment: .top) {
                    // Left: title + status
                    VStack(alignment: .leading, spacing: 4) {
                        Text("NEURAL DRIFT")
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))

                        if gameState.isInteracting {
                            Text("⟡ MUTATING")
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundColor(.cyan.opacity(0.9))
                        }

                        Text("Mutations: \(gameCenterManager.totalMutations)")
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.5))
                    }
                    .padding(12)
                    .background(.ultraThinMaterial)
                    .cornerRadius(10)

                    Spacer()

                    // Right: controls + Game Center
                    VStack(alignment: .trailing, spacing: 6) {
                        #if os(macOS)
                        Text("WASD: Move | Mouse: Look | Click: Mutate | R: Reset")
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.4))
                        #else
                        Text("Left: Move | Right: Look | Double-tap: Mutate")
                            .font(.system(size: 10, weight: .regular, design: .monospaced))
                            .foregroundColor(.white.opacity(0.4))
                        #endif

                        if gameCenterManager.isAuthenticated {
                            Button(action: { gameCenterManager.showLeaderboard() }) {
                                Label("Leaderboard", systemImage: "trophy.fill")
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.6))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                }
                .padding()

                Spacer()
            }
        }
        .onAppear {
            gameCenterManager.authenticate()
        }
        .onChange(of: gameState.isInteracting) { wasInteracting, isNowInteracting in
            if wasInteracting && !isNowInteracting {
                gameCenterManager.recordMutation()
                // Submit scores periodically
                if gameCenterManager.totalMutations % 10 == 0 {
                    gameCenterManager.submitScores()
                }
            }
        }
    }
}

/// Container that creates the MTKView and Renderer, bridging SwiftUI to Metal.
struct MetalViewContainer: View {
    @Binding var renderer: Renderer?
    let gameState: GameState

    var body: some View {
        GeometryReader { geometry in
            MetalViewBridge(renderer: $renderer)
                .onAppear {
                    // Renderer is created inside the platform-specific representable
                }
        }
    }
}

/// Bridge to the platform-specific MetalViewRepresentable.
struct MetalViewBridge {
    @Binding var renderer: Renderer?
}

#if os(macOS)
extension MetalViewBridge: NSViewRepresentable {
    func makeNSView(context: Context) -> MTKView {
        let mtkView = GameMTKView()
        if let r = Renderer(metalView: mtkView) {
            DispatchQueue.main.async {
                self.renderer = r
            }
            mtkView.gameState = r.gameState
            mtkView.onResetWorld = { [weak r] in r?.resetWorld() }
        }
        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}
#else
extension MetalViewBridge: UIViewRepresentable {
    func makeUIView(context: Context) -> GameMTKView {
        let mtkView = GameMTKView()
        if let r = Renderer(metalView: mtkView) {
            DispatchQueue.main.async {
                self.renderer = r
            }
            mtkView.gameState = r.gameState
            mtkView.onResetWorld = { [weak r] in r?.resetWorld() }
            mtkView.isMultipleTouchEnabled = true
        }
        return mtkView
    }

    func updateUIView(_ uiView: GameMTKView, context: Context) {}
}
#endif
