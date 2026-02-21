import SwiftUI
import MetalKit
import GameKit

struct ContentView: View {
    @StateObject private var gameState = GameState()
    @StateObject private var gameCenterManager = GameCenterManager()
    @State private var renderer: Renderer?
    @State private var showControls = true

    var body: some View {
        ZStack {
            // Metal rendering view
            MetalViewContainer(renderer: $renderer, gameState: gameState)
                .ignoresSafeArea()

            // HUD overlay
            VStack {
                HStack(alignment: .top, spacing: 12) {
                    // Left: title + status
                    HUDPanel {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("NEURAL DRIFT")
                                .font(.system(size: 14, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.85))

                            HStack(spacing: 8) {
                                if gameState.isInteracting {
                                    Text("⟡ MUTATING")
                                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                        .foregroundStyle(.cyan)
                                        .transition(.opacity.combined(with: .scale(scale: 0.9)))
                                }

                                Text("\(gameCenterManager.totalMutations) mutations")
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.5))
                                    .contentTransition(.numericText())
                            }
                            .animation(.easeInOut(duration: 0.2), value: gameState.isInteracting)
                        }
                    }

                    Spacer()

                    // Right: controls + leaderboard
                    if showControls {
                        HUDPanel {
                            VStack(alignment: .trailing, spacing: 8) {
                                #if os(macOS)
                                Text("WASD Move | Mouse Look | Click Mutate | R Reset")
                                    .font(.system(size: 9, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.45))
                                #else
                                Text("Left Move | Right Look | 2-Tap Mutate")
                                    .font(.system(size: 9, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.45))
                                #endif

                                if gameCenterManager.isAuthenticated {
                                    Button(action: { gameCenterManager.showLeaderboard() }) {
                                        Label("Leaderboard", systemImage: "trophy.fill")
                                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                                            .foregroundStyle(.white.opacity(0.65))
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                        .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)

                Spacer()
            }
        }
        .onAppear {
            gameCenterManager.authenticate()
            // Auto-hide controls after 8 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
                withAnimation(.easeOut(duration: 0.5)) {
                    showControls = false
                }
            }
        }
        .onTapGesture(count: 3) {
            withAnimation { showControls.toggle() }
        }
        .onChange(of: gameState.isInteracting) { wasInteracting, isNowInteracting in
            if wasInteracting && !isNowInteracting {
                gameCenterManager.recordMutation()
                if gameCenterManager.totalMutations % 10 == 0 {
                    gameCenterManager.submitScores()
                }
            }
        }
    }
}

/// Liquid Glass-style HUD panel for macOS 26+ / iPadOS 26+.
struct HUDPanel<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 12))
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
