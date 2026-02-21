import SwiftUI
import MetalKit
import GameKit

struct ContentView: View {
    @StateObject private var gameCenterManager = GameCenterManager()
    @State private var renderer: Renderer?
    @State private var showControls = true
    @State private var lastCaptureScore: Int? = nil

    private var resonanceDetector: ResonanceDetector? {
        renderer?.resonanceDetector
    }

    var body: some View {
        ZStack {
            // Metal rendering view
            MetalViewContainer(renderer: $renderer)
                .ignoresSafeArea()

            // HUD overlay
            VStack {
                HStack(alignment: .top, spacing: 12) {
                    // Left: title + status + resonance
                    HUDPanel {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("NEURAL DRIFT")
                                .font(.system(size: 14, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.85))

                            HStack(spacing: 8) {
                                if renderer?.gameState.isInteracting == true {
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

                            // Resonance stats
                            if let rd = resonanceDetector {
                                HStack(spacing: 10) {
                                    Label("\(rd.capturedCount)", systemImage: "sparkles")
                                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                                        .foregroundStyle(.yellow.opacity(0.8))

                                    Text("\(rd.totalScore) pts")
                                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                                        .foregroundStyle(.white.opacity(0.7))
                                        .contentTransition(.numericText())
                                }
                            }
                        }
                    }

                    Spacer()

                    // Center: nearby orb indicator
                    if resonanceDetector?.nearbyOrb != nil {
                        HUDPanel {
                            VStack(spacing: 4) {
                                Image(systemName: "circle.hexagongrid.fill")
                                    .font(.system(size: 20))
                                    .foregroundStyle(.yellow)
                                    .symbolEffect(.pulse)
                                Text("RESONANCE NEAR")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(.yellow.opacity(0.9))
                                Text("Click to capture")
                                    .font(.system(size: 8, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.5))
                            }
                        }
                        .transition(.scale.combined(with: .opacity))
                    }

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

                // Bottom: capture score popup
                if let score = lastCaptureScore {
                    Text("+\(score)")
                        .font(.system(size: 28, weight: .bold, design: .monospaced))
                        .foregroundStyle(.yellow)
                        .shadow(color: .yellow.opacity(0.5), radius: 10)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                        .padding(.bottom, 40)
                }
            }
            .animation(.easeInOut(duration: 0.3), value: resonanceDetector?.nearbyOrb != nil)
            .animation(.spring(duration: 0.5), value: lastCaptureScore)
        }
        .onAppear {
            gameCenterManager.authenticate()
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
                withAnimation(.easeOut(duration: 0.5)) {
                    showControls = false
                }
            }
        }
        .onTapGesture(count: 3) {
            withAnimation { showControls.toggle() }
        }
        .onChange(of: renderer?.gameState.isInteracting ?? false) { wasInteracting, isNowInteracting in
            if wasInteracting && !isNowInteracting {
                gameCenterManager.recordMutation()
                if gameCenterManager.totalMutations % 10 == 0 {
                    gameCenterManager.submitScores()
                }
            }
            // Try to capture nearby orb when interaction starts
            if !wasInteracting && isNowInteracting {
                if let rd = resonanceDetector {
                    let score = rd.captureNearestOrb()
                    if score > 0 {
                        withAnimation { lastCaptureScore = score }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                            withAnimation { lastCaptureScore = nil }
                        }
                    }
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

    var body: some View {
        GeometryReader { geometry in
            MetalViewBridge(renderer: $renderer)
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
