# Neural Drift

**A macOS 26+ / iPadOS 26+ game where the world is a neural network.**

The entire terrain and color of the world are generated in real-time by small neural networks running inside Metal shaders. Walk through the landscape and click to *mutate* the network weights — watch the world ripple, deform, and shift color as the neural network rewires itself around you.

## What Makes This Different

Traditional games use static meshes or procedural noise for terrain. Neural Drift uses actual neural network inference **per-vertex and per-fragment** at 60fps:

- **Terrain Network** (1,732 weights): Positional-encoded coordinates → 2-layer ReLU → height
- **Color Network** (1,371 weights): Positional-encoded world position + normals + view direction → 2-layer Tanh → RGB
- **Weight Mutation**: Player interactions perturb network weights via compute shaders with wave-like propagation patterns
- **Decay**: Weights slowly drift back toward their initial state, creating a living, breathing landscape

## Features

- **Neural Network Rendering** — NeRF-style sinusoidal positional encoding (3 frequency bands) lets tiny networks produce rich spatial detail
- **Real-Time Weight Modification** — Click/tap to mutate the neural network; terrain deforms and colors shift
- **MetalFX Spatial Upscaling** — Renders at 2/3 resolution and upscales for better performance
- **Blinn-Phong Lighting** — Key + fill lights, specular highlights, sky-colored ambient
- **Distance Fog** — Quadratic falloff with dark blue-purple atmosphere
- **Interaction Glow** — Cyan pulse ring + energy ripples at mutation point
- **Finite-Difference Normals** — Accurate surface normals computed from height samples
- **Liquid Glass UI** — macOS 26 / iPadOS 26 native glass effect HUD
- **Game Center** — Authentication, mutation leaderboard
- **Dual Platform** — macOS + iPadOS from shared codebase

## Requirements

- **macOS 26.0+** or **iPadOS 26.0+**
- **Xcode 26.0+**
- Apple Silicon (M-series) — designed for M5, runs on any Apple Silicon Mac/iPad
- Metal Toolchain (auto-downloaded on first build)

## Build

```bash
# Generate Xcode project (if project.yml was modified)
xcodegen generate --spec project.yml

# Build macOS target
xcodebuild -project NeuralDrift.xcodeproj \
  -scheme NeuralDrift-macOS \
  -destination 'platform=macOS' \
  build

# Build iPadOS target
xcodebuild -project NeuralDrift.xcodeproj \
  -scheme NeuralDrift-iPadOS \
  -destination 'generic/platform=iOS' \
  build
```

Or open `NeuralDrift.xcodeproj` in Xcode and run.

## Controls

| Action | macOS | iPadOS |
|--------|-------|--------|
| Move | WASD / Arrow keys | Left half touch drag |
| Look | Mouse | Right half touch drag |
| Mutate | Left click (hold) | Double-tap (hold) |
| Reset world | R | — |
| Toggle controls | Triple-click | Triple-tap |

## Architecture

```
Shared/
├── App/           SwiftUI entry point + ContentView with Liquid Glass HUD
├── Game/          GameState, NeuralWeights (Xavier/He init), InputAbstraction, GameCenter
├── Renderer/      Metal renderer, MetalFX upscaler, ShaderTypes.h bridge
└── Shaders/       Common.metal (NN inference), NeuralTerrain.metal, WeightUpdate.metal, Blit.metal
macOS/             NSView-based MTKView + keyboard/mouse input
iPadOS/            UIView-based MTKView + touch input
```

## Tech Stack

Pure Apple APIs — zero third-party dependencies:

- **Metal** — GPU rendering + compute shaders
- **MetalFX** — Spatial upscaling
- **GameKit** — Game Center integration
- **SwiftUI** — UI with Liquid Glass effects
- **Swift 6** — Strict concurrency

## License

MIT
