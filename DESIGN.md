# Neural Drift — Design Document

## Elevator Pitch
A macOS 26+ game where the world is rendered by a neural network running inside Metal 4 shaders. Player interactions modify the network weights in real-time, causing the environment to evolve unpredictably. The game IS the neural network.

---

## Hard Constraints

### Platform
- **Target**: macOS 26+ (Tahoe) AND iPadOS 26+
- **Hardware**: MacBook Pro M5 + iPad Pro M5 (reference devices)
- **Language**: Swift 6 + Metal Shading Language (with tensor extensions)
- **Framework**: SwiftUI + Metal 4 + MetalFX + GameKit
- **Architecture**: Single codebase, two targets (macOS + iPadOS)
- **Input**: Abstracted — keyboard/mouse on Mac, touch gestures on iPad

### What's IN Scope
- [x] Metal 4 renderer with ML inference in shaders (core differentiator)
- [x] Small neural network whose output IS the terrain/world geometry + color
- [x] Player avatar with basic movement (WASD + mouse on Mac, touch on iPad)
- [x] Neural "impression" system: player actions modify network weights
- [x] MetalFX spatial upscaling (render at lower res, upscale to display)
- [x] Game Center: leaderboard for "most unique world generated" (hash-based)
- [x] Games App integration (automatic via Metal 4)
- [x] Liquid Glass UI for HUD/menus
- [x] Ambient audio that reacts to neural state

### What's OUT of Scope (Do Not Add)
- [ ] Multiplayer / networking
- [ ] Natural language input (that's Concept 2)
- [ ] Complex physics simulation
- [ ] Traditional enemies/combat
- [ ] Story/narrative system
- [ ] Inventory or item systems
- [ ] Multiple levels or loading screens
- [ ] Cloud saves
- [ ] Any third-party dependencies

---

## Core Gameplay Loop

```
1. OBSERVE  → The neural network renders a surreal, evolving landscape
2. MOVE     → Player walks through the world (first-person)
3. INTERACT → Player touches/activates surfaces, leaving "neural impressions"
4. MUTATE   → Impressions modify network weights in the shader
5. EMERGE   → The world shifts, morphs, evolves in response
6. DISCOVER → Player finds "resonance points" — stable beautiful patterns
7. CAPTURE  → Player can snapshot resonance points (scored + leaderboard)
```

The game has no win state. It's an exploration of an alien intelligence.
The "score" is the aesthetic uniqueness of patterns you discover and capture.

---

## Technical Architecture

### Rendering Pipeline

```
┌─────────────────────────────────────────────────────┐
│                    Game Loop (60fps)                  │
├─────────────────────────────────────────────────────┤
│                                                       │
│  1. Input Processing (platform-abstracted)           │
│       ↓                                               │
│  2. Update Neural Weights (if interaction occurred)  │
│       ↓                                               │
│  3. Metal 4 Render Pass                              │
│     ┌──────────────────────────────────────┐         │
│     │  Vertex Shader                        │         │
│     │  - Neural net forward pass (tensors) │         │
│     │  - Output: vertex positions + normals │         │
│     ├──────────────────────────────────────┤         │
│     │  Fragment Shader                      │         │
│     │  - Neural net for color/material     │         │
│     │  - Output: pixel color + depth       │         │
│     └──────────────────────────────────────┘         │
│       ↓                                               │
│  4. MetalFX Spatial Upscale (1080p → native res)    │
│       ↓                                               │
│  5. UI Overlay (Liquid Glass HUD)                    │
│       ↓                                               │
│  6. Present to Display                               │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Neural Network Design

The network is intentionally SMALL — it runs per-vertex and per-fragment every frame.

**Terrain Network (Vertex Shader)**:
- Input: world position (x, z) + time + player influence field
- Hidden: 2 layers × 32 neurons (ReLU activation)
- Output: y-height + normal perturbation (4 floats)
- Weights: ~2,200 parameters stored in MTLBuffer

**Color Network (Fragment Shader)**:
- Input: world position (x, y, z) + normal + view direction + time
- Hidden: 2 layers × 16 neurons (tanh activation)
- Output: RGB color (3 floats)
- Weights: ~900 parameters stored in MTLBuffer

**Weight Modification System**:
- Player position creates a "influence sphere" (radius ~5 units)
- When player interacts, a compute shader adds small perturbations to weights
  whose receptive field overlaps the influence sphere
- Perturbations are seeded by: interaction type + position + current weight state
- Decay: weights slowly drift back toward initial state (prevents chaos)

### Memory Budget
- Neural weights: ~12 KB (tiny)
- Terrain mesh: 256×256 grid = ~1.5 MB vertex buffer
- Textures: minimal (network generates colors)
- MetalFX buffers: ~50 MB (for upscaling)
- Total GPU memory: < 100 MB

---

## Project Structure

```
NeuralDrift/
├── NeuralDrift.xcodeproj
├── Shared/                                    # Code shared across both platforms
│   ├── App/
│   │   ├── NeuralDriftApp.swift               # SwiftUI App entry point
│   │   └── ContentView.swift                  # Root view with MetalView
│   ├── Renderer/
│   │   ├── Renderer.swift                     # Metal 4 renderer coordinator
│   │   ├── RenderPipeline.swift               # Pipeline state management
│   │   ├── MetalFXUpscaler.swift              # MetalFX spatial upscaling
│   │   └── ShaderTypes.h                      # Shared CPU/GPU data types
│   ├── Shaders/
│   │   ├── NeuralTerrain.metal                # Vertex shader with neural net
│   │   ├── NeuralColor.metal                  # Fragment shader with neural net
│   │   ├── WeightUpdate.metal                 # Compute shader for weight mods
│   │   └── Common.metal                       # Shared neural net functions
│   ├── Game/
│   │   ├── GameState.swift                    # Core game state
│   │   ├── InputAbstraction.swift             # Platform-agnostic input protocol
│   │   ├── NeuralWeights.swift                # Weight buffer management
│   │   ├── InteractionSystem.swift            # Neural impression logic
│   │   └── ResonanceDetector.swift            # Finds stable/beautiful patterns
│   ├── UI/
│   │   ├── HUDView.swift                      # Liquid Glass HUD overlay
│   │   ├── MainMenuView.swift                 # Title screen
│   │   └── CaptureView.swift                  # Resonance capture UI
│   ├── Audio/
│   │   ├── AudioEngine.swift                  # Ambient audio system
│   │   └── NeuralSonification.swift           # Maps neural state to sound
│   ├── Social/
│   │   ├── GameCenterManager.swift            # Leaderboards + achievements
│   │   └── WorldHasher.swift                  # Unique world fingerprinting
│   └── Resources/
│       ├── Assets.xcassets
│       └── InitialWeights.json                # Starting neural net weights
├── macOS/                                      # macOS-specific code
│   ├── MetalView+macOS.swift                  # NSViewRepresentable wrapper
│   ├── KeyboardMouseInput.swift               # WASD + mouse input handler
│   └── Info.plist
├── iPadOS/                                     # iPadOS-specific code
│   ├── MetalView+iPadOS.swift                 # UIViewRepresentable wrapper
│   ├── TouchInput.swift                       # Touch gesture input handler
│   └── Info.plist
├── DESIGN.md
└── README.md
```

---

## Development Phases

### Phase 1: Foundation (Sessions 1-2)
- Xcode project scaffold with macOS + iPadOS targets
- SwiftUI app shell with platform-specific MetalView wrappers
- Basic Metal 4 renderer showing a flat grid
- Input abstraction layer (keyboard/mouse on Mac, touch on iPad)
- Camera + movement on both platforms
- Verify Metal 4 API availability on macOS 26 / iPadOS 26

### Phase 2: Neural Terrain (Sessions 3-4)
- Implement small neural network in Metal Shading Language using tensors
- Feed grid positions through network → get height values
- Render the neural terrain with basic lighting
- Confirm it runs at 60fps on M5

### Phase 3: Interaction (Sessions 5-6)
- Implement weight modification compute shader
- Player "touch" interactions that perturb weights
- Visual feedback when weights change (ripple effect)
- Weight decay system

### Phase 4: Polish & Integration (Sessions 7-8)
- MetalFX spatial upscaling
- Liquid Glass HUD (neural state visualization)
- Resonance detection + capture system
- Game Center leaderboard
- Ambient audio tied to neural state

### Phase 5: Ship (Session 9)
- Performance profiling on M5
- Final tuning of neural net size / interaction feel
- README + build instructions

---

## Research Validation (2026-02-20)

### Novelty Check
- **NeRF**: Passive view synthesis, no gameplay or weight modification — different.
- **NVIDIA RTX Neural Shaders**: Optimization tool (texture compression, denoising) — NN is a helper, not the world itself.
- **ShaderNN (academic, 2024)**: NN inference in OpenGL shaders for image processing — not game worlds.
- **Procedural generation**: Deterministic noise functions — not neural, not emergent.
- **Verdict**: No shipped game uses a neural network as the world itself with real-time weight modification from player interaction. Genuinely novel.

### Power & Battery Assessment
- M5 base GPU: ~5-8W estimated for our workload (vs 25-30W for AAA games)
- Our NN forward pass: ~13M FLOPs/frame (M5 sustains ~4 TFLOPS = 0.0003% utilization)
- Single draw call, no textures, no physics, MetalFX upscaling from 1080p
- Estimated battery: 9-14 hours MacBook, 4-7 hours iPad Pro at 60fps
- Weight updates only on interaction (not every frame)
- **Verdict**: Battery is a non-issue. Lighter than most casual games.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Metal 4 tensor API differs from docs | Medium | Fall back to manual matrix math in MSL |
| Neural net too slow per-frame | Low (M5) | Reduce hidden layer size, or run every Nth frame |
| World devolves into noise | High | Weight decay + clamping + careful perturbation scale |
| MetalFX API changes | Low | It's an enhancement, not core — can skip |
| Scope creep | High | This document. Refer back constantly. |

---

## Key Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| No third-party deps | Showcase pure Apple APIs | 2026-02-20 |
| Small neural net (2 layers) | Must run per-vertex at 60fps | 2026-02-20 |
| First-person perspective | Most immersive for exploration | 2026-02-20 |
| No combat/enemies | Not the point — this is about discovery | 2026-02-20 |
| Hash-based leaderboard | Unique scoring without subjective judging | 2026-02-20 |
| Dual platform (macOS + iPadOS) | Same M5 GPU, natural fit, wider audience | 2026-02-20 |
| SwiftUI over AppKit | Required for cross-platform, modern approach | 2026-02-20 |
| Input abstraction protocol | Clean separation of platform-specific input | 2026-02-20 |
