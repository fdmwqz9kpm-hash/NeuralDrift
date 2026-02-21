#include <metal_stdlib>
using namespace metal;

#import "../Renderer/ShaderTypes.h"

// --- Pseudo-random hash for coherent noise ---
inline float hash(float n) {
    return fract(sin(n) * 43758.5453123f);
}

inline float smoothNoise(float x) {
    float i = floor(x);
    float f = fract(x);
    f = f * f * (3.0f - 2.0f * f); // Smoothstep
    return mix(hash(i), hash(i + 1.0f), f);
}

// --- Terrain Weight Update ---
// Creates wave-like propagation patterns that ripple through the network,
// producing coherent terrain deformations rather than random noise.
kernel void updateTerrainWeights(
    device float*         weights        [[buffer(0)]],
    constant PlayerState& player         [[buffer(1)]],
    constant float&       deltaTime      [[buffer(2)]],
    constant float&       decayRate      [[buffer(3)]],
    device const float*   initialWeights [[buffer(4)]],
    uint                  tid            [[thread_position_in_grid]]
) {
    if (tid >= TERRAIN_WEIGHT_COUNT) return;

    float w = weights[tid];
    float w0 = initialWeights[tid];

    if (player.isInteracting) {
        float t = float(tid);

        // Layer-aware: identify which layer this weight belongs to
        // Layer 1: 0..<544, Layer 2: 544..<1600, Layer 3: 1600..<1732
        float layerPhase = 0.0f;
        float layerStrength = 1.0f;
        if (tid < 544) {
            layerPhase = 0.0f;         // Input layer: spatial structure
            layerStrength = 1.2f;
        } else if (tid < 1600) {
            layerPhase = 2.094f;       // Hidden layer: feature mixing (2π/3)
            layerStrength = 0.8f;
        } else {
            layerPhase = 4.189f;       // Output layer: height/normal control (4π/3)
            layerStrength = 1.5f;
        }

        // Wave propagation: creates expanding rings of mutation
        float playerAngle = atan2(player.position.z, player.position.x + 0.001f);
        float wave = sin(t * 0.05f + playerAngle * 3.0f + layerPhase);

        // Coherent directional perturbation
        float direction = smoothNoise(t * 0.02f + player.position.x * 0.5f);
        float magnitude = smoothNoise(t * 0.03f + player.position.z * 0.5f);

        float perturbation = wave * direction * magnitude;
        perturbation *= player.interactionStrength * deltaTime * layerStrength * 0.6f;

        w += perturbation;
        w = clamp(w, -4.0f, 4.0f);
    }

    // Gentle decay toward initial weights
    w = mix(w, w0, decayRate * deltaTime);

    weights[tid] = w;
}

// --- Color Weight Update ---
// Uses different patterns than terrain for visual variety — more swirly/chromatic.
kernel void updateColorWeights(
    device float*         weights        [[buffer(0)]],
    constant PlayerState& player         [[buffer(1)]],
    constant float&       deltaTime      [[buffer(2)]],
    constant float&       decayRate      [[buffer(3)]],
    device const float*   initialWeights [[buffer(4)]],
    uint                  tid            [[thread_position_in_grid]]
) {
    if (tid >= COLOR_WEIGHT_COUNT) return;

    float w = weights[tid];
    float w0 = initialWeights[tid];

    if (player.isInteracting) {
        float t = float(tid);

        // Chromatic wave: creates color shifts that spiral through the palette
        float playerDist = length(player.position.xz);
        float chromaWave = sin(t * 0.07f + playerDist * 0.3f)
                         * cos(t * 0.03f - playerDist * 0.2f);

        // Output-layer boost: last 75 weights control RGB directly
        float outputBoost = (tid >= 1296) ? 2.0f : 1.0f;

        float perturbation = chromaWave * player.interactionStrength * deltaTime * 0.4f * outputBoost;

        w += perturbation;
        w = clamp(w, -4.0f, 4.0f);
    }

    w = mix(w, w0, decayRate * deltaTime);

    weights[tid] = w;
}
