#include <metal_stdlib>
using namespace metal;

#import "../Renderer/ShaderTypes.h"

// Compute shader for modifying neural network weights based on player interaction.
// Only dispatched when the player actively interacts — not every frame.

kernel void updateTerrainWeights(
    device float*         weights      [[buffer(0)]],
    constant PlayerState& player       [[buffer(1)]],
    constant float&       deltaTime    [[buffer(2)]],
    constant float&       decayRate    [[buffer(3)]],
    device const float*   initialWeights [[buffer(4)]],
    uint                  tid          [[thread_position_in_grid]]
) {
    if (tid >= TERRAIN_WEIGHT_COUNT) return;

    float w = weights[tid];
    float w0 = initialWeights[tid];

    if (player.isInteracting) {
        // Perturbation based on player position and weight index
        // Use a hash-like function to create spatially-varying perturbations
        float seed = float(tid) * 0.01f + player.position.x * 0.37f + player.position.z * 0.73f;
        float perturbation = sin(seed * 17.3f) * cos(seed * 31.7f);
        perturbation *= player.interactionStrength * deltaTime * 0.5f;

        w += perturbation;

        // Clamp weights to prevent explosion
        w = clamp(w, -3.0f, 3.0f);
    }

    // Decay toward initial weights (prevents world from devolving into noise)
    w = mix(w, w0, decayRate * deltaTime);

    weights[tid] = w;
}

kernel void updateColorWeights(
    device float*         weights      [[buffer(0)]],
    constant PlayerState& player       [[buffer(1)]],
    constant float&       deltaTime    [[buffer(2)]],
    constant float&       decayRate    [[buffer(3)]],
    device const float*   initialWeights [[buffer(4)]],
    uint                  tid          [[thread_position_in_grid]]
) {
    if (tid >= COLOR_WEIGHT_COUNT) return;

    float w = weights[tid];
    float w0 = initialWeights[tid];

    if (player.isInteracting) {
        float seed = float(tid) * 0.013f + player.position.x * 0.41f + player.position.z * 0.67f;
        float perturbation = sin(seed * 23.1f) * cos(seed * 41.3f);
        perturbation *= player.interactionStrength * deltaTime * 0.3f;

        w += perturbation;
        w = clamp(w, -3.0f, 3.0f);
    }

    w = mix(w, w0, decayRate * deltaTime);

    weights[tid] = w;
}
