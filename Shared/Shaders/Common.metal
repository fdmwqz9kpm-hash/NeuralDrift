#include <metal_stdlib>
using namespace metal;

#import "../Renderer/ShaderTypes.h"

// --- Neural Network Helper Functions ---
// These run inside vertex and fragment shaders for real-time inference.
// Kept intentionally small for per-vertex/per-fragment execution at 60fps.

// ReLU activation
inline float relu(float x) {
    return max(0.0f, x);
}

// Tanh activation (used for color network — bounded output)
inline float tanh_act(float x) {
    return tanh(x);
}

// Forward pass: single dense layer with ReLU
// weights layout: [outputSize * inputSize weights] followed by [outputSize biases]
// Returns offset past the consumed weights+biases
template<int inputSize, int outputSize>
inline int denseLayerReLU(thread const float* input,
                          device const float* weights,
                          int offset,
                          thread float* output) {
    for (int o = 0; o < outputSize; o++) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weights[offset + o * inputSize + i];
        }
        // Bias
        sum += weights[offset + outputSize * inputSize + o];
        output[o] = relu(sum);
    }
    return offset + outputSize * inputSize + outputSize;
}

// Forward pass: single dense layer with Tanh
template<int inputSize, int outputSize>
inline int denseLayerTanh(thread const float* input,
                          device const float* weights,
                          int offset,
                          thread float* output) {
    for (int o = 0; o < outputSize; o++) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weights[offset + o * inputSize + i];
        }
        sum += weights[offset + outputSize * inputSize + o];
        output[o] = tanh_act(sum);
    }
    return offset + outputSize * inputSize + outputSize;
}

// Forward pass: single dense layer with NO activation (linear output)
template<int inputSize, int outputSize>
inline int denseLayerLinear(thread const float* input,
                            device const float* weights,
                            int offset,
                            thread float* output) {
    for (int o = 0; o < outputSize; o++) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weights[offset + o * inputSize + i];
        }
        sum += weights[offset + outputSize * inputSize + o];
        output[o] = sum;
    }
    return offset + outputSize * inputSize + outputSize;
}

// --- Terrain Neural Network ---
// Input:  (x, z, time, playerInfluence) -> 4 floats
// Hidden: 2 x 32 neurons (ReLU)
// Output: (height, normalPerturbX, normalPerturbY, normalPerturbZ) -> 4 floats
struct TerrainOutput {
    float height;
    float3 normalPerturbation;
};

inline TerrainOutput evaluateTerrainNetwork(float2 worldXZ,
                                            float time,
                                            float playerInfluence,
                                            device const float* weights) {
    // Prepare input
    float input[TERRAIN_INPUT_SIZE];
    input[0] = worldXZ.x * 0.1f; // Scale inputs to reasonable range
    input[1] = worldXZ.y * 0.1f;
    input[2] = sin(time * 0.5f);  // Periodic time input
    input[3] = playerInfluence;

    // Hidden layer 1: 4 -> 32 (ReLU)
    float hidden1[TERRAIN_HIDDEN1_SIZE];
    int offset = denseLayerReLU<TERRAIN_INPUT_SIZE, TERRAIN_HIDDEN1_SIZE>(
        input, weights, 0, hidden1);

    // Hidden layer 2: 32 -> 32 (ReLU)
    float hidden2[TERRAIN_HIDDEN2_SIZE];
    offset = denseLayerReLU<TERRAIN_HIDDEN1_SIZE, TERRAIN_HIDDEN2_SIZE>(
        hidden1, weights, offset, hidden2);

    // Output layer: 32 -> 4 (Linear)
    float output[TERRAIN_OUTPUT_SIZE];
    denseLayerLinear<TERRAIN_HIDDEN2_SIZE, TERRAIN_OUTPUT_SIZE>(
        hidden2, weights, offset, output);

    TerrainOutput result;
    result.height = output[0];
    result.normalPerturbation = float3(output[1], output[2], output[3]);
    return result;
}

// --- Color Neural Network ---
// Input:  (x, y, z, nx, ny, nz, vx, vy, vz, time) -> 10 floats
// Hidden: 2 x 16 neurons (Tanh)
// Output: (r, g, b) -> 3 floats (sigmoid applied for [0,1] range)
inline float3 evaluateColorNetwork(float3 worldPos,
                                   float3 normal,
                                   float3 viewDir,
                                   float time,
                                   device const float* weights) {
    float input[COLOR_INPUT_SIZE];
    input[0] = worldPos.x * 0.1f;
    input[1] = worldPos.y * 0.1f;
    input[2] = worldPos.z * 0.1f;
    input[3] = normal.x;
    input[4] = normal.y;
    input[5] = normal.z;
    input[6] = viewDir.x;
    input[7] = viewDir.y;
    input[8] = viewDir.z;
    input[9] = sin(time * 0.3f);

    // Hidden layer 1: 10 -> 16 (Tanh)
    float hidden1[COLOR_HIDDEN1_SIZE];
    int offset = denseLayerTanh<COLOR_INPUT_SIZE, COLOR_HIDDEN1_SIZE>(
        input, weights, 0, hidden1);

    // Hidden layer 2: 16 -> 16 (Tanh)
    float hidden2[COLOR_HIDDEN2_SIZE];
    offset = denseLayerTanh<COLOR_HIDDEN1_SIZE, COLOR_HIDDEN2_SIZE>(
        hidden1, weights, offset, hidden2);

    // Output layer: 16 -> 3 (Linear, then sigmoid for [0,1])
    float output[COLOR_OUTPUT_SIZE];
    denseLayerLinear<COLOR_HIDDEN2_SIZE, COLOR_OUTPUT_SIZE>(
        hidden2, weights, offset, output);

    // Sigmoid to clamp to [0, 1] color range
    return float3(1.0f / (1.0f + exp(-output[0])),
                  1.0f / (1.0f + exp(-output[1])),
                  1.0f / (1.0f + exp(-output[2])));
}
