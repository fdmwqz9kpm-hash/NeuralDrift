#include <metal_stdlib>
using namespace metal;

#import "../Renderer/ShaderTypes.h"

// --- Neural Network Helper Functions ---
// These run inside vertex and fragment shaders for real-time inference.
// Kept intentionally small for per-vertex/per-fragment execution at 60fps.

// Activations
inline float relu(float x) { return max(0.0f, x); }
inline float tanh_act(float x) { return tanh(x); }

// --- Sinusoidal Positional Encoding (the NeRF trick) ---
// Expands a single coordinate into [raw, sin(freq*c), cos(freq*c), ...] features.
// This lets tiny networks represent high-frequency spatial detail.
inline int positionalEncode(float coord, thread float* out, int startIdx) {
    out[startIdx] = coord;
    int idx = startIdx + 1;
    float freq = 1.0f;
    for (int b = 0; b < POS_ENCODE_BANDS; b++) {
        out[idx]     = sin(freq * coord);
        out[idx + 1] = cos(freq * coord);
        idx += 2;
        freq *= 2.0f;
    }
    return idx;
}

// --- Dense Layer Forward Passes ---
// weights layout: [outputSize * inputSize weights] then [outputSize biases]

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
        sum += weights[offset + outputSize * inputSize + o];
        output[o] = relu(sum);
    }
    return offset + outputSize * inputSize + outputSize;
}

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
// Evaluates height at a single (x,z) point.
// Used both for vertex displacement AND finite-difference normal computation.
inline float evaluateTerrainHeight(float2 worldXZ,
                                   float time,
                                   float playerInfluence,
                                   device const float* weights) {
    float input[TERRAIN_INPUT_SIZE];

    // Positional encoding for x and z with time-varying phase shift
    // This makes the terrain slowly drift and morph over time
    float phase = time * 0.15f;
    int idx = 0;
    idx = positionalEncode(worldXZ.x * 0.15f + sin(phase) * 0.3f, input, idx);  // 7 features
    idx = positionalEncode(worldXZ.y * 0.15f + cos(phase * 0.7f) * 0.3f, input, idx);  // 7 features
    input[idx++] = sin(time * 0.4f) * cos(time * 0.17f);  // Complex time signal
    input[idx++] = playerInfluence;

    // Hidden layer 1: 16 -> 32 (ReLU)
    float hidden1[TERRAIN_HIDDEN1_SIZE];
    int offset = denseLayerReLU<TERRAIN_INPUT_SIZE, TERRAIN_HIDDEN1_SIZE>(
        input, weights, 0, hidden1);

    // Hidden layer 2: 32 -> 32 (ReLU)
    float hidden2[TERRAIN_HIDDEN2_SIZE];
    offset = denseLayerReLU<TERRAIN_HIDDEN1_SIZE, TERRAIN_HIDDEN2_SIZE>(
        hidden1, weights, offset, hidden2);

    // Output layer: 32 -> 4 (Linear), we only use the height
    float output[TERRAIN_OUTPUT_SIZE];
    denseLayerLinear<TERRAIN_HIDDEN2_SIZE, TERRAIN_OUTPUT_SIZE>(
        hidden2, weights, offset, output);

    return tanh(output[0]) * 4.0f; // tanh keeps range [-4, 4], prevents runaway heights
}

// Full terrain evaluation: height + finite-difference normals
struct TerrainOutput {
    float  height;
    float3 normal;
};

inline TerrainOutput evaluateTerrainFull(float2 worldXZ,
                                         float time,
                                         float playerInfluence,
                                         float gridSpacing,
                                         device const float* weights) {
    float eps = gridSpacing * 0.5f;

    float hC = evaluateTerrainHeight(worldXZ, time, playerInfluence, weights);
    float hR = evaluateTerrainHeight(worldXZ + float2(eps, 0), time, playerInfluence, weights);
    float hF = evaluateTerrainHeight(worldXZ + float2(0, eps), time, playerInfluence, weights);

    // Finite difference normal
    float3 tangentX = float3(eps, hR - hC, 0.0f);
    float3 tangentZ = float3(0.0f, hF - hC, eps);
    float3 normal = normalize(cross(tangentZ, tangentX));

    TerrainOutput result;
    result.height = hC;
    result.normal = normal;
    return result;
}

// --- Color Neural Network ---
// Positional encoding on world position + raw normal/viewDir/time
inline float3 evaluateColorNetwork(float3 worldPos,
                                   float3 normal,
                                   float3 viewDir,
                                   float time,
                                   device const float* weights) {
    float input[COLOR_INPUT_SIZE];
    int idx = 0;

    // Positional encoding for x, y, z with slow color drift
    float cPhase = time * 0.1f;
    idx = positionalEncode(worldPos.x * 0.1f + sin(cPhase * 0.6f) * 0.2f, input, idx);
    idx = positionalEncode(worldPos.y * 0.2f, input, idx);
    idx = positionalEncode(worldPos.z * 0.1f + cos(cPhase * 0.4f) * 0.2f, input, idx);

    // Raw vectors
    input[idx++] = normal.x;
    input[idx++] = normal.y;
    input[idx++] = normal.z;
    input[idx++] = viewDir.x;
    input[idx++] = viewDir.y;
    input[idx++] = viewDir.z;
    input[idx++] = sin(time * 0.25f) * cos(time * 0.11f);

    // Hidden layer 1: 28 -> 24 (Tanh)
    float hidden1[COLOR_HIDDEN1_SIZE];
    int offset = denseLayerTanh<COLOR_INPUT_SIZE, COLOR_HIDDEN1_SIZE>(
        input, weights, 0, hidden1);

    // Hidden layer 2: 24 -> 24 (Tanh)
    float hidden2[COLOR_HIDDEN2_SIZE];
    offset = denseLayerTanh<COLOR_HIDDEN1_SIZE, COLOR_HIDDEN2_SIZE>(
        hidden1, weights, offset, hidden2);

    // Output layer: 24 -> 3 (Linear, then sigmoid)
    float output[COLOR_OUTPUT_SIZE];
    denseLayerLinear<COLOR_HIDDEN2_SIZE, COLOR_OUTPUT_SIZE>(
        hidden2, weights, offset, output);

    return float3(1.0f / (1.0f + exp(-output[0])),
                  1.0f / (1.0f + exp(-output[1])),
                  1.0f / (1.0f + exp(-output[2])));
}
