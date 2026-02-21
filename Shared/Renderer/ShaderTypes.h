#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Buffer indices for vertex shader
enum BufferIndex {
    BufferIndexVertices     = 0,
    BufferIndexUniforms     = 1,
    BufferIndexTerrainWeights = 2,
    BufferIndexColorWeights = 3,
    BufferIndexPlayerState  = 4
};

// Vertex attribute indices
enum VertexAttribute {
    VertexAttributePosition = 0,
    VertexAttributeTexcoord = 1
};

// Uniforms shared between CPU and GPU
struct Uniforms {
    simd_float4x4 modelViewProjection;
    simd_float4x4 modelView;
    simd_float3x3 normalMatrix;
    simd_float3   cameraPosition;
    float         time;
    float         gridSize;
    float         gridSpacing;
    float         _padding;
};

// Player state passed to GPU for influence calculations
struct PlayerState {
    simd_float3 position;
    float       influenceRadius;
    float       interactionStrength;
    int         isInteracting;
    float       _padding[2];
};

// Terrain neural network dimensions
// Input: (x, z, time, playerInfluence) = 4
// Hidden1: 32 neurons
// Hidden2: 32 neurons
// Output: (height, normalX, normalY, normalZ) = 4
#define TERRAIN_INPUT_SIZE   4
#define TERRAIN_HIDDEN1_SIZE 32
#define TERRAIN_HIDDEN2_SIZE 32
#define TERRAIN_OUTPUT_SIZE  4

// Color neural network dimensions
// Input: (x, y, z, nx, ny, nz, vx, vy, vz, time) = 10
// Hidden1: 16 neurons
// Hidden2: 16 neurons
// Output: (r, g, b) = 3
#define COLOR_INPUT_SIZE   10
#define COLOR_HIDDEN1_SIZE 16
#define COLOR_HIDDEN2_SIZE 16
#define COLOR_OUTPUT_SIZE  3

// Total weight counts (weights + biases per layer)
// Terrain: (4*32 + 32) + (32*32 + 32) + (32*4 + 4) = 160 + 1056 + 132 = 1348
#define TERRAIN_WEIGHT_COUNT 1348

// Color: (10*16 + 16) + (16*16 + 16) + (16*3 + 3) = 176 + 272 + 51 = 499
#define COLOR_WEIGHT_COUNT 499

// Grid vertex structure
struct GridVertex {
    simd_float3 position;
    simd_float2 texcoord;
};

#endif /* ShaderTypes_h */
