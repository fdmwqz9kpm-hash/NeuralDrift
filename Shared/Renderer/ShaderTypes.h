#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// Buffer indices for vertex shader
enum BufferIndex {
    BufferIndexVertices       = 0,
    BufferIndexUniforms       = 1,
    BufferIndexTerrainWeights = 2,
    BufferIndexColorWeights   = 3,
    BufferIndexPlayerState    = 4,
    BufferIndexResonance      = 5
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

// Positional encoding: 3 frequency bands per spatial coordinate
// Each coord c -> [c, sin(c), cos(c), sin(2c), cos(2c), sin(4c), cos(4c)] = 7 features
#define POS_ENCODE_BANDS 3
#define POS_ENCODE_PER_COORD 7  // raw + 2*bands

// Terrain neural network dimensions
// Input: posEncode(x,z)=14 + time(1) + playerInfluence(1) = 16
// Hidden1: 32 neurons, Hidden2: 32 neurons
// Output: (height, normalX, normalY, normalZ) = 4
#define TERRAIN_INPUT_SIZE   16
#define TERRAIN_HIDDEN1_SIZE 32
#define TERRAIN_HIDDEN2_SIZE 32
#define TERRAIN_OUTPUT_SIZE  4

// Color neural network dimensions
// Input: posEncode(x,y,z)=21 + normal(3) + viewDir(3) + time(1) = 28
// Hidden1: 24 neurons, Hidden2: 24 neurons
// Output: (r, g, b) = 3
#define COLOR_INPUT_SIZE   28
#define COLOR_HIDDEN1_SIZE 24
#define COLOR_HIDDEN2_SIZE 24
#define COLOR_OUTPUT_SIZE  3

// Total weight counts (weights + biases per layer)
// Terrain: (16*32 + 32) + (32*32 + 32) + (32*4 + 4) = 544 + 1056 + 132 = 1732
#define TERRAIN_WEIGHT_COUNT 1732

// Color: (28*24 + 24) + (24*24 + 24) + (24*3 + 3) = 696 + 600 + 75 = 1371
#define COLOR_WEIGHT_COUNT 1371

// Resonance orb data passed to GPU
#define MAX_RESONANCE_ORBS 5

struct ResonanceOrbGPU {
    simd_float3 position;
    float       intensity;
    simd_float3 color;
    float       spawnTime;
};

struct ResonanceData {
    struct ResonanceOrbGPU orbs[MAX_RESONANCE_ORBS];
    int   orbCount;
    float currentTime;
    float _padding[2];
};

// Grid vertex structure
struct GridVertex {
    simd_float3 position;
    simd_float2 texcoord;
};

#endif /* ShaderTypes_h */
