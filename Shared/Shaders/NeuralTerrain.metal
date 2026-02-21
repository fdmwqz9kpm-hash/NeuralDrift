#include <metal_stdlib>
using namespace metal;

#import "../Renderer/ShaderTypes.h"
#import "Common.metal"

// Vertex output / Fragment input
struct VertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float3 viewDirection;
    float  time;
};

// --- Vertex Shader ---
// Evaluates the terrain neural network per-vertex to generate height + normals.
vertex VertexOut neuralTerrainVertex(
    uint                     vertexID      [[vertex_id]],
    device const GridVertex* vertices      [[buffer(BufferIndexVertices)]],
    constant Uniforms&       uniforms      [[buffer(BufferIndexUniforms)]],
    device const float*      terrainWeights [[buffer(BufferIndexTerrainWeights)]],
    constant PlayerState&    player        [[buffer(BufferIndexPlayerState)]]
) {
    GridVertex vert = vertices[vertexID];
    float2 worldXZ = vert.position.xz;

    // Calculate player influence at this vertex
    float distToPlayer = length(worldXZ - player.position.xz);
    float influence = max(0.0f, 1.0f - distToPlayer / player.influenceRadius);
    influence *= player.interactionStrength;

    // Evaluate terrain neural network
    TerrainOutput terrain = evaluateTerrainNetwork(
        worldXZ, uniforms.time, influence, terrainWeights);

    // Apply neural network output to vertex position
    float3 worldPos = float3(vert.position.x,
                             terrain.height,
                             vert.position.z);

    // Compute normal: base up vector + neural perturbation
    float3 normal = normalize(float3(0.0, 1.0, 0.0) + terrain.normalPerturbation * 0.3f);

    VertexOut out;
    out.position = uniforms.modelViewProjection * float4(worldPos, 1.0);
    out.worldPosition = worldPos;
    out.normal = uniforms.normalMatrix * normal;
    out.viewDirection = normalize(uniforms.cameraPosition - worldPos);
    out.time = uniforms.time;

    return out;
}

// --- Fragment Shader ---
// Evaluates the color neural network per-fragment.
fragment float4 neuralColorFragment(
    VertexOut             in            [[stage_in]],
    device const float*   colorWeights  [[buffer(BufferIndexColorWeights)]]
) {
    float3 normal = normalize(in.normal);
    float3 viewDir = normalize(in.viewDirection);

    // Evaluate color neural network
    float3 color = evaluateColorNetwork(
        in.worldPosition, normal, viewDir, in.time, colorWeights);

    // Basic directional lighting
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float diffuse = max(dot(normal, lightDir), 0.0f);
    float ambient = 0.15f;

    float3 finalColor = color * (ambient + diffuse * 0.85f);

    return float4(finalColor, 1.0);
}
