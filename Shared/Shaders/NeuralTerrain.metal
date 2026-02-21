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
    float  distToCamera;
};

// --- Vertex Shader ---
// Evaluates terrain neural network per-vertex: height + finite-difference normals.
vertex VertexOut neuralTerrainVertex(
    uint                     vertexID       [[vertex_id]],
    device const GridVertex* vertices       [[buffer(BufferIndexVertices)]],
    constant Uniforms&       uniforms       [[buffer(BufferIndexUniforms)]],
    device const float*      terrainWeights [[buffer(BufferIndexTerrainWeights)]],
    constant PlayerState&    player         [[buffer(BufferIndexPlayerState)]]
) {
    GridVertex vert = vertices[vertexID];
    float2 worldXZ = vert.position.xz;

    // Player influence falloff
    float distToPlayer = length(worldXZ - player.position.xz);
    float influence = max(0.0f, 1.0f - distToPlayer / player.influenceRadius);
    influence *= player.interactionStrength;

    // Evaluate terrain with finite-difference normals
    TerrainOutput terrain = evaluateTerrainFull(
        worldXZ, uniforms.time, influence, uniforms.gridSpacing, terrainWeights);

    float3 worldPos = float3(vert.position.x, terrain.height, vert.position.z);

    VertexOut out;
    out.position = uniforms.modelViewProjection * float4(worldPos, 1.0);
    out.worldPosition = worldPos;
    out.normal = terrain.normal;
    out.viewDirection = normalize(uniforms.cameraPosition - worldPos);
    out.time = uniforms.time;
    out.distToCamera = length(uniforms.cameraPosition - worldPos);

    return out;
}

// --- Fragment Shader ---
// Neural color + Blinn-Phong lighting + interaction glow + distance fog.
fragment float4 neuralColorFragment(
    VertexOut             in           [[stage_in]],
    device const float*   colorWeights [[buffer(BufferIndexColorWeights)]],
    constant PlayerState& player       [[buffer(BufferIndexPlayerState)]]
) {
    float3 N = normalize(in.normal);
    float3 V = normalize(in.viewDirection);

    // Neural network base color
    float3 baseColor = evaluateColorNetwork(
        in.worldPosition, N, V, in.time, colorWeights);

    // --- Blinn-Phong Lighting ---
    float3 lightDir = normalize(float3(0.4, 0.8, 0.3));
    float3 lightColor = float3(1.0, 0.95, 0.9);
    float3 fillDir = normalize(float3(-0.3, 0.4, -0.5));
    float3 fillColor = float3(0.15, 0.2, 0.35);

    // Diffuse
    float NdotL = max(dot(N, lightDir), 0.0f);
    float fillDiffuse = max(dot(N, fillDir), 0.0f);

    // Specular (Blinn-Phong)
    float3 H = normalize(lightDir + V);
    float NdotH = max(dot(N, H), 0.0f);
    float specular = pow(NdotH, 64.0f) * 0.5f;

    // Ambient (sky-colored)
    float ambient = 0.08f;
    float skyAmbient = 0.06f * (0.5f + 0.5f * N.y);

    float3 litColor = baseColor * (ambient + skyAmbient
                                   + NdotL * 0.7f * lightColor
                                   + fillDiffuse * 0.25f * fillColor)
                    + specular * lightColor * 0.4f;

    // --- Interaction Glow ---
    float distToPlayer = length(in.worldPosition.xz - player.position.xz);
    float influenceNorm = distToPlayer / player.influenceRadius;

    // Soft glow ring at influence boundary
    float ringWidth = 0.15f;
    float ring = exp(-pow((influenceNorm - 1.0f) / ringWidth, 2.0f));

    // Inner glow that pulses during interaction
    float innerGlow = saturate(1.0f - influenceNorm);
    innerGlow = innerGlow * innerGlow;

    // Energy ripples during active interaction
    float ripple = 0.0f;
    if (player.isInteracting) {
        ripple = sin(distToPlayer * 8.0f - in.time * 6.0f) * 0.5f + 0.5f;
        ripple *= innerGlow * 0.6f;
    }

    // Glow color: cyan-white for interaction, dim cyan for passive
    float3 glowColor = float3(0.1, 0.7, 1.0);
    float glowIntensity = ring * 0.3f + innerGlow * 0.15f + ripple * 0.4f;
    if (player.isInteracting) {
        glowColor = float3(0.3, 0.9, 1.0);
        glowIntensity *= 2.0f;
    }

    litColor += glowColor * glowIntensity;

    // --- Distance Fog ---
    float fogStart = 15.0f;
    float fogEnd = 55.0f;
    float fogFactor = saturate((in.distToCamera - fogStart) / (fogEnd - fogStart));
    fogFactor = fogFactor * fogFactor;

    float3 fogColor = mix(float3(0.02, 0.02, 0.06),
                          float3(0.05, 0.03, 0.1),
                          saturate(V.y * 0.5f + 0.5f));

    float3 finalColor = mix(litColor, fogColor, fogFactor);

    // Tone mapping
    finalColor = finalColor / (finalColor + 0.8f);

    return float4(finalColor, 1.0);
}
