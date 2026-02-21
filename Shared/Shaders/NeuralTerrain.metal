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
    VertexOut               in           [[stage_in]],
    device const float*     colorWeights [[buffer(BufferIndexColorWeights)]],
    constant PlayerState&   player       [[buffer(BufferIndexPlayerState)]],
    constant ResonanceData& resonance    [[buffer(BufferIndexResonance)]]
) {
    float3 N = normalize(in.normal);
    float3 V = normalize(in.viewDirection);

    // Neural network base color (sigmoid output, range 0-1)
    float3 baseColor = evaluateColorNetwork(
        in.worldPosition, N, V, in.time, colorWeights);

    // Add height-based tint for visual variety even with flat terrain
    float heightNorm = saturate(in.worldPosition.y * 0.15f + 0.5f);
    float3 heightTint = mix(float3(0.2, 0.35, 0.5),   // Low: blue-grey
                            float3(0.6, 0.8, 0.4),     // High: green
                            heightNorm);
    baseColor = mix(heightTint, baseColor, 0.6f); // Blend neural + height

    // --- Blinn-Phong Lighting ---
    float3 lightDir = normalize(float3(0.4, 0.8, 0.3));
    float3 lightColor = float3(1.0, 0.97, 0.92);
    float3 fillDir = normalize(float3(-0.3, 0.4, -0.5));
    float3 fillColor = float3(0.3, 0.4, 0.6);

    // Diffuse
    float NdotL = max(dot(N, lightDir), 0.0f);
    float fillDiffuse = max(dot(N, fillDir), 0.0f);

    // Specular (Blinn-Phong)
    float3 H = normalize(lightDir + V);
    float NdotH = max(dot(N, H), 0.0f);
    float specular = pow(NdotH, 32.0f) * 0.3f;

    // Strong ambient so terrain is always visible
    float ambient = 0.25f;
    float skyAmbient = 0.15f * (0.5f + 0.5f * N.y);

    float3 litColor = baseColor * (ambient + skyAmbient
                                   + NdotL * 0.8f * lightColor
                                   + fillDiffuse * 0.3f * fillColor)
                    + specular * lightColor * 0.3f;

    // --- Interaction Glow ---
    float distToPlayer = length(in.worldPosition.xz - player.position.xz);
    float influenceNorm = distToPlayer / player.influenceRadius;

    // Bold ring at influence boundary
    float ringWidth = 0.2f;
    float ring = exp(-pow((influenceNorm - 1.0f) / ringWidth, 2.0f));

    // Strong inner glow with cubic falloff
    float innerGlow = saturate(1.0f - influenceNorm);
    innerGlow = innerGlow * innerGlow * innerGlow;

    // Multiple ripple layers during active interaction
    float ripple = 0.0f;
    if (player.isInteracting) {
        // Fast expanding rings
        float r1 = sin(distToPlayer * 12.0f - in.time * 8.0f) * 0.5f + 0.5f;
        // Slow deep pulses
        float r2 = sin(distToPlayer * 4.0f - in.time * 3.0f) * 0.5f + 0.5f;
        // Interference pattern
        ripple = (r1 * 0.6f + r2 * 0.4f) * innerGlow;
    }

    // Passive: subtle cyan ring; Active: bright pulsing white-cyan
    float3 glowColor;
    float glowIntensity;
    if (player.isInteracting) {
        // Pulse between cyan and white during interaction
        float pulse = sin(in.time * 4.0f) * 0.3f + 0.7f;
        glowColor = mix(float3(0.2, 0.8, 1.0), float3(1.0, 1.0, 1.0), pulse * innerGlow);
        glowIntensity = ring * 1.5f + innerGlow * 2.0f + ripple * 1.5f;
    } else {
        glowColor = float3(0.1, 0.5, 0.8);
        glowIntensity = ring * 0.4f + innerGlow * 0.1f;
    }

    litColor += glowColor * glowIntensity;

    // --- Resonance Orbs ---
    // Render orbs as glowing spots on the terrain surface
    for (int i = 0; i < resonance.orbCount && i < MAX_RESONANCE_ORBS; i++) {
        float3 orbPos = resonance.orbs[i].position;
        float orbDist = length(in.worldPosition.xz - orbPos.xz);
        float orbAge = resonance.currentTime - resonance.orbs[i].spawnTime;

        // Fade in over 1 second, pulse gently
        float fadeIn = saturate(orbAge * 1.0f);
        float pulse = 0.8f + 0.2f * sin(orbAge * 3.0f + orbDist * 2.0f);

        // Core glow (sharp center + soft halo)
        float core = exp(-orbDist * orbDist * 2.0f) * 3.0f;
        float halo = exp(-orbDist * orbDist * 0.15f) * 0.6f;
        float orbGlow = (core + halo) * fadeIn * pulse * resonance.orbs[i].intensity;

        // Vertical beam effect
        float beam = exp(-orbDist * orbDist * 0.8f) * 0.4f;

        float3 orbColor = resonance.orbs[i].color;
        litColor += orbColor * orbGlow + float3(1.0f) * beam * fadeIn * 0.3f;
    }

    // --- Atmospheric Fog + Sky ---
    float fogStart = 10.0f;
    float fogEnd = 50.0f;
    float fogFactor = saturate((in.distToCamera - fogStart) / (fogEnd - fogStart));
    fogFactor = fogFactor * fogFactor;

    // Sky gradient: deep indigo at zenith → warm haze at horizon
    float viewUp = saturate(V.y * 0.5f + 0.5f);
    float3 horizonColor = float3(0.12, 0.08, 0.15);   // Warm purple haze
    float3 zenithColor  = float3(0.02, 0.02, 0.08);   // Deep space indigo
    float3 skyColor = mix(horizonColor, zenithColor, viewUp * viewUp);

    // Subtle aurora bands (time-shifting color)
    float aurora = sin(V.y * 8.0f + in.time * 0.3f) * 0.5f + 0.5f;
    aurora *= exp(-abs(V.y - 0.3f) * 6.0f); // Concentrated near horizon
    float3 auroraColor = mix(float3(0.0, 0.3, 0.4), float3(0.2, 0.0, 0.4),
                             sin(in.time * 0.15f) * 0.5f + 0.5f);
    skyColor += auroraColor * aurora * 0.15f;

    // Fog blends toward sky
    float3 finalColor = mix(litColor, skyColor, fogFactor);

    // Filmic tone mapping (ACES-like, preserves colors better)
    finalColor = finalColor * (finalColor * 2.51f + 0.03f)
               / (finalColor * (finalColor * 2.43f + 0.59f) + 0.14f);

    return float4(finalColor, 1.0);
}
