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

// --- Curated color palette ---
// Maps neural output into beautiful iridescent/bioluminescent tones
inline float3 neuralPalette(float3 raw, float height, float time) {
    // Palette: deep ocean → teal → magenta → gold based on neural + height
    float t = raw.x * 0.4f + raw.y * 0.3f + raw.z * 0.3f; // Neural hue driver
    float h = saturate(height * 0.12f + 0.5f);

    // Four-stop gradient: deep blue → cyan → magenta → warm gold
    float3 c0 = float3(0.02, 0.05, 0.15);   // Deep ocean
    float3 c1 = float3(0.0, 0.35, 0.45);    // Teal
    float3 c2 = float3(0.5, 0.05, 0.4);     // Magenta
    float3 c3 = float3(0.9, 0.6, 0.15);     // Warm gold

    // Blend based on height + neural signal
    float blend = saturate(t + h * 0.5f + sin(time * 0.08f) * 0.15f);
    float3 color;
    if (blend < 0.33f) {
        color = mix(c0, c1, blend / 0.33f);
    } else if (blend < 0.66f) {
        color = mix(c1, c2, (blend - 0.33f) / 0.33f);
    } else {
        color = mix(c2, c3, (blend - 0.66f) / 0.34f);
    }

    // Iridescence: slight color shift based on view angle (computed later)
    return color;
}

// --- Fragment Shader ---
// Bioluminescent neural landscape with rim lighting, emission, and atmosphere.
fragment float4 neuralColorFragment(
    VertexOut               in           [[stage_in]],
    device const float*     colorWeights [[buffer(BufferIndexColorWeights)]],
    constant PlayerState&   player       [[buffer(BufferIndexPlayerState)]],
    constant ResonanceData& resonance    [[buffer(BufferIndexResonance)]]
) {
    float3 N = normalize(in.normal);
    float3 V = normalize(in.viewDirection);
    float NdotV = max(dot(N, V), 0.0f);

    // Neural network raw color (sigmoid 0-1)
    float3 neuralRaw = evaluateColorNetwork(
        in.worldPosition, N, V, in.time, colorWeights);

    // Map through curated palette
    float3 baseColor = neuralPalette(neuralRaw, in.worldPosition.y, in.time);

    // --- Iridescence: color shifts at glancing angles ---
    float fresnel = pow(1.0f - NdotV, 3.0f);
    float3 iridescentShift = float3(0.15, -0.1, 0.25) * fresnel;
    baseColor = saturate(baseColor + iridescentShift);

    // --- Lighting ---
    float3 lightDir = normalize(float3(0.3, 0.7, 0.4));
    float3 lightColor = float3(1.0, 0.95, 0.85);

    float NdotL = max(dot(N, lightDir), 0.0f);
    float3 H = normalize(lightDir + V);
    float NdotH = max(dot(N, H), 0.0f);
    float specular = pow(NdotH, 48.0f) * 0.5f;

    // Hemisphere ambient (sky vs ground)
    float3 skyAmbient  = float3(0.08, 0.06, 0.15);
    float3 gndAmbient  = float3(0.03, 0.04, 0.06);
    float3 ambientLight = mix(gndAmbient, skyAmbient, N.y * 0.5f + 0.5f);

    float3 litColor = baseColor * (ambientLight + NdotL * 0.7f * lightColor)
                    + specular * lightColor * 0.4f;

    // --- Rim lighting: edges glow like bioluminescence ---
    float rim = pow(1.0f - NdotV, 4.0f) * 0.6f;
    float3 rimColor = mix(float3(0.1, 0.4, 0.8), float3(0.6, 0.1, 0.5),
                          sin(in.worldPosition.x * 0.3f + in.time * 0.2f) * 0.5f + 0.5f);
    litColor += rimColor * rim;

    // --- Terrain emission: surface emits soft glow from neural activity ---
    float neuralEnergy = (neuralRaw.x + neuralRaw.y + neuralRaw.z) / 3.0f;
    float emission = pow(neuralEnergy, 2.0f) * 0.2f;
    float3 emissionColor = baseColor * emission;
    litColor += emissionColor;

    // --- Grid lines: subtle wireframe overlay for neural-network feel ---
    float2 gridUV = fract(in.worldPosition.xz * 0.8f);
    float gridLine = smoothstep(0.02f, 0.0f, min(gridUV.x, gridUV.y))
                   + smoothstep(0.98f, 1.0f, max(gridUV.x, gridUV.y));
    gridLine = saturate(gridLine) * 0.08f;
    litColor += float3(0.2, 0.5, 0.8) * gridLine;

    // --- Interaction shockwave ---
    float distToPlayer = length(in.worldPosition.xz - player.position.xz);
    float influenceNorm = distToPlayer / player.influenceRadius;

    if (player.isInteracting) {
        // Expanding rings of energy
        float wave1 = sin(distToPlayer * 10.0f - in.time * 10.0f) * 0.5f + 0.5f;
        float wave2 = sin(distToPlayer * 5.0f - in.time * 4.0f) * 0.5f + 0.5f;
        float waveMask = exp(-distToPlayer * 0.15f); // Fade with distance

        float3 waveColor = mix(float3(0.0, 0.8, 1.0), float3(1.0, 0.3, 0.8),
                                wave1 * 0.5f + 0.5f);
        litColor += waveColor * wave1 * waveMask * 0.8f;
        litColor += float3(1.0) * wave2 * waveMask * 0.15f; // White flash
    }

    // Passive proximity glow (always visible, subtle)
    float proximity = exp(-influenceNorm * influenceNorm * 2.0f) * 0.15f;
    litColor += float3(0.1, 0.3, 0.5) * proximity;

    // --- Resonance Orbs ---
    for (int i = 0; i < resonance.orbCount && i < MAX_RESONANCE_ORBS; i++) {
        float3 orbPos = resonance.orbs[i].position;
        float orbDist = length(in.worldPosition.xz - orbPos.xz);
        float orbAge = resonance.currentTime - resonance.orbs[i].spawnTime;

        float fadeIn = saturate(orbAge);
        float pulse = 0.7f + 0.3f * sin(orbAge * 2.5f + orbDist * 1.5f);

        // Bright core + wide soft halo + ring
        float core = exp(-orbDist * orbDist * 3.0f) * 4.0f;
        float halo = exp(-orbDist * orbDist * 0.08f) * 0.4f;
        float orbRing = exp(-pow(orbDist - 1.5f, 2.0f) * 2.0f) * 0.5f;

        float orbGlow = (core + halo + orbRing) * fadeIn * pulse
                       * resonance.orbs[i].intensity;

        float3 orbColor = resonance.orbs[i].color;
        litColor += orbColor * orbGlow;
        litColor += float3(1.0f) * core * fadeIn * 0.2f; // White center
    }

    // --- Atmospheric Fog + Sky ---
    float fogStart = 8.0f;
    float fogEnd = 45.0f;
    float fogFactor = saturate((in.distToCamera - fogStart) / (fogEnd - fogStart));
    fogFactor = fogFactor * fogFactor;

    // Sky: deep space with color-shifting horizon glow
    float viewUp = saturate(V.y * 0.5f + 0.5f);
    float3 horizonColor = float3(0.08, 0.04, 0.12) +
        float3(0.06, 0.02, 0.04) * sin(in.time * 0.05f + 1.0f);
    float3 zenithColor  = float3(0.005, 0.005, 0.02);
    float3 skyColor = mix(horizonColor, zenithColor, pow(viewUp, 1.5f));

    // Aurora / nebula bands
    float nebula1 = sin(V.y * 6.0f + in.time * 0.2f + V.x * 2.0f) * 0.5f + 0.5f;
    float nebula2 = sin(V.y * 10.0f - in.time * 0.15f) * 0.5f + 0.5f;
    float nebulaMask = exp(-pow(V.y - 0.25f, 2.0f) * 8.0f);
    float3 nebulaColor = mix(float3(0.0, 0.15, 0.3), float3(0.3, 0.0, 0.25),
                              nebula1);
    skyColor += nebulaColor * nebulaMask * nebula2 * 0.2f;

    // Scattered stars (static noise)
    float starField = fract(sin(dot(V.xy * 400.0f, float2(12.9898, 78.233))) * 43758.5453f);
    starField = step(0.998f, starField) * viewUp * 0.4f;
    skyColor += float3(starField);

    float3 finalColor = mix(litColor, skyColor, fogFactor);

    // ACES filmic tone mapping
    finalColor = finalColor * (finalColor * 2.51f + 0.03f)
               / (finalColor * (finalColor * 2.43f + 0.59f) + 0.14f);

    // Subtle vignette
    float2 screenUV = in.position.xy / float2(2560.0f, 1440.0f); // approximate
    float2 vignetteUV = (screenUV - 0.5f) * 2.0f;
    float vignette = 1.0f - dot(vignetteUV, vignetteUV) * 0.15f;
    finalColor *= saturate(vignette);

    return float4(finalColor, 1.0);
}
