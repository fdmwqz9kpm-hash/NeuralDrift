#include <metal_stdlib>
using namespace metal;

#import "../Renderer/ShaderTypes.h"

// Per-creature instance data passed from CPU
struct CreatureInstance {
    packed_float3 position;  // World XYZ (Y = terrain height)
    float         energy;
    packed_float3 color;     // Species color
    float         age;
    float         heading;
    float         speed;
    float         _pad0;
    float         _pad1;
};

struct CreatureVertexOut {
    float4 position [[position]];
    float3 color;
    float  energy;
    float  age;
    float  glow;
    float  pointSize [[point_size]];
};

// --- Creature Vertex Shader ---
// Renders each creature as a glowing point sprite on the terrain surface.
vertex CreatureVertexOut creatureVertex(
    uint                     vertexID  [[vertex_id]],
    uint                     instID    [[instance_id]],
    constant Uniforms&       uniforms  [[buffer(0)]],
    constant CreatureInstance* creatures [[buffer(1)]]
) {
    CreatureInstance c = creatures[instID];
    float3 worldPos = float3(c.position[0], c.position[1], c.position[2]);

    CreatureVertexOut out;
    out.position = uniforms.modelViewProjection * float4(worldPos, 1.0);
    out.color = float3(c.color[0], c.color[1], c.color[2]);
    out.energy = c.energy;
    out.age = c.age;

    // Point size based on distance (perspective) and energy
    float dist = length(uniforms.cameraPosition - worldPos);
    float baseSize = 8.0 + c.energy * 4.0;
    out.pointSize = max(2.0, baseSize * 15.0 / max(dist, 1.0));

    // Glow intensity: higher energy = brighter
    out.glow = 0.5 + c.energy * 0.5;

    return out;
}

// --- Creature Fragment Shader ---
// Renders point sprites as soft glowing circles with species color.
fragment float4 creatureFragment(
    CreatureVertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    // Soft circle
    float2 uv = pointCoord * 2.0 - 1.0;
    float dist = length(uv);

    // Sharp core + soft glow halo
    float core = exp(-dist * dist * 8.0);
    float halo = exp(-dist * dist * 2.0) * 0.4;
    float alpha = core + halo;

    if (alpha < 0.01) discard_fragment();

    // Color: species color with energy-based brightness
    float3 color = in.color * in.glow;

    // White hot center
    color = mix(color, float3(1.0), core * 0.6);

    // Pulse based on age (life sign)
    float pulse = 0.9 + 0.1 * sin(in.age * 5.0);
    color *= pulse;

    return float4(color, alpha);
}
