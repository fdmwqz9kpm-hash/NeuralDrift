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
    float         species;   // 0-7 species ID for shape variation
    float         scale;     // Size multiplier (generation/energy based)
};

struct CreatureVertexOut {
    float4 position [[position]];
    float3 color;
    float  energy;
    float  age;
    float  heading;
    float  speed;
    float  species;
    float  scale;
    float  pointSize [[point_size]];
};

// --- Creature Vertex Shader ---
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
    out.heading = c.heading;
    out.speed = c.speed;
    out.species = c.species;
    out.scale = c.scale;

    // Much larger sprites — visible from distance
    float dist = length(uniforms.cameraPosition - worldPos);
    float energySize = 0.5 + c.energy * 0.5;
    float baseSize = (20.0 + c.scale * 15.0) * energySize;
    out.pointSize = clamp(baseSize * 25.0 / max(dist, 1.0), 4.0, 120.0);

    return out;
}

// --- Organic shape functions ---

// Rotating blob shape (species 0-1)
inline float blobShape(float2 uv, float time, float species) {
    float angle = atan2(uv.y, uv.x);
    float r = length(uv);
    // Organic undulating boundary
    float boundary = 0.6 + 0.15 * sin(angle * 3.0 + time * 2.0)
                         + 0.08 * sin(angle * 5.0 - time * 3.0)
                         + 0.05 * cos(angle * 7.0 + time * 1.5);
    return smoothstep(boundary, boundary - 0.15, r);
}

// Star/radial shape (species 2-3)
inline float starShape(float2 uv, float time, float arms) {
    float angle = atan2(uv.y, uv.x);
    float r = length(uv);
    float star = 0.5 + 0.2 * cos(angle * arms + time * 1.5);
    return smoothstep(star, star - 0.12, r);
}

// Jellyfish/tentacle shape (species 4-5)
inline float jellyShape(float2 uv, float time) {
    float2 p = uv;
    // Dome top
    float dome = smoothstep(0.55, 0.4, length(float2(p.x, p.y - 0.1)));
    // Tentacles below
    float tentacles = 0.0;
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float tx = p.x * 4.0 + sin(time * 2.0 + fi * 1.5) * 0.5;
        float wave = sin(tx * 3.0 + time * 3.0 + fi * 2.0) * 0.15;
        float ty = p.y + 0.3 + wave;
        tentacles += exp(-tx * tx * 3.0 - ty * ty * 8.0) * 0.5;
    }
    return saturate(dome + tentacles * step(0.0, -uv.y + 0.1));
}

// Butterfly/wing shape (species 6-7)
inline float wingShape(float2 uv, float time) {
    float2 p = float2(abs(uv.x), uv.y); // Mirror
    // Wing lobes
    float wing = exp(-length(p - float2(0.25, 0.1)) * 4.0)
               + exp(-length(p - float2(0.15, -0.15)) * 5.0) * 0.7;
    // Flap animation
    float flap = 0.8 + 0.2 * sin(time * 8.0);
    wing *= flap;
    // Body
    float body = exp(-uv.x * uv.x * 30.0 - uv.y * uv.y * 6.0) * 0.8;
    return saturate(wing + body);
}

// --- Creature Fragment Shader ---
fragment float4 creatureFragment(
    CreatureVertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    float2 uv = pointCoord * 2.0 - 1.0;

    // Rotate UV by heading so creatures face their movement direction
    float c = cos(in.heading);
    float s = sin(in.heading);
    float2 ruv = float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);

    // Select shape based on species
    float shape = 0.0;
    int sp = int(in.species) % 8;
    if (sp < 2) {
        shape = blobShape(ruv, in.age, in.species);
    } else if (sp < 4) {
        shape = starShape(ruv, in.age, 3.0 + float(sp));
    } else if (sp < 6) {
        shape = jellyShape(ruv, in.age);
    } else {
        shape = wingShape(ruv, in.age);
    }

    if (shape < 0.01) discard_fragment();

    // Inner glow: bright core
    float dist = length(uv);
    float core = exp(-dist * dist * 4.0);

    // Color with energy-based brightness
    float brightness = 0.6 + in.energy * 0.6;
    float3 color = in.color * brightness;

    // White-hot center
    color = mix(color, float3(1.0), core * 0.5 * shape);

    // Outer glow halo (extends beyond shape)
    float halo = exp(-dist * dist * 1.5) * 0.3;
    color += in.color * halo;

    // Life pulse
    float pulse = 0.85 + 0.15 * sin(in.age * 4.0 + in.species * 1.5);
    color *= pulse;

    // Speed trails: stretch glow behind creature when moving fast
    if (in.speed > 0.5) {
        float2 trailUV = float2(ruv.x, ruv.y + 0.3); // Behind
        float trail = exp(-trailUV.x * trailUV.x * 4.0 - trailUV.y * trailUV.y * 1.0);
        trail *= (in.speed - 0.5) * 0.4;
        color += in.color * trail * 0.5;
        shape = max(shape, trail * 0.5);
    }

    float alpha = saturate(shape + halo * 0.5);
    return float4(color, alpha);
}
