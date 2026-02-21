#include <metal_stdlib>
using namespace metal;

// Simple fullscreen blit shader — copies the MetalFX upscaled texture to the drawable.

struct BlitVertexOut {
    float4 position [[position]];
    float2 texcoord;
};

// Fullscreen triangle (no vertex buffer needed)
vertex BlitVertexOut blitVertex(uint vertexID [[vertex_id]]) {
    BlitVertexOut out;
    // Generate fullscreen triangle from vertex ID
    out.texcoord = float2((vertexID << 1) & 2, vertexID & 2);
    out.position = float4(out.texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return out;
}

fragment float4 blitFragment(BlitVertexOut in [[stage_in]],
                              texture2d<float> sourceTexture [[texture(0)]]) {
    constexpr sampler s(min_filter::linear, mag_filter::linear);
    return sourceTexture.sample(s, in.texcoord);
}
