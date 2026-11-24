// Samples a 3D texture through one sampler from a descriptor table. Each thread takes the texel
// centre (t + 0.5) / size on every axis, with t in [-size, 2 * size), so every address mode
// lands on a whole texel of the level the case expects.
Texture3D<float4> volume : register(t0);
SamplerState sampler_x : register(s0);
RWByteAddressBuffer dst : register(u0);
cbuffer Case : register(b0) { uint mode, value_bits, size, first; }

[RootSignature("DescriptorTable(SRV(t0)),DescriptorTable(Sampler(s0)),UAV(u0),"
               "RootConstants(num32BitConstants=4,b0)")]
[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint span = 3 * size;
    if (any(id >= span)) return;
    const float3 uvw = (float3(int3(id) - int(size)) + 0.5) / float(size);
    const float value = asfloat(value_bits);
    float4 color;
    if (mode == 0) {
        color = volume.SampleLevel(sampler_x, uvw, value);
    } else {
        // An isotropic footprint of `value` level-0 texels on an 8-texel base: LOD = log2(value).
        const float step = value / 8.0;
        color = volume.SampleGrad(sampler_x, uvw, float3(step, 0, 0), float3(0, step, 0));
    }
    const uint4 rgba = uint4(round(saturate(color) * 255.0));
    dst.Store(4 * (first + (id.z * span + id.y) * span + id.x),
              rgba.r | (rgba.g << 8) | (rgba.b << 16) | (rgba.a << 24));
}
