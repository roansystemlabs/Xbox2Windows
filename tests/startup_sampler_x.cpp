// Source-built CreateSamplerX desktop contract (#874): three 0x6C-byte descriptors go through
// d3d12x_sampler::to_desktop, become desktop samplers in one shader-visible sampler heap, and
// sample a 3D texture whose texels encode level and position. The expected words come from an
// independent CPU addressing model. The descriptor bytes are written here from the offsets in
// this file, not from the translator's constants.
#include "bvh_fixture.h"
#include "startup_sampler_x_cs.h"

#include "../src/d3d12x_sampler.h"

#include <algorithm>
#include <cmath>

using namespace bvh_fixture;

namespace {
constexpr UINT base_size = 8;
constexpr UINT mip_count = 4;
constexpr Word guard = 0x5a17c3e9u;

// Console descriptor layout used by this fixture (XCL ABI notes, device slot #61).
constexpr std::size_t console_bytes = 0x6c;
constexpr std::size_t at_address_u = 0x18, at_address_v = 0x1c, at_address_w = 0x20;
constexpr std::size_t at_min_lod = 0x24, at_max_lod = 0x28, at_bias = 0x2c;
constexpr std::size_t at_anisotropy = 0x34, at_comparison = 0x40;

enum Address : std::uint32_t { wrap = 1, mirror = 2, clamp = 3, mirror_once = 5 };

struct SamplerCase {
    const char* name;
    Address u, v, w;
    float min_lod, max_lod, bias;
    std::uint32_t anisotropy, comparison;
};

// Each sampler gives the three axes three different modes, so an exchanged axis changes texels.
constexpr SamplerCase samplers[] = {
    {"a", mirror, clamp, wrap, 1.0f, 2.0f, 0.0f, 4, D3D12_COMPARISON_FUNC_LESS},
    {"b", clamp, wrap, mirror_once, 0.0f, 0.0f, 0.0f, 1, D3D12_COMPARISON_FUNC_ALWAYS},
    {"c", wrap, mirror, clamp, 0.0f, 3.0f, 1.0f, 16, D3D12_COMPARISON_FUNC_EQUAL},
};
constexpr UINT sampler_count = sizeof(samplers) / sizeof(samplers[0]);

enum class Mode : UINT { level = 0, gradient = 1 };

struct DrawCase {
    UINT sampler;
    Mode mode;
    float value;  // LOD for level; isotropic footprint in level-0 texels for gradient
    UINT level;   // expected level after bias and the MinLOD/MaxLOD clamp
};

constexpr DrawCase draws[] = {
    {0, Mode::level, 0.0f, 1},     // below MinLOD
    {0, Mode::level, 1.0f, 1},
    {0, Mode::level, 2.0f, 2},
    {0, Mode::level, 3.0f, 2},     // above MaxLOD
    {0, Mode::gradient, 1.0f, 1},  // LOD 0, clamped to MinLOD
    {0, Mode::gradient, 8.0f, 2},  // LOD 3, clamped to MaxLOD
    {1, Mode::level, 2.0f, 0},     // MaxLOD 0
    {1, Mode::gradient, 4.0f, 0},
    {2, Mode::gradient, 1.0f, 1},  // LOD 0 + bias 1
    {2, Mode::gradient, 2.0f, 2},  // LOD 1 + bias 1
    {2, Mode::gradient, 4.0f, 3},  // LOD 2 + bias 1 = MaxLOD
};
constexpr UINT draw_count = sizeof(draws) / sizeof(draws[0]);

UINT extent(UINT level) { return (std::max)(1u, base_size >> level); }

Word texel(UINT level, UINT x, UINT y, UINT z)
{
    return (x | level << 4) | (y | 0x40u) << 8 | (z | 0x80u) << 16 | (0xf0u | level << 1) << 24;
}

// D3D addressing of a texel index t in [-size, 2 * size).
UINT address(Address mode, int t, int size)
{
    switch (mode) {
    case wrap: return static_cast<UINT>(((t % size) + size) % size);
    case clamp: return static_cast<UINT>((std::min)((std::max)(t, 0), size - 1));
    case mirror: {
        const int period = 2 * size;
        const int m = ((t % period) + period) % period;
        return static_cast<UINT>(m < size ? m : period - 1 - m);
    }
    case mirror_once: {
        const int m = t < 0 ? -t - 1 : t;
        return static_cast<UINT>((std::min)(m, size - 1));
    }
    }
    throw std::runtime_error("Unknown address mode in the sampler fixture");
}

void put_word(std::array<unsigned char, console_bytes>& bytes, std::size_t at, std::uint32_t value)
{
    std::memcpy(bytes.data() + at, &value, sizeof(value));
}

void put_float(std::array<unsigned char, console_bytes>& bytes, std::size_t at, float value)
{
    std::memcpy(bytes.data() + at, &value, sizeof(value));
}

std::array<unsigned char, console_bytes> console_desc(const SamplerCase& s, UINT index)
{
    // Fill the undocumented bytes with a pattern so the translation must ignore them.
    std::array<unsigned char, console_bytes> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<unsigned char>(0x3b * (i + 1) + index);
    put_word(bytes, at_address_u, s.u);
    put_word(bytes, at_address_v, s.v);
    put_word(bytes, at_address_w, s.w);
    put_float(bytes, at_min_lod, s.min_lod);
    put_float(bytes, at_max_lod, s.max_lod);
    put_float(bytes, at_bias, s.bias);
    put_word(bytes, at_anisotropy, s.anisotropy);
    put_word(bytes, at_comparison, s.comparison);
    return bytes;
}

ComPtr<ID3D12Resource> make_volume(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                                   ComPtr<ID3D12Resource>& staging_buffer)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = desc.Height = base_size;
    desc.DepthOrArraySize = base_size;
    desc.MipLevels = mip_count;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    auto volume = resource(device, desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, mip_count> layouts{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&desc, 0, mip_count, 0, layouts.data(), nullptr, nullptr, &bytes);
    std::vector<Word> staging((bytes + 3) / 4, guard);
    for (UINT level = 0; level < mip_count; ++level) {
        const UINT n = extent(level);
        const auto& f = layouts[level];
        for (UINT z = 0; z < n; ++z)
            for (UINT y = 0; y < n; ++y)
                for (UINT x = 0; x < n; ++x)
                    staging[(f.Offset + (UINT64{z} * f.Footprint.Height + y) * f.Footprint.RowPitch) / 4 + x] =
                        texel(level, x, y, z);
    }
    staging_buffer = upload(device, staging);
    for (UINT level = 0; level < mip_count; ++level) {
        D3D12_TEXTURE_COPY_LOCATION dest{}, src{};
        dest.pResource = volume.Get();
        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dest.SubresourceIndex = level;
        src.pResource = staging_buffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layouts[level];
        list->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
    }
    transition(list, volume.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return volume;
}
}  // namespace

// With `swapped`, the descriptors of samplers a and b exchange heap slots while the draws keep
// their own references, so the readback must differ from the reference.
void check_startup_sampler_x(ID3D12Device* device, bool swapped)
{
    Kernel shader(device, {startup_sampler_x_cs, sizeof(startup_sampler_x_cs)});

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    heap_desc.NumDescriptors = sampler_count;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&sampler_heap)), "Create CreateSamplerX heap");
    const UINT sampler_stride = device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    for (UINT i = 0; i < sampler_count; ++i) {
        const auto bytes = console_desc(samplers[i], i);
        D3D12_SAMPLER_DESC desc{};
        const auto refusal = d3d12x_sampler::to_desktop(bytes.data(), desc);
        if (refusal != d3d12x_sampler::Refusal::none) {
            std::fprintf(stderr, "sampler %s refused: %s\n", samplers[i].name, d3d12x_sampler::refusal_name(refusal));
            throw std::runtime_error("CreateSamplerX translation refused a fixture descriptor");
        }
        std::printf("startup_sampler_x sampler=%s filter=0x%x address=%d,%d,%d lod=[%g,%g] bias=%g aniso=%u cmp=%d\n",
                    samplers[i].name, static_cast<unsigned>(desc.Filter), desc.AddressU, desc.AddressV,
                    desc.AddressW, desc.MinLOD, desc.MaxLOD, desc.MipLODBias, desc.MaxAnisotropy,
                    desc.ComparisonFunc);
        const UINT slot = swapped && i < 2 ? 1 - i : i;
        auto handle = sampler_heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T{slot} * sampler_stride;
        device->CreateSampler(&desc, handle);
    }

    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> view_heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&view_heap)), "Create CreateSamplerX SRV heap");

    std::vector<Word> expected;
    std::array<UINT, draw_count> firsts{};
    for (UINT d = 0; d < draw_count; ++d) {
        const auto& draw = draws[d];
        const auto& s = samplers[draw.sampler];
        const int size = static_cast<int>(extent(draw.level));
        firsts[d] = static_cast<UINT>(expected.size());
        for (int k = 0; k < 3 * size; ++k)
            for (int j = 0; j < 3 * size; ++j)
                for (int i = 0; i < 3 * size; ++i)
                    expected.push_back(texel(draw.level, address(s.u, i - size, size), address(s.v, j - size, size),
                                             address(s.w, k - size, size)));
    }
    const UINT64 bytes = expected.size() * sizeof(Word);
    auto guards = upload(device, std::vector<Word>(expected.size(), guard));
    auto output = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto back = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    ComputeCommands cmd(device, shader.pipeline.Get());
    auto* list = cmd.list.Get();
    ComPtr<ID3D12Resource> staging;
    auto volume = make_volume(device, list, staging);
    device->CreateShaderResourceView(volume.Get(), nullptr, view_heap->GetCPUDescriptorHandleForHeapStart());
    transition(list, output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(output.Get(), guards.Get());
    transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[]{view_heap.Get(), sampler_heap.Get()};
    list->SetDescriptorHeaps(2, heaps);
    list->SetComputeRootSignature(shader.root.Get());
    list->SetComputeRootDescriptorTable(0, view_heap->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
    for (UINT d = 0; d < draw_count; ++d) {
        const auto& draw = draws[d];
        auto gpu = sampler_heap->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += UINT64{draw.sampler} * sampler_stride;
        list->SetComputeRootDescriptorTable(1, gpu);
        Word value_bits = 0;
        std::memcpy(&value_bits, &draw.value, sizeof(value_bits));
        const UINT size = extent(draw.level);
        const UINT params[]{static_cast<UINT>(draw.mode), value_bits, size, firsts[d]};
        list->SetComputeRoot32BitConstants(3, 4, params, 0);
        const UINT groups = (3 * size + 3) / 4;
        list->Dispatch(groups, groups, groups);
    }
    transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(back.Get(), output.Get());
    cmd.submit_and_wait(device);
    const auto actual = read(back.Get(), expected.size());

    UINT total_mismatches = 0;
    for (UINT d = 0; d < draw_count; ++d) {
        const auto& draw = draws[d];
        const UINT end = d + 1 < draw_count ? firsts[d + 1] : static_cast<UINT>(expected.size());
        UINT mismatches = 0, first_bad = end;
        for (UINT w = firsts[d]; w < end; ++w) {
            if (actual[w] == expected[w]) continue;
            if (!mismatches++) first_bad = w;
        }
        total_mismatches += mismatches;
        std::printf("startup_sampler_x draw=%u sampler=%s mode=%s value=%g level=%u words=%u mismatches=%u",
                    d, samplers[draw.sampler].name, draw.mode == Mode::level ? "level" : "gradient", draw.value,
                    draw.level, end - firsts[d], mismatches);
        if (mismatches)
            std::printf(" first=%u expected=%08x actual=%08x", first_bad - firsts[d], expected[first_bad],
                        actual[first_bad]);
        std::printf("\n");
    }
    std::printf("startup_sampler_x swapped=%u draws=%u words=%zu mismatches=%u\n", static_cast<unsigned>(swapped),
                draw_count, expected.size(), total_mismatches);
    if (!swapped && total_mismatches)
        throw std::runtime_error("CreateSamplerX readback differs from the CPU addressing model");
    if (swapped && !total_mismatches)
        throw std::runtime_error("Swapped CreateSamplerX descriptors did not change the readback");
}
