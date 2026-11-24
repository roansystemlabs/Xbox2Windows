// CPU checks for d3d12x_sampler::to_desktop (#874): accepted fields, ignored bytes, and one
// refusal per rejected field with the output left untouched.
#include "../src/d3d12x_sampler.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
using Desc = std::array<unsigned char, 0x6c>;
using d3d12x_sampler::Refusal;

int failures = 0;

void expect(bool ok, const char* what)
{
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

void put(Desc& d, std::size_t at, std::uint32_t v) { std::memcpy(d.data() + at, &v, sizeof(v)); }
void putf(Desc& d, std::size_t at, float v) { std::memcpy(d.data() + at, &v, sizeof(v)); }

// Offsets restated from the XCL notes so a moved constant in the header is caught.
Desc valid(unsigned char fill)
{
    Desc d;
    d.fill(fill);
    put(d, 0x18, D3D12_TEXTURE_ADDRESS_MODE_MIRROR);
    put(d, 0x1c, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    put(d, 0x20, D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE);
    putf(d, 0x24, 0.5f);
    putf(d, 0x28, 7.25f);
    putf(d, 0x2c, -1.5f);
    put(d, 0x34, 8);
    put(d, 0x40, D3D12_COMPARISON_FUNC_GREATER_EQUAL);
    return d;
}

D3D12_SAMPLER_DESC sentinel()
{
    D3D12_SAMPLER_DESC s{};
    std::memset(&s, 0xcd, sizeof(s));
    return s;
}

void refuse(const char* label, Desc d, Refusal want)
{
    auto out = sentinel();
    const auto before = out;
    const auto got = d3d12x_sampler::to_desktop(d.data(), out);
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s refused as %s, expected %s\n", label, d3d12x_sampler::refusal_name(got),
                     d3d12x_sampler::refusal_name(want));
        ++failures;
    }
    expect(std::memcmp(&out, &before, sizeof(out)) == 0, label);
    std::printf("refused %-28s %s\n", label, d3d12x_sampler::refusal_name(got));
}
}  // namespace

int main()
{
    // Accepted fields land in the desktop descriptor; the filter is the fixed policy.
    {
        const auto d = valid(0x00);
        auto out = sentinel();
        expect(d3d12x_sampler::to_desktop(d.data(), out) == Refusal::none, "valid descriptor accepted");
        expect(out.Filter == D3D12_FILTER_ANISOTROPIC, "filter is the fixed anisotropic policy");
        expect(out.AddressU == D3D12_TEXTURE_ADDRESS_MODE_MIRROR, "AddressU from +0x18");
        expect(out.AddressV == D3D12_TEXTURE_ADDRESS_MODE_CLAMP, "AddressV from +0x1c");
        expect(out.AddressW == D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE, "AddressW from +0x20");
        expect(out.MinLOD == 0.5f, "MinLOD from +0x24");
        expect(out.MaxLOD == 7.25f, "MaxLOD from +0x28");
        expect(out.MipLODBias == -1.5f, "MipLODBias from +0x2c");
        expect(out.MaxAnisotropy == 8, "MaxAnisotropy from +0x34");
        expect(out.ComparisonFunc == D3D12_COMPARISON_FUNC_NONE, "ComparisonFunc validated, not applied");
        const float zero[4]{};
        expect(std::memcmp(out.BorderColor, zero, sizeof(zero)) == 0, "border colour is zero");
        std::printf("applied fields: AddressU/V/W MinLOD MaxLOD MipLODBias MaxAnisotropy; "
                    "validated only: ComparisonFunc; fixed: Filter=ANISOTROPIC\n");
    }

    // Bytes outside the eight documented fields do not change the result.
    {
        const auto a = valid(0x00);
        const auto b = valid(0xa5);
        D3D12_SAMPLER_DESC da{}, db{};
        expect(d3d12x_sampler::to_desktop(a.data(), da) == Refusal::none, "zero-filled accepted");
        expect(d3d12x_sampler::to_desktop(b.data(), db) == Refusal::none, "pattern-filled accepted");
        expect(std::memcmp(&da, &db, sizeof(da)) == 0, "undocumented bytes are ignored");
        std::printf("ignored bytes: %zu of %zu\n", std::size_t{0x6c - 8 * 4}, std::size_t{0x6c});
    }

    // Boundary values that stay accepted.
    {
        auto d = valid(0);
        put(d, 0x34, 1);
        putf(d, 0x2c, D3D12_MIP_LOD_BIAS_MIN);
        putf(d, 0x24, 3.0f);
        putf(d, 0x28, 3.0f);
        D3D12_SAMPLER_DESC out{};
        expect(d3d12x_sampler::to_desktop(d.data(), out) == Refusal::none, "anisotropy 1, bias min, MinLOD == MaxLOD");
        put(d, 0x34, D3D12_MAX_MAXANISOTROPY);
        putf(d, 0x2c, D3D12_MIP_LOD_BIAS_MAX);
        putf(d, 0x28, D3D12_FLOAT32_MAX);
        put(d, 0x40, D3D12_COMPARISON_FUNC_NEVER);
        expect(d3d12x_sampler::to_desktop(d.data(), out) == Refusal::none, "anisotropy 16, bias max, MaxLOD FLT_MAX");
        for (std::uint32_t mode : {1u, 2u, 3u, 5u}) {
            put(d, 0x18, mode);
            expect(d3d12x_sampler::to_desktop(d.data(), out) == Refusal::none && out.AddressU == mode,
                   "every non-border address mode");
        }
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    {
        auto out = sentinel();
        const auto before = out;
        expect(d3d12x_sampler::to_desktop(nullptr, out) == Refusal::null_desc, "null descriptor");
        expect(std::memcmp(&out, &before, sizeof(out)) == 0, "null descriptor leaves output untouched");
    }
    {
        auto d = valid(0); put(d, 0x18, 0); refuse("AddressU 0", d, Refusal::address_u);
        d = valid(0); put(d, 0x1c, 6); refuse("AddressV 6", d, Refusal::address_v);
        d = valid(0); put(d, 0x20, 0xffffffffu); refuse("AddressW 0xffffffff", d, Refusal::address_w);
        d = valid(0); put(d, 0x18, D3D12_TEXTURE_ADDRESS_MODE_BORDER); refuse("AddressU BORDER", d, Refusal::border_address);
        d = valid(0); put(d, 0x20, D3D12_TEXTURE_ADDRESS_MODE_BORDER); refuse("AddressW BORDER", d, Refusal::border_address);
        d = valid(0); putf(d, 0x24, nan); refuse("MinLOD NaN", d, Refusal::min_lod);
        d = valid(0); putf(d, 0x28, inf); refuse("MaxLOD +inf", d, Refusal::max_lod);
        d = valid(0); putf(d, 0x24, 2.0f); putf(d, 0x28, 1.0f); refuse("MinLOD > MaxLOD", d, Refusal::lod_order);
        d = valid(0); putf(d, 0x2c, 16.0f); refuse("MipLODBias 16", d, Refusal::mip_lod_bias);
        d = valid(0); putf(d, 0x2c, -16.5f); refuse("MipLODBias -16.5", d, Refusal::mip_lod_bias);
        d = valid(0); putf(d, 0x2c, nan); refuse("MipLODBias NaN", d, Refusal::mip_lod_bias);
        d = valid(0); put(d, 0x34, 0); refuse("MaxAnisotropy 0", d, Refusal::max_anisotropy);
        d = valid(0); put(d, 0x34, 17); refuse("MaxAnisotropy 17", d, Refusal::max_anisotropy);
        d = valid(0); put(d, 0x40, 0); refuse("ComparisonFunc 0", d, Refusal::comparison);
        d = valid(0); put(d, 0x40, 9); refuse("ComparisonFunc 9", d, Refusal::comparison);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("d3d12x_sampler contract: all checks passed\n");
    return 0;
}
