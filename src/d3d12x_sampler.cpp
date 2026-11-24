#include "d3d12x_sampler.h"

#include <cmath>
#include <cstring>

namespace d3d12x_sampler {
namespace {

std::uint32_t word_at(const unsigned char* bytes, std::size_t offset) noexcept
{
    std::uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

float float_at(const unsigned char* bytes, std::size_t offset) noexcept
{
    float value = 0.0f;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

// WRAP, MIRROR, CLAMP and MIRROR_ONCE. BORDER needs a border colour, which has no documented
// offset in the descriptor.
Refusal address_mode(std::uint32_t value, Refusal invalid, D3D12_TEXTURE_ADDRESS_MODE& mode) noexcept
{
    switch (value) {
    case D3D12_TEXTURE_ADDRESS_MODE_WRAP:
    case D3D12_TEXTURE_ADDRESS_MODE_MIRROR:
    case D3D12_TEXTURE_ADDRESS_MODE_CLAMP:
    case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE:
        mode = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(value);
        return Refusal::none;
    case D3D12_TEXTURE_ADDRESS_MODE_BORDER:
        return Refusal::border_address;
    default:
        return invalid;
    }
}

}  // namespace

Refusal to_desktop(const void* console_desc, D3D12_SAMPLER_DESC& out) noexcept
{
    if (!console_desc) return Refusal::null_desc;
    unsigned char bytes[desc_bytes];
    std::memcpy(bytes, console_desc, sizeof(bytes));

    D3D12_SAMPLER_DESC desc{};
    desc.Filter = fixed_filter;
    if (const auto r = address_mode(word_at(bytes, address_u_offset), Refusal::address_u, desc.AddressU);
        r != Refusal::none)
        return r;
    if (const auto r = address_mode(word_at(bytes, address_v_offset), Refusal::address_v, desc.AddressV);
        r != Refusal::none)
        return r;
    if (const auto r = address_mode(word_at(bytes, address_w_offset), Refusal::address_w, desc.AddressW);
        r != Refusal::none)
        return r;

    desc.MinLOD = float_at(bytes, min_lod_offset);
    desc.MaxLOD = float_at(bytes, max_lod_offset);
    if (!std::isfinite(desc.MinLOD)) return Refusal::min_lod;
    if (!std::isfinite(desc.MaxLOD)) return Refusal::max_lod;
    if (desc.MinLOD > desc.MaxLOD) return Refusal::lod_order;

    desc.MipLODBias = float_at(bytes, mip_lod_bias_offset);
    if (!(desc.MipLODBias >= D3D12_MIP_LOD_BIAS_MIN && desc.MipLODBias <= D3D12_MIP_LOD_BIAS_MAX))
        return Refusal::mip_lod_bias;

    desc.MaxAnisotropy = word_at(bytes, max_anisotropy_offset);
    if (desc.MaxAnisotropy < 1 || desc.MaxAnisotropy > D3D12_MAX_MAXANISOTROPY) return Refusal::max_anisotropy;

    const std::uint32_t comparison = word_at(bytes, comparison_offset);
    if (comparison < D3D12_COMPARISON_FUNC_NEVER || comparison > D3D12_COMPARISON_FUNC_ALWAYS)
        return Refusal::comparison;
    // The fixed filter is not a comparison filter, so desktop D3D12 would ignore the function
    // (debug message 1361). The value is validated and not applied.
    desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;

    out = desc;
    return Refusal::none;
}

const char* refusal_name(Refusal refusal) noexcept
{
    switch (refusal) {
    case Refusal::none: return "none";
    case Refusal::null_desc: return "null_desc";
    case Refusal::address_u: return "address_u";
    case Refusal::address_v: return "address_v";
    case Refusal::address_w: return "address_w";
    case Refusal::border_address: return "border_address";
    case Refusal::min_lod: return "min_lod";
    case Refusal::max_lod: return "max_lod";
    case Refusal::lod_order: return "lod_order";
    case Refusal::mip_lod_bias: return "mip_lod_bias";
    case Refusal::max_anisotropy: return "max_anisotropy";
    case Refusal::comparison: return "comparison";
    }
    return "unknown";
}

}  // namespace d3d12x_sampler
