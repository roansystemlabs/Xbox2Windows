#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>

#include <cstddef>
#include <cstdint>

// Desktop policy for the D3D12.X CreateSamplerX descriptor (#874).
//
// The only public layout reference is the XCL ABI notes for device slot #61: a 0x6C-byte
// descriptor whose desktop fields sit at the offsets below
// (https://github.com/ivall/xcl/blob/main/HANDOFF.md). These offsets are fixture policy, not a
// verified console contract. The filter selection and border colour have no documented offset,
// so the translation uses a fixed anisotropic filter and refuses BORDER addressing. The
// comparison function is validated but not applied, because that filter does not compare. Every
// other byte of the descriptor is ignored.
namespace d3d12x_sampler {

inline constexpr std::size_t desc_bytes = 0x6c;

inline constexpr std::size_t address_u_offset = 0x18;
inline constexpr std::size_t address_v_offset = 0x1c;
inline constexpr std::size_t address_w_offset = 0x20;
inline constexpr std::size_t min_lod_offset = 0x24;
inline constexpr std::size_t max_lod_offset = 0x28;
inline constexpr std::size_t mip_lod_bias_offset = 0x2c;
inline constexpr std::size_t max_anisotropy_offset = 0x34;
inline constexpr std::size_t comparison_offset = 0x40;

inline constexpr D3D12_FILTER fixed_filter = D3D12_FILTER_ANISOTROPIC;

enum class Refusal : std::uint8_t {
    none,
    null_desc,
    address_u,
    address_v,
    address_w,
    border_address,
    min_lod,
    max_lod,
    lod_order,
    mip_lod_bias,
    max_anisotropy,
    comparison,
};

// Translates a 0x6C-byte descriptor into a desktop D3D12_SAMPLER_DESC. On refusal `out` is not
// written. `console_desc` is read with unaligned loads.
Refusal to_desktop(const void* console_desc, D3D12_SAMPLER_DESC& out) noexcept;

const char* refusal_name(Refusal refusal) noexcept;

}  // namespace d3d12x_sampler
