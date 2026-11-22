"""Check the single-store compute profile with original synthetic kernels."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

from pixel_aot_check import envelope, must_reject
import compute_store_aot as aot
import shader_profile_audit
from pixel_aot import _read_named_container, disassemble
from resource_inventory import parts

SENTINEL = 0xc5c5c5c5
INDEXED = [
    "s_mov_b32 s12, s7", "s_mov_b32 s13, 0",
    "s_load_dwordx4 s[20:23], s[12:13], 0x30",
    "v_lshl_add_u32 v8, s11, 5, v0",
    "v_mov_b32_e32 v15, 19", "v_mov_b32_e32 v16, -5",
    "s_waitcnt lgkmcnt(0)", "buffer_store_dwordx2 v[15:16], v8, s[20:23], 0 idxen offset:4",
    "s_endpgm",
]
FIXED = [
    "s_mov_b32 s9, 0", "s_load_dwordx4 s[12:15], s[8:9], null",
    "v_mov_b32_e32 v20, 11", "v_mov_b32_e32 v21, 23", "v_mov_b32_e32 v22, -2",
    "s_waitcnt lgkmcnt(0)", "buffer_store_dwordx3 v[20:22], off, s[12:15], 0 offset:20",
    "s_endpgm",
]
MULTI_FIXED = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "s_mov_b32 s10, 0x12345678",
    "s_movk_i32 s11, 0xfffc",
    "v_mov_b32_e32 v4, s10",
    "v_mov_b32_e32 v5, s11",
    "v_mov_b32_e32 v6, v4",
    "v_mov_b32_e32 v7, 7",
    "s_waitcnt lgkmcnt(0)",
    "s_clause 0x1",
    "buffer_store_dwordx2 v[4:5], off, s[12:15], 0 offset:8",
    "buffer_store_dwordx2 v[6:7], off, s[12:15], 0 offset:32",
    "s_endpgm",
]
MULTI_INDEXED = [
    "s_mov_b32 s12, s7",
    "s_mov_b32 s13, 0",
    "s_load_dwordx4 s[20:23], s[12:13], 0x30",
    "s_mov_b32 s14, 0x00abcdef",
    "s_movk_i32 s15, 0xfff0",
    "v_lshl_add_u32 v8, s11, 5, v0",
    "v_mov_b32_e32 v10, s14",
    "v_mov_b32_e32 v11, s15",
    "s_waitcnt lgkmcnt(0) vmcnt(0)",
    "buffer_store_dword v10, v8, s[20:23], 0 idxen",
    "buffer_store_dword v11, v8, s[20:23], 0 idxen offset:8",
    "s_endpgm",
]
RUNTIME = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "s_load_dwordx4 s[16:19], s[8:9], 0x20",
    "s_waitcnt lgkmcnt(0)",
    "s_buffer_load_dword s20, s[16:19], null",
    "s_buffer_load_dwordx2 s[22:23], s[16:19], 0xc",
    "s_buffer_load_dwordx4 s[24:27], s[16:19], 0x30",
    "s_waitcnt lgkmcnt(0)",
    "s_mov_b32 s28, s20",
    "v_mov_b32_e32 v10, s28", "v_mov_b32_e32 v11, s22", "v_mov_b32_e32 v12, s23",
    "v_mov_b32_e32 v13, s27", "v_mov_b32_e32 v14, s26", "v_mov_b32_e32 v15, s25",
    "v_mov_b32_e32 v16, s24", "v_mov_b32_e32 v17, v16",
    "buffer_store_dwordx3 v[10:12], off, s[12:15], 0 offset:16",
    "buffer_store_dwordx4 v[13:16], off, s[12:15], 0 offset:32",
    "buffer_store_dword v17, off, s[12:15], 0 offset:64",
    "s_endpgm",
]
# 8x8 tile: row = dispatch Y through idxen, column = dispatch X through offen (8-byte
# elements); each invocation stores (tid.x, tid.y).
TILE = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "v_lshl_add_u32 v3, s10, 3, v0",
    "v_lshl_add_u32 v4, s11, 3, v1",
    "v_lshlrev_b32_e32 v5, 3, v3",
    "v_mov_b32_e32 v6, v3",
    "v_mov_b32_e32 v7, v4",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dwordx2 v[6:7], v[4:5], s[12:15], 0 idxen offen",
    "s_endpgm",
]
# 4x1x8 slab: row = dispatch Z, column = dispatch X (16-byte elements); stores (tid.z, tid.x)
# after a gap word.
SLAB = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "v_lshl_add_u32 v10, s16, 3, v2",
    "v_lshl_add_u32 v3, s10, 2, v0",
    "v_lshlrev_b32_e32 v11, 4, v3",
    "v_mov_b32_e32 v20, v10",
    "v_mov_b32_e32 v21, v3",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dwordx2 v[20:21], v[10:11], s[12:15], 0 idxen offen offset:4",
    "s_endpgm",
]
# 4x4 tile with several groups in X and Y: one word per invocation, the dispatch X index.
QUAD = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "v_lshl_add_u32 v4, s11, 2, v1",
    "v_lshl_add_u32 v3, s10, 2, v0",
    "v_lshlrev_b32_e32 v5, 2, v3",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dword v3, v[4:5], s[12:15], 0 idxen offen",
    "s_endpgm",
]
# One 256-thread group: one word per invocation, the dispatch X index.
WIDE = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "v_lshl_add_u32 v3, s10, 8, v0",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dword v3, v3, s[12:15], 0 idxen",
    "s_endpgm",
]
# 4x4x4 cube with all three axes in the address: row = dispatch Z through idxen, byte
# offset = (dispatch Y << 4) + (dispatch X << 2); each invocation stores tid.z. The address
# instructions are llc-15 -mcpu=gfx1013 output for an amdgpu_kernel; the kernel-argument load
# is replaced by the descriptor-table load the other fixtures use.
CUBE = [
    "s_mov_b32 s1, 0",
    "s_load_dwordx4 s[8:11], s[0:1], null",
    "v_lshl_add_u32 v1, s3, 2, v1",
    "s_lshl_b32 s0, s2, 2",
    "v_add_lshl_u32 v3, s0, v0, 2",
    "v_lshl_add_u32 v0, s4, 2, v2",
    "v_lshl_add_u32 v1, v1, 4, v3",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dword v0, v[0:1], s[8:11], 0 idxen offen",
    "s_endpgm",
]
INDEX_BINDING = dict(table_reg=7, group_reg=11, descriptor_offset=48, stride=12)
FIXED_BINDING = dict(table_reg=8, descriptor_offset=0)
RUNTIME_BINDING = dict(FIXED_BINDING, constant_descriptor=32)
TILE_BINDING = dict(FIXED_BINDING, group_reg=10, group_reg_y=11, stride=72)
SLAB_BINDING = dict(FIXED_BINDING, group_reg=10, group_reg_z=16, stride=68)
QUAD_BINDING = dict(FIXED_BINDING, group_reg=10, group_reg_y=11, stride=52)
WIDE_BINDING = dict(FIXED_BINDING, group_reg=10, stride=4)
CUBE_BINDING = dict(table_reg=0, descriptor_offset=0, group_reg=2, group_reg_y=3, group_reg_z=4, stride=136)
CASES = [
    ("indexed", INDEXED, 32, INDEX_BINDING),
    ("fixed", FIXED, 1, FIXED_BINDING),
    ("multi_fixed", MULTI_FIXED, 1, FIXED_BINDING),
    ("multi_indexed", MULTI_INDEXED, 32, INDEX_BINDING),
    ("runtime", RUNTIME, 1, RUNTIME_BINDING),
    ("tile", TILE, (8, 8, 1), TILE_BINDING),
    ("slab", SLAB, (4, 1, 8), SLAB_BINDING),
    ("quad", QUAD, (4, 4, 1), QUAD_BINDING),
    ("wide", WIDE, (256, 1, 1), WIDE_BINDING),
    ("cube", CUBE, (4, 4, 4), CUBE_BINDING),
]


def grid_words(name, groups):
    """Independent expected readback for the multi-dimensional cases."""
    dims, words = dict((case[0], case[2]) for case in CASES)[name], [SENTINEL] * 320
    gx, gy, gz = groups
    for x in range(gx * dims[0]):
        for y in range(gy * dims[1]):
            for z in range(gz * dims[2]):
                if name == "tile":
                    words[(72 * y + 8 * x) // 4:(72 * y + 8 * x) // 4 + 2] = [x, y]
                elif name == "slab":
                    words[(68 * z + 16 * x + 4) // 4:(68 * z + 16 * x + 4) // 4 + 2] = [z, x]
                elif name == "quad":
                    words[(52 * y + 4 * x) // 4] = x
                elif name == "cube":
                    words[(136 * z + 16 * y + 4 * x) // 4] = z
                else:
                    words[x] = x
    return words


GRID_DISPATCHES = {
    "tile": ((1, 1, 1), (1, 2, 1)),
    "slab": ((1, 1, 1), (1, 1, 2)),
    "quad": ((1, 1, 1), (3, 2, 1)),
    "wide": ((1, 1, 1),),
    # Several groups in Z, then in Y and Z. Dispatch X stays one group: its column span must
    # fit below the Y shift, and the Z rows (8 x 136 bytes) must fit the 320-word buffer.
    "cube": ((1, 1, 1), (1, 1, 2), (1, 2, 2)),
}
# Recompiled mutations run on the largest dispatch. Swapping two axes that span the same range
# leaves the buffer unchanged: Y and Z both span 8 in the 1x2x2 cube, so that swap runs on 1x1x2,
# where Y spans 4 and Z spans 8.
MUTATION_DISPATCH = {("cube", "swapped Y/Z"): (1, 1, 2)}


def controls():
    for change in (dict(table_reg=6), dict(group_reg=10), dict(descriptor_offset=32), dict(stride=8)):
        must_reject(lambda: aot.lower(INDEXED, 32, **(INDEX_BINDING | change)))
    for old, new in (("lgkmcnt(0)", "lgkmcnt(1)"), ("s11, 5", "s11, 4"),
                     ("s[20:23], 0 idxen", "s[24:27], 0 idxen"), ("s13, 0", "s13, 1")):
        must_reject(lambda: aot.lower([s.replace(old, new) for s in INDEXED], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(INDEXED[:6] + INDEXED[7:], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(INDEXED[:7] + ["s_mov_b32 s21, 0"] + INDEXED[7:], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(FIXED, 32, **FIXED_BINDING))
    must_reject(lambda: aot.lower(["s_endpgm"], 1, **FIXED_BINDING))

    # Read a constant word directly into v10 before the constant-load wait.
    must_reject(lambda: aot.lower(
        RUNTIME[:7] + ["v_mov_b32_e32 v10, s20", RUNTIME[7]] + RUNTIME[10:],
        1, **RUNTIME_BINDING))
    for old, new in (("s11, 5, v0", "s11, 5, v1"),
                     ("v[15:16], v8,", "v[15:16], v0,"),
                     ("0 idxen offset:4", "0 offset:4")):
        must_reject(lambda: aot.lower([s.replace(old, new) for s in INDEXED], 32, **INDEX_BINDING))
    must_reject(lambda: aot.lower(
        RUNTIME[:4] + ["s_mov_b32 s17, 0"] + RUNTIME[4:], 1, **RUNTIME_BINDING))
    must_reject(lambda: aot.lower(
        INDEXED[:7] + ["s_mov_b32 s23, 0"] + INDEXED[7:], 32, **INDEX_BINDING))
    movk_boundary = FIXED[:2] + ["s_movk_i32 s10, 0x8000", "v_mov_b32_e32 v20, s10"] + FIXED[3:]
    assert "    outputData.Store3(20u, uint3(0xffff8000u, 0x00000017u, 0xfffffffeu));" in (
        aot.lower(movk_boundary, 1, **FIXED_BINDING).splitlines())
    must_reject(lambda: aot.lower(FIXED, 1, **dict(FIXED_BINDING, constant_descriptor=8)))

    # Overlapping store spans in one invocation are rejected.
    must_reject(lambda: aot.lower(
        MULTI_FIXED[:-1] + ["buffer_store_dword v7, off, s[12:15], 0 offset:12", "s_endpgm"],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        MULTI_INDEXED[:-1] + ["buffer_store_dwordx2 v[10:11], v8, s[20:23], 0 idxen offset:4", "s_endpgm"],
        32, **INDEX_BINDING))
    # Non-constant or pending scalar/vector sources and out-of-range hints/immediates are rejected.
    must_reject(lambda: aot.lower(
        [s.replace("v_mov_b32_e32 v4, s10", "v_mov_b32_e32 v4, s8") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("v_mov_b32_e32 v4, s10", "v_mov_b32_e32 v4, s12") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("v_mov_b32_e32 v6, v4", "v_mov_b32_e32 v6, v0") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("s_clause 0x1", "s_clause 0x40") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(lambda: aot.lower(
        [s.replace("s_movk_i32 s11, 0xfffc", "s_movk_i32 s11, 0x10000") for s in MULTI_FIXED],
        1, **FIXED_BINDING))
    must_reject(aot.group_size, bytes(52))
    for descriptor in (None, 8, 16, 33):
        must_reject(lambda: aot.lower(RUNTIME, 1, **dict(FIXED_BINDING, constant_descriptor=descriptor)))
    for old, new in (
        ("s[16:19], 0x30", "s[16:19], 0x34"),
        ("s[16:19], 0xc", "s[16:19], 0xa"),
        ("s[16:19], 0xc", "s[12:15], 0xc"),
        ("s[22:23]", "s[21:22]"),
        ("s[24:27]", "s[20:23]"),
        ("s[12:15], 0 offset:64", "s[16:19], 0 offset:64"),
    ):
        must_reject(lambda: aot.lower([line.replace(old, new) for line in RUNTIME], 1, **RUNTIME_BINDING))
    for slot in (3, 7):
        early = RUNTIME.copy()
        early[slot] = "s_waitcnt vmcnt(0)"
        must_reject(lambda: aot.lower(early, 1, **RUNTIME_BINDING))
    grid_controls()


def grid_controls():
    """Multi-dimensional groups: exact output, then one rejection per broken assumption."""
    tile = aot.lower(TILE, (8, 8, 1), **TILE_BINDING).splitlines()
    assert "[numthreads(8,8,1)]" in tile
    assert "    outputData.Store2(tid.y * 72u + (tid.x << 3u) + 0u, uint2(tid.x, tid.y));" in tile
    assert "    outputData.Store2(tid.z * 68u + (tid.x << 4u) + 4u, uint2(tid.z, tid.x));" in (
        aot.lower(SLAB, (4, 1, 8), **SLAB_BINDING).splitlines())
    assert "    outputData.Store(tid.x * 4u + 0u, tid.x);" in aot.lower(WIDE, 256, **WIDE_BINDING).splitlines()
    # Group sizes: per-axis and total D3D12 limits; the former 1D widths still pass.
    for group in ((1024, 1, 1), (1, 1024, 1), (1, 1, 64), (16, 16, 4), 128, 32, 64, 1):
        aot.checked_group(group)
    for group in ((1025, 1, 1), (1, 1, 65), (64, 4, 8), (0, 1, 1), (8, 0, 1), 0, (8.0, 8, 1)):
        must_reject(aot.checked_group, group)
    # Wrong group size: the shifts no longer match, or the size is not a power of two.
    for group in ((8, 4, 1), (4, 8, 1), (8, 6, 1), (8, 12, 1), (12, 8, 1), (8, 8, 2), (8, 1, 1)):
        must_reject(lambda: aot.lower(TILE, group, **TILE_BINDING))
    # A group whose Y is one launches without a Y local ID; v1 is then undefined.
    must_reject(lambda: aot.lower([s.replace("s11, 2, v1", "s11, 0, v1") for s in QUAD], (4, 1, 1),
                                  **QUAD_BINDING))
    # Axes swapped or ignored: local X with group Y, local Z read in a group whose Z is one.
    for old, new in (("s11, 3, v1", "s11, 3, v0"), ("s10, 3, v0", "s10, 3, v1"),
                     ("s10, 3, v0", "s11, 3, v0")):
        must_reject(lambda: aot.lower([s.replace(old, new) for s in TILE], (8, 8, 1), **TILE_BINDING))
    must_reject(lambda: aot.lower(SLAB, (4, 8, 1), **SLAB_BINDING))
    must_reject(lambda: aot.lower([s.replace("3, v2", "3, v1") for s in SLAB], (4, 1, 8), **SLAB_BINDING))
    # Wrong or missing group-register bindings.
    for change in (dict(group_reg_y=12), dict(group_reg_y=None), dict(group_reg=11), dict(group_reg_y=8),
                   dict(group_reg_z=11)):
        must_reject(lambda: aot.lower(TILE, (8, 8, 1), **(TILE_BINDING | change)))
    must_reject(lambda: aot.lower(SLAB, (4, 1, 8), **(SLAB_BINDING | dict(group_reg_z=None, group_reg_y=16))))
    must_reject(lambda: aot.lower(SLAB, (4, 1, 8), **(SLAB_BINDING | dict(group_reg_y=16))))
    # Index and byte offset from the same axis would alias rows.
    same_axis = WIDE[:3] + ["v_lshlrev_b32_e32 v4, 2, v3", WIDE[3],
                            "buffer_store_dword v3, v[3:4], s[12:15], 0 idxen offen", "s_endpgm"]
    must_reject(lambda: aot.lower(same_axis, 256, **(WIDE_BINDING | dict(stride=1024))))
    # Addresses that leave an element or a row, or that two invocations of one group share.
    for binding in (dict(stride=60), dict(stride=None)):
        must_reject(lambda: aot.lower(TILE, (8, 8, 1), **(TILE_BINDING | binding)))
    for old, new in (("v_lshlrev_b32_e32 v5, 3, v3", "v_lshlrev_b32_e32 v5, 2, v3"),
                     ("v_lshlrev_b32_e32 v5, 3, v3", "v_lshlrev_b32_e32 v5, 1, v3"),
                     ("v_lshlrev_b32_e32 v5, 3, v3", "v_lshlrev_b32_e32 v5, 3, v4"),
                     ("v_lshlrev_b32_e32 v5, 3, v3", "v_lshlrev_b32_e32 v5, 3, v0"),
                     ("v[4:5], s[12:15], 0 idxen offen", "v4, s[12:15], 0 idxen"),
                     ("v[4:5], s[12:15], 0 idxen offen", "v[4:5], s[12:15], 0 offen"),
                     ("v[4:5], s[12:15], 0 idxen offen", "v[3:4], s[12:15], 0 idxen offen"),
                     ("0 idxen offen", "0 idxen offen offset:4")):
        must_reject(lambda: aot.lower([s.replace(old, new) for s in TILE], (8, 8, 1), **TILE_BINDING))
    must_reject(lambda: aot.lower(WIDE, (256, 2, 1), **WIDE_BINDING))
    must_reject(lambda: aot.lower(QUAD[:-1] + ["buffer_store_dword v3, v4, s[12:15], 0 idxen offset:4", "s_endpgm"],
                                  (4, 4, 1), **QUAD_BINDING))
    # Two stores that are valid alone but transpose the layout race across invocations.
    transposed = ["v_mov_b32_e32 v6, v3", "v_lshlrev_b32_e32 v7, 3, v4",
                  "buffer_store_dword v3, v[6:7], s[12:15], 0 idxen offen offset:4"]
    aot.lower(QUAD[:5] + transposed[:2] + QUAD[5:6] + transposed[2:] + ["s_endpgm"], (4, 4, 1), **QUAD_BINDING)
    must_reject(lambda: aot.lower(QUAD[:5] + transposed[:2] + QUAD[5:-1] + transposed[2:] + ["s_endpgm"],
                                  (4, 4, 1), **QUAD_BINDING))
    # A store with an unresolved local ID as data is still rejected.
    must_reject(lambda: aot.lower([s.replace("v_mov_b32_e32 v6, v3", "v_mov_b32_e32 v6, v1") for s in TILE],
                                  (8, 8, 1), **TILE_BINDING))
    cube_controls()
    audit_candidates()


def cube_controls():
    """Three address axes: exact output, then one rejection per broken assumption."""
    def cube(*changes, group=(4, 4, 4), **binding):
        lines = CUBE
        for old, new in changes:
            assert sum(old in line for line in lines) == 1, old
            lines = [line.replace(old, new) for line in lines]
        return aot.lower(lines, group, **(CUBE_BINDING | binding))

    source = cube().splitlines()
    assert "[numthreads(4,4,4)]" in source
    assert "    outputData.Store(tid.z * 136u + (tid.y << 4u) + (tid.x << 2u) + 0u, tid.z);" in source
    # Other group sizes: the shifts no longer match, or local Z is not launched.
    for group in ((8, 4, 4), (4, 8, 4), (4, 4, 8), (4, 4, 2), (2, 4, 4), (4, 4, 1), (4, 1, 4), (4, 4, 3)):
        must_reject(lambda: cube(group=group))
    # A group X of 6 keeps the base shift of 2; with Y moved to bit 5 its span fits, so only the
    # power-of-two rule of the group base rejects it.
    must_reject(lambda: cube(("v1, v1, 4, v3", "v1, v1, 5, v3"), group=(6, 4, 4)))
    # Group base: wrong shift, a register that is not a group ID, a pending destination.
    for change in (("s0, s2, 2", "s0, s2, 3"), ("s0, s2, 2", "s0, s5, 2"), ("s0, s2, 2", "s0, s3, 2"),
                   ("s_lshl_b32 s0,", "s_lshl_b32 s9,")):
        must_reject(lambda: cube(change))
    # Group base plus the local ID of another axis, or a base that was never scaled.
    for change in (("s0, v0, 2", "s0, v2, 2"), ("v_add_lshl_u32 v3, s0,", "v_add_lshl_u32 v3, s2,")):
        must_reject(lambda: cube(change))
    # Byte offset terms: the same axis twice (in a 4x1x4 group, where X and Z still cover every
    # invocation), a term that is not a dispatch index, and nesting that lets one group's X span
    # reach the Y step, or one element reach the next X.
    must_reject(lambda: cube(("v1, s3, 2, v1", "v1, s2, 2, v0"), group=(4, 1, 4), group_reg_y=None))
    # The term control keeps local Y in v1, so v1 is not a dispatch index when it is shifted.
    for changes in ((("v_lshl_add_u32 v1, s3, 2, v1", "v_lshl_add_u32 v4, s3, 2, v1"),),
                    (("v1, v1, 4, v3", "v1, v1, 3, v3"),), (("s0, v0, 2", "s0, v0, 1"),),
                    (("0 idxen offen", "0 idxen offen offset:4"),)):
        must_reject(lambda: cube(*changes))
    # The rows must hold one group's Y span; the index axis must not also be an offset term.
    for binding in (dict(stride=60), dict(stride=None)):
        must_reject(lambda: cube(**binding))
    must_reject(lambda: cube(("v_lshl_add_u32 v0, s4, 2, v2", "v_mov_b32_e32 v0, v1"), group=(4, 4, 1),
                             group_reg_z=None))
    # Group registers: missing Z, Z equal to Y, or the table register as a group ID.
    for binding in (dict(group_reg_z=None), dict(group_reg_z=3), dict(group_reg_z=5), dict(table_reg=2)):
        must_reject(lambda: cube(**binding))
    # Store address with only the X offset (Y not separated) or with an unscaled offset.
    must_reject(lambda: cube(("v_lshl_add_u32 v1, v1, 4, v3", "v_mov_b32_e32 v1, v3")))
    must_reject(lambda: cube(("v_lshl_add_u32 v1, v1, 4, v3", "v_lshl_add_u32 v1, v1, 4, v1")))


def audit_candidates():
    """The audit tries group Z in the SGPR after group X when the group's Y is one."""
    slab = ["s_mov_b32 s4, s5", "s_mov_b32 s5, 0", "s_load_dwordx4 s[12:15], s[4:5], null",
            "v_lshl_add_u32 v3, s6, 2, v0", "v_lshl_add_u32 v4, s7, 3, v2", "v_lshlrev_b32_e32 v5, 2, v3",
            "s_waitcnt lgkmcnt(0)", "buffer_store_dword v4, v[4:5], s[12:15], 0 idxen offen", "s_endpgm"]
    for lines, expected in ((slab, "    outputData.Store(tid.z * 16u + (tid.x << 2u) + 0u, tid.z);"),
                            ([s.replace("s7, 3", "s8, 3") for s in slab], None)):
        emitted = []
        with patch.object(shader_profile_audit.pixel_aot, "disassemble", return_value=lines):
            profiles = shader_profile_audit.compute_profiles(b"", (4, 1, 8), None,
                                                             lambda kind, source: emitted.append(source))
        if expected is None:
            assert profiles == [] and not emitted
        else:
            assert profiles == ["store"] and expected in emitted[0].splitlines()


def binary(directory, assembler, objcopy, library):
    for _, lines, width, binding in CASES:
        src, obj, raw = [directory / x for x in ("test.s", "test.o", "test.bin")]
        src.write_text("\n".join(lines) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", src, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        code = raw.read_bytes()
        padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
        dims = aot.checked_group(width)
        payload = envelope(padded, directory_end=(40 + len(padded), 288 + len(padded)), compute_width=dims[0])
        blob = bytearray(_read_named_container(payload, 1 + len(b"fixture") + 1))
        # The shared envelope writes X only; set Y and Z in the PSV0 runtime info.
        struct.pack_into("<2I", blob, blob.index(b"PSV0") + 8 + 44, *dims[1:])
        blob = bytes(blob)
        sections = parts(blob)
        assert aot.group_size(sections[b"PSV0"]) == dims
        assert aot.container_code(sections[b"DXIL"], stage=5) == code
        must_reject(lambda: aot.container_code(sections[b"DXIL"], stage=0))
        assert aot.lower(disassemble(code, library), width, **binding) == aot.lower(lines, width, **binding)
        container, output = directory / "input.dxbc", directory / "cli.hlsl"
        container.write_bytes(blob)
        options = [item for key, value in binding.items() for item in ("--" + key.replace("_", "-"), str(value))]
        subprocess.run([sys.executable, "-B", str(Path(aot.__file__)), container, "--output", output,
                        "--llvm", library, *options], check=True)
        assert output.read_text() == aot.lower(lines, width, **binding)


def draw_runtime(cmd, case):
    first = [0x80000000 + i * 0x101 for i in range(16)]
    second = [0x7fc00000 + i * 0x181 for i in range(16)]
    refs = []
    for values in (first, second):
        words = [SENTINEL] * 320
        words[4:7] = [values[0], values[3], values[4]]
        words[8:12] = values[12:16][::-1]
        words[16] = values[12]
        refs.append(words)
        case.write_bytes(struct.pack("<337I", 1, *words, *values))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    case.write_bytes(struct.pack("<337I", 1, *refs[0], *second))
    stale = subprocess.run(cmd, capture_output=True, text=True)
    if (stale.returncode != 1 or "word=4 expected=80000000 actual=7fc00000" not in stale.stderr
            or "groups=1 compared_words=320 mismatches=8" not in stale.stdout.splitlines()):
        raise AssertionError(f"Expected a stale runtime constant mismatch\n{stale.stdout}\n{stale.stderr}")
    case.write_bytes(struct.pack("<337I", 1, *refs[1], *second)[:-4])
    truncated = subprocess.run(cmd, capture_output=True, text=True)
    if truncated.returncode != 1 or "Invalid compute store constants" not in truncated.stderr:
        raise AssertionError(f"Expected a truncated constants error\n{truncated.stdout}\n{truncated.stderr}")
    return 0


def grid_mutations(source, name):
    """Translations that a dispatch-dimension mistake would produce; each must fail on the GPU."""
    a, b = {"slab": ("tid.z", "tid.x")}.get(name, ("tid.x", "tid.y"))
    if name != "wide":
        yield "swapped axes", source.replace(a, "tid.T").replace(b, a).replace("tid.T", b)
    if name in ("slab", "cube"):
        yield "Z ignored", source.replace("tid.z", "tid.y")
    dims = dict((case[0], case[2]) for case in CASES)[name]
    numthreads = "[numthreads({},{},{})]".format(*dims)
    yield "halved group X", source.replace(numthreads, "[numthreads({},{},{})]".format(dims[0] // 2, *dims[1:]))
    if name == "cube":
        for a, b in (("tid.y", "tid.z"), ("tid.x", "tid.z")):
            yield f"swapped {a[-1].upper()}/{b[-1].upper()}", (
                source.replace(a, "tid.T").replace(b, a).replace("tid.T", b))
        yield "Y/Z swapped in the address only", source.replace(
            "tid.z * 136u + (tid.y << 4u)", "tid.y * 136u + (tid.z << 4u)")
        yield "halved group Z", source.replace(numthreads, "[numthreads(4,4,2)]")


def draw_grid(directory, compiler, cmd, case, name):
    dispatches = GRID_DISPATCHES[name]
    for groups in dispatches:
        words = grid_words(name, groups)
        packed = groups[0] | groups[1] << 8 | groups[2] << 16
        case.write_bytes(struct.pack("<321I", packed, *words))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    guard = words.index(SENTINEL)
    words[guard] ^= 1
    case.write_bytes(struct.pack("<321I", packed, *words))
    bad = subprocess.run(cmd, capture_output=True, text=True)
    if bad.returncode != 1 or f"word={guard} expected=" not in bad.stderr:
        raise AssertionError(f"{name}: a wrong untouched word escaped detection")
    words[guard] ^= 1
    if len(dispatches) > 1:
        first = dispatches[0]
        case.write_bytes(struct.pack("<321I", first[0] | first[1] << 8 | first[2] << 16, *words))
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{name}: a wrong group count escaped detection")
    source = (directory / "store.hlsl").read_text(encoding="ascii")
    for label, mutated in grid_mutations(source, name):
        if mutated == source:
            raise AssertionError(f"{name}: mutation '{label}' did not change the source")
        src, cso = directory / "mutated.hlsl", directory / "mutated.cso"
        src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        groups = MUTATION_DISPATCH.get((name, label), dispatches[-1])
        assert groups in dispatches, (name, label)
        case.write_bytes(struct.pack("<321I", groups[0] | groups[1] << 8 | groups[2] << 16,
                                     *grid_words(name, groups)))
        bad = subprocess.run([*cmd[:3], cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{name}: mutation '{label}' escaped detection with groups={groups}")
    return 0


def draw(directory, compiler, renderer, backend):
    for name, lines, width, binding in CASES:
        src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
        src.write_text(aot.lower(lines, width, **binding), encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        cmd = [renderer, backend, "--compute-store-case", cso, case]
        if name == "runtime" or name in GRID_DISPATCHES:
            result = draw_runtime(cmd, case) if name == "runtime" else draw_grid(directory, compiler, cmd, case, name)
            if result:
                return result
            continue
        for groups in ((1, 3) if width == 32 else (1,)):
            words = [SENTINEL] * 320
            if name == "fixed":
                words[5:8] = [11, 23, 0xfffffffe]
                guard_word, stored_word = 0, 5
            elif name == "indexed":
                for i in range(groups * 32):
                    words[3*i+1:3*i+3] = [19, 0xfffffffb]
                guard_word, stored_word = 0, 1
            elif name == "multi_fixed":
                words[2:4] = [0x12345678, 0xfffffffc]
                words[8:10] = [0x12345678, 7]
                guard_word, stored_word = 4, 9
            else:
                for i in range(groups * 32):
                    words[3*i] = 0x00abcdef
                    words[3*i+2] = 0xfffffff0
                guard_word, stored_word = 1, 2
            case.write_bytes(struct.pack("<321I", groups, *words))
            result = subprocess.run(cmd)
            if result.returncode:
                return result.returncode
        words[guard_word] ^= 1
        case.write_bytes(struct.pack("<321I", groups, *words))
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={guard_word} expected=" not in bad.stderr:
            raise AssertionError("A wrong untouched word escaped detection")
        words[guard_word] ^= 1
        words[stored_word] ^= 1
        case.write_bytes(struct.pack("<321I", groups, *words))
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError("A wrong stored word escaped detection")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit({"binary": binary, "draw": draw}[mode](Path(tmp), *args))
