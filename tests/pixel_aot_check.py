"""Exercise offline pixel translation with independently assembled test programs."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pixel_aot
from pixel_aot import ALU, disassemble, native_program

# First word of each line of every program lower() accepted in this run. tools/form_coverage.py counts
# these when this test passes; a program a check expects to be rejected raises before it is recorded.
ACCEPTED = set()
SOPC_OPS = (
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
    "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    "s_cmp_eq_u64", "s_cmp_lg_u64",
    "s_bitcmp0_b32", "s_bitcmp1_b32",
    "s_bitcmp0_b64", "s_bitcmp1_b64",
)
SOP2_LSHL_ADD_OPS = (
    "s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32",
)
VOP2_INT_OPS = (
    "v_lshrrev_b32_e32", "v_ashrrev_i32_e32", "v_lshlrev_b32_e32",
    "v_and_b32_e32", "v_or_b32_e32", "v_xor_b32_e32",
    "v_add_nc_u32_e32", "v_sub_nc_u32_e32", "v_subrev_nc_u32_e32",
)
# The 32-bit no-carry forms of nc_integer_group(1..3), as ACCEPTED records them (first word of the line).
NC_INT_OPS = (
    "v_add_nc_u32_e32", "v_sub_nc_u32_e32", "v_subrev_nc_u32_e32",
    "v_add_nc_u32_e64", "v_sub_nc_u32_e64", "v_subrev_nc_u32_e64",
    "v_add_nc_i32", "v_sub_nc_i32",
)


def lower(instructions, *args, **kwargs):
    source = pixel_aot.lower(instructions, *args, **kwargs)
    ACCEPTED.update(line.split()[0] for line in instructions if line.strip())
    return source


def write_record(path):
    """Save ACCEPTED where CTest's FORM_COVERAGE_RECORD points; nothing is written without it."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(f"{word}\n" for word in sorted(ACCEPTED)), encoding="utf-8")


RAMP = (0, 16, 37, 64, 101, 153, 219, 254)

# CPU references use plain Python floats, which keep binary32 subnormals: the fixtures assume the
# compiled desktop shader preserves them, and DXC runs with a matching -denorm flag. The Scarlett
# pipeline's FLOAT_MODE is unmeasured, so this records the harness's assumption rather than a claim
# about the console; binary16 and binary64 denormal behavior is established separately, not here.
F32_DENORM_MODE = "preserve"
# Binary16 subnormals are preserved by construction, whatever -denorm says. Packed f16 operations
# widen each half from its encoding (widen16), compute in binary32 and narrow in integer arithmetic
# (pack16_rne). Every binary16 subnormal and every product or sum of two of them is a binary32
# normal, so a binary32 flush never reaches them, and DXIL records no f16 denormal mode to request.
# The Python references keep binary16 subnormals too. Like the f32 mode, this is the harness's
# recorded assumption, not a claim about the Scarlett FLOAT_MODE f16 bits.
F16_DENORM_MODE = "preserve"

# Integer-to-binary32 references convert through Python's binary64 float, which holds every 32-bit
# integer exactly, then pack to binary32, rounding ties to even. The Scarlett pipeline's MODE.FP_ROUND
# is unmeasured, so this records the harness's assumption the same way F32_DENORM_MODE records its
# denormal assumption: an adapter disagreeing on a row would be recorded as a mismatch, not chosen.
F32_ROUND_ASSUMPTION = "nearest-even"

# Binary32-to-word references follow the issue #360 reading of the RDNA 2 ISA: truncate toward zero, NaN
# gives 0, out-of-range values saturate. That reading has not been measured on the console, and
# MODE.FP_ROUND is assumed not to affect truncation.
F32_TRUNC_ASSUMPTION = "toward-zero, NaN->0, saturating"


def u32f(word):
    """Reference for v_cvt_f32_u32_e32: read the word as unsigned and round to binary32."""
    return struct.unpack("<f", struct.pack("<f", float(word)))[0]


def i32f(word):
    """Reference for v_cvt_f32_i32_e32: sign-extend the word, then convert as u32f does."""
    return u32f(word - 2**32 if word & 0x80000000 else word)


def assembly(swap):
    red, blue = ("v14", "v12") if swap else ("v12", "v14")
    return [
        "s_mov_b64 s[50:51], exec",
        "s_wqm_b64 exec, exec",
        "s_mov_b32 m0, s24",
        "v_interp_p1_f32_e32 v8, v0, attr0.x",
        "v_interp_p1_f32_e32 v9, v0, attr0.y",
        "v_interp_p2_f32_e32 v9, v1, attr0.y",
        "v_interp_p2_f32_e32 v8, v1, attr0.x",
        "s_and_b64 exec, exec, s[50:51]",
        "s_mov_b32 s21, 0",
        "s_load_dwordx8 s[28:35], s[20:21], null",
        "s_mov_b32 s40, 1",
        "s_mov_b32 s41, 2",
        "s_mov_b32 s42, 3",
        "s_mov_b32 s43, 4",
        "s_waitcnt lgkmcnt(0)",
        "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        f"v_cvt_pkrtz_f16_f32_e32 v16, {red}, v13",
        f"v_cvt_pkrtz_f16_f32_e32 v17, {blue}, v15",
        "exp mrt0 v16, v16, v17, v17 done compr vm",
        "s_endpgm",
    ]


def rgb_sample() -> list[str]:
    program = assembly(False)
    sample = next(i for i, line in enumerate(program) if line.startswith("image_sample"))
    program[sample] = program[sample].replace("v[12:15]", "v[12:14]").replace("dmask:0xf", "dmask:0x7")
    program.insert(sample, "v_mov_b32_e32 v15, 0.5")
    return program


def lod_zero_sample() -> list[str]:
    mask_ops = ("s_mov_b64", "s_wqm_b64", "s_and_b64")
    return [line.replace("image_sample ", "image_sample_lz ") for line in assembly(False)
            if not line.startswith(mask_ops)]


DUAL_CASES = ("forward", "reverse", "reuse", "partial")


def dual_program(case="forward", *, swap_weights=False):
    program = assembly(False)
    first = next(i for i, line in enumerate(program) if line.startswith("image_sample"))
    sample_a = program[first]
    sample_b = sample_a.replace("v[12:15]", "v[20:23]").replace("s[28:35]", "s[56:63]")
    load_b = "s_load_dwordx8 s[56:63], s[20:21], 0x20"
    if case == "reuse":
        # Finish A before reloading the same descriptor registers for B.
        program[first:first+1] = [sample_a, "s_waitcnt vmcnt(0)",
                                  load_b.replace("56:63", "28:35"), "s_waitcnt lgkmcnt(0)",
                                  sample_b.replace("56:63", "28:35")]
    elif case in ("forward", "reverse", "partial"):
        program[first:first+1] = [sample_a, sample_b] if case != "reverse" else [sample_b, sample_a]
        program.insert(program.index("s_mov_b32 s40, 1"), load_b)
    else:
        raise ValueError(f"Unknown dual-texture case: {case}")
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    a, b = ("0x3e800000", "0x3f400000") if swap_weights else ("0x3f400000", "0x3e800000")
    ops = []
    for i in range(4):
        ops.extend([f"v_mul_legacy_f32_e32 v{12+i}, {a}, v{12+i}",
                    f"v_fma_f32 v{12+i}, {b}, v{20+i}, v{12+i}"])
    if case == "partial":
        # Scale A's red channel while B is still outstanding.
        wait = program.index("s_waitcnt vmcnt(0)")
        first_mul = ops.pop(0).replace(", v12", ", v24")
        program[wait:wait] = ["s_waitcnt vmcnt(1)", "v_mov_b32_e32 v24, v12", first_mul]
        pos += 3
    program[pos:pos] = ops
    return program


def translate_dual(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], second_texture_offset=32)


def shader_record(blob: bytes, name: bytes) -> bytes:
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (bytes([len(name) + 1]) + name + b"\0\0"
            + struct.pack("<4I", len(payload), 0, len(payload), len(blob)) + payload)


def envelope(code: bytes, name: bytes = b"fixture", *, private_data: bytes | None = None, directory_end=None,
             compute_width=None, mesh_limits=None) -> bytes:
    program_start = 288 if directory_end is not None else 256
    image = bytearray(program_start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, program_start - 40)
    if directory_end is not None:
        extent, end_field = directory_end
        struct.pack_into("<I", image, 4, 0x101)
        struct.pack_into("<Q", image, 200, extent)
        struct.pack_into("<Q", image, 232, end_field)
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(image)
    program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
               + struct.pack("<III", 13, 48, len(native)) + native)
    if compute_width is not None:
        program = program[:2] + struct.pack("<H", 13 if mesh_limits else 5) + program[4:]
    sections = [b"DXIL" + struct.pack("<I", len(program)) + program]
    if compute_width is not None:
        psv = bytearray(52)
        struct.pack_into("<I", psv, 0, 48)
        psv[28] = 13 if mesh_limits else 5
        if mesh_limits:
            psv[31] = 2
            struct.pack_into("<HH", psv, 16, *mesh_limits)
        struct.pack_into("<3I", psv, 40, compute_width, 1, 1)
        sections.append(b"PSV0" + struct.pack("<I", len(psv)) + psv)
    if private_data is not None:
        sections.append(b"PRIV" + struct.pack("<I", len(private_data)) + private_data)
    offset = 32 + 4 * len(sections)
    offsets = bytearray()
    for section in sections:
        offsets.extend(struct.pack("<I", offset))
        offset += len(section)
    blob = b"DXBC" + bytes(16) + struct.pack("<3I", 1, offset, len(sections)) + offsets + b"".join(sections)
    return shader_record(blob, name)


def must_reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"{function.__name__} accepted invalid input")


def expect_value_error(function, program, message):
    """Require a ValueError carrying message, so a rejection cannot pass for an unrelated reason."""
    try:
        function(program)
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"Expected rejection: {message}")


def compile_shader(compiler, source, shader, *, listing=None, mode=F32_DENORM_MODE, ieee=True):
    """Run DXC on one translated shader under an explicit binary32 denormal mode."""
    # DXC accepts -denorm only at shader model 6.2 and above; ps_6_2 is the minimum target.
    command = [compiler, *(["-Gis"] if ieee else []), "-denorm", mode,
               "-T", "ps_6_2", "-E", "main", "-Fo", shader]
    if listing is not None:
        command += ["-Fc", listing]
    command.append(source)
    subprocess.run(command, check=True)


def flush_ftz(value):
    """Flush one binary32 subnormal input or result to signed zero, as -denorm ftz would."""
    bits, = struct.unpack("<I", struct.pack("<f", value))
    if bits & 0x7f800000 == 0 and bits & 0x007fffff:
        return struct.unpack("<f", struct.pack("<I", bits & 0x80000000))[0]
    return value


def write_case(path, constants, pixels, *, nan_mask=None, integer_rt=False):
    """Write a renderer case: 256 constant bytes, then 64 expected float4 rows in row-major order.

    A single float4 broadcasts to every texel. The optional trailer appends one flags byte, whose
    bit 0 selects the lossless R32G32B32A32_UINT readback target. Floating-point cases may instead
    append a zero flags byte and 64 mask bytes for per-channel NaN-class checks. Cases written without
    integer_rt or nan_mask keep the original 1280-byte format exactly.
    """
    rows = [pixels] * 64 if len(pixels) == 4 and not isinstance(pixels[0], (list, tuple)) else list(pixels)
    if len(rows) != 64 or any(len(row) != 4 for row in rows):
        raise ValueError("A case needs 64 float4 expectations")
    data = bytearray(bytes(constants))
    for row in rows:
        data += struct.pack("<4f", *row)
    if integer_rt:
        if nan_mask is not None:
            raise ValueError("NaN-class masks require a floating-point render target")
        data.append(1)
    elif nan_mask is not None:
        if isinstance(nan_mask, int):
            nan_mask = bytes([nan_mask]) * 64
        elif len(nan_mask) != 64:
            raise ValueError("The NaN-class mask needs exactly 64 texel bytes")
        data.append(0)
        data += nan_mask
    Path(path).write_bytes(data)
    return Path(path)


def check_container_selector(directory: Path, code: bytes, library: str | None = None) -> None:
    archive = directory / "same-name.rpf"
    records = [envelope(code, private_data=metadata) for metadata in (b"first", b"second")]
    containers = []
    for record in records:
        archive.write_bytes(record)
        container, native = native_program(archive, "fixture")
        assert native == code
        containers.append(container)
    digests = [hashlib.sha256(blob).hexdigest() for blob in containers]
    first, second = records
    broken = b"\x08fixture\0\0" + struct.pack("<4I", 2, 0, 2, 8) + b"\x07\xff"
    archive.write_bytes(first + second)
    must_reject(native_program, archive, "fixture")
    for digest, blob in zip(digests, containers, strict=True):
        assert native_program(archive, "fixture", container_sha256=digest.upper()) == (blob, code)

    for order in (broken + first, first + broken):
        archive.write_bytes(order)
        assert native_program(archive, "fixture", container_sha256=digests[0]) == (containers[0], code)
    # A bad frame can claim bytes covering a subsequent record of the same name.
    archive.write_bytes(b"\x08fixture\0\0" + struct.pack("<4I", len(first) + 2, 0, len(first) + 2, 8)
                        + b"\x07\xff" + first)
    assert native_program(archive, "fixture", container_sha256=digests[0]) == (containers[0], code)
    archive.write_bytes(first * 257 + second)
    assert native_program(archive, "fixture", container_sha256=digests[1]) == (containers[1], code)

    archive.write_bytes(envelope(code))
    intact, _ = native_program(archive, "fixture")
    corrupt = intact[:-1] + bytes([intact[-1] ^ 1])
    archive.write_bytes(first + shader_record(corrupt, b"fixture"))
    try:
        native_program(archive, "fixture", container_sha256=hashlib.sha256(corrupt).hexdigest())
    except zlib.error as error:
        assert "incorrect data check" in str(error), error
    else:
        raise AssertionError("Container selection hid a corrupt native stream")

    for contents, digest, reason in (
        (first + second, hashlib.sha256(code).hexdigest(), "No record matches the name and container SHA-256"),
        (first + second, "bad", "Invalid decoded container SHA-256"),
        (first + broken + first + second, digests[0], "Multiple records match the name and container SHA-256"),
    ):
        archive.write_bytes(contents)
        try:
            native_program(archive, "fixture", container_sha256=digest)
        except ValueError as error:
            assert str(error) == reason, error
        else:
            raise AssertionError("Container selection accepted an invalid digest or ambiguous record")

    if library is None:
        return
    archive.write_bytes(first + second)
    output = directory / "selected.hlsl"
    command = [sys.executable, "-B", Path(pixel_aot.__file__), archive, "fixture", "--llvm", library,
               "--texture-sgpr", "20", "--interp-sgpr", "24", "--sampler-words", "1", "2", "3", "4",
               "--output", output]
    ambiguous = subprocess.run(command, capture_output=True, text=True)
    assert ambiguous.returncode == 1 and "Multiple records match the name" in ambiguous.stderr
    for digest in digests:
        result = subprocess.run([*command, "--container-sha256", digest], capture_output=True, text=True, check=True)
        summary = json.loads(result.stdout)
        assert summary["container_sha256"] == digest
        assert summary["native_sha256"] == hashlib.sha256(code).hexdigest()
        assert output.read_text(encoding="utf-8") == lower(assembly(False), 20, 24, [1, 2, 3, 4])
    wrong = subprocess.run([*command, "--container-sha256", "0" * 64], capture_output=True, text=True)
    assert wrong.returncode == 1 and "No record matches the name and container SHA-256" in wrong.stderr


def translate(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64)


def translate_int(instructions):
    """translate() for a program that exports through the bit-exact R32G32B32A32_UINT target."""
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True)


def with_constants(body):
    """Load constant words 3 to 6 into s44 to s47, then run body on the sample in v12 to v15."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc",
        "s_waitcnt lgkmcnt(0)",
    ]
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = body
    return program


# s_movk_i32 sign-extends its 16-bit immediate. s_bfe_u32 takes bytes 1 and 3 of each result, and each byte is
# widened as a binary16 subnormal and scaled by 2^24 to the exact integer: 0x8000 gives 0xffff8000 (bytes 0x80 and
# 0xff) and 0x7fff gives 0x00007fff (bytes 0x7f and 0x00). Zero extension would export 0 for byte 3 of 0x8000.
MOVK_BODY = [
    "s_movk_i32 s48, 0x8000",
    "s_movk_i32 s51, 0x7fff",
    "s_bfe_u32 s49, s48, 0x80018",
    "s_bfe_u32 s50, s48, 0x80008",
    "s_bfe_u32 s52, s51, 0x80008",
    "s_bfe_u32 s53, s51, 0x80018",
    *[line for channel, source in enumerate(("s49", "s50", "s52", "s53"))
      for line in (f"v_cvt_f32_f16_e32 v{12 + channel}, {source}", f"v_mul_f32_e32 v{12 + channel}, 0x4b800000, v{12 + channel}")],
]
MOVK_EXPECTED = [255.0, 128.0, 127.0, 0.0]


def check_movk_draw(directory, compiler, renderer, backend):
    source = directory / "movk.hlsl"
    shader = directory / "movk.cso"
    case = directory / "movk.case"
    words = [(imm - 0x10000 if imm & 0x8000 else imm) & 0xffffffff for imm in (0x8000, 0x7fff)]
    assert [float((words[0] >> 24) & 0xff), float((words[0] >> 8) & 0xff),
            float((words[1] >> 8) & 0xff), float((words[1] >> 24) & 0xff)] == MOVK_EXPECTED
    source.write_text(translate(with_constants(MOVK_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytes(256)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    write_case(case, constants, tuple(MOVK_EXPECTED))
    subprocess.run(draw, check=True)
    # Zero extension clears byte 3 of 0x8000; sign-extending 0x7fff as negative sets byte 3.
    for channel, value in ((0, 0.0), (3, 255.0)):
        wrong = list(MOVK_EXPECTED)
        wrong[channel] = value
        write_case(case, constants, tuple(wrong))
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"A wrong s_movk_i32 extension escaped detection in channel {channel}")


def uncompressed_export(body):
    """Run body on the sample, then export v12 to v15 as four uncompressed binary32 channels."""
    program = with_constants(body)
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:] = ["exp mrt0 v12, v13, v14, v15 done vm", "s_endpgm"]
    return program


def with_scalar_constant_reads() -> list[str]:
    program = with_constants(["v_mov_b32_e32 v12, s61", "v_mov_b32_e32 v13, s62", "v_mov_b32_e32 v14, s63"])
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = [
        "s_buffer_load_dword s61, s[44:47], null",
        "s_buffer_load_dword s62, s[44:47], 0x8",
        "s_buffer_load_dword s63, s[44:47], 0xfc",
    ]
    return program


def affine_coordinates() -> list[str]:
    program = assembly(False)
    high_address = "s_mov_b32 s21, 0"
    program.remove(high_address)
    restore = program.index("s_and_b64 exec, exec, s[50:51]")
    program[restore:restore] = [
        high_address,
        "s_load_dwordx4 s[44:47], s[20:21], 0x40",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], null",
        "s_waitcnt lgkmcnt(0)",
        "v_add_f32_e32 v8, 0.5, v8",
        "v_mul_legacy_f32_e32 v8, s62, v8",
        "v_fma_f32 v8, 0.5, v8, 0",
    ]
    return program


PAIR_LOADS = ("s_buffer_load_dwordx2 s[60:61], s[44:47], 0xc",
              "s_buffer_load_dwordx2 s[62:63], s[44:47], 0x14",
              "s_buffer_load_dwordx2 s[64:65], s[44:47], 0xf8")


def with_scalar_pair_reads() -> list[str]:
    """Read constant words 3 and 4, 5 and 6, 62 and 63 as pairs; the first pair spans two float4 rows."""
    program = with_constants([
        "v_mul_f32_e32 v12, s60, v12",
        "v_max_f32_e32 v12, s64, v12",
        "v_add_f32_e32 v13, s61, v13",
        "v_sub_f32_e32 v14, s62, v14",
        "v_mov_b32_e32 v15, s63",
        "v_mul_f32_e32 v15, s65, v15",
    ])
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = PAIR_LOADS
    return program


def scalar_pair_case(directory: Path, scale: float, bias: float, minuend: float, alpha: float,
                     floor: float, gain: float) -> Path:
    """Expected colors for with_scalar_pair_reads(). Every step is exact in binary32."""
    def exact(value):
        assert single(value) == value, value
        return value
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, *map(exact, (scale, bias, minuend, alpha)))
    struct.pack_into("<2f", case, 248, exact(floor), exact(gain))
    for y in RAMP:
        for x in RAMP:
            r, g, b = (single(c / 255) for c in (x, y, 128))
            color = (max(floor, exact(scale * r)), exact(bias + g), exact(minuend - b), exact(gain * alpha))
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / "scalar-pairs.case"
    path.write_bytes(case)
    return path


def curve():
    return with_constants([
        "v_max_f32_e32 v18, s46, v12",
        "v_log_f32_e32 v18, v18",
        "v_rcp_f32_e32 v19, s44",
        "v_mul_legacy_f32_e32 v18, v19, v18",
        "v_exp_f32_e32 v18, v18",
        "v_sub_f32_e32 v19, 1.0, v13",
        "v_fma_f32 v12, s45, v18, -v19",
        "v_fmac_f32_e32 v12, s47, v13",
        "v_min_f32_e32 v12, 1.0, v12",
        "v_mul_legacy_f32_e64 v12, v12, 1.0 clamp",
    ])


def shaping():
    return with_constants([
        "v_max_f32_e32 v18, s46, v12",
        "v_sqrt_f32_e32 v18, v18",
        "v_mul_f32_e32 v18, s45, v18",
        "v_add_f32_e32 v19, s44, v13",
        "v_rsq_f32_e32 v19, v19",
        "v_add_f32_e32 v18, v19, v18",
        "v_mul_f32_e32 v20, s44, v13",
        "v_floor_f32_e32 v21, v20",
        "v_fract_f32_e32 v20, v20",
        "v_fma_f32 v12, s47, v20, v18",
        "v_sub_f32_e32 v12, v12, v21",
        "v_mul_f32_e64 v12, v12, 1.0 clamp",
    ])


def truncate_values() -> list[str]:
    return with_constants([
        "v_add_f32_e32 v12, s44, v12",
        "v_trunc_f32_e32 v12, v12",
        "v_trunc_f32_e32 v13, s45",
        "v_trunc_f32_e64 v14, -|s46|",
        "v_trunc_f32_e64 v15, s47 clamp",
    ])


def sopk_scalar() -> list[str]:
    return with_constants([
        "s_mov_b32 s50, s44",
        "s_mulk_i32 s50, 0x3",
        "s_mov_b32 s51, s45",
        "s_addk_i32 s51, 0xfffe",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
    ])


SOPK_CMPK_OPS = ("eq", "lg", "gt", "ge", "lt", "le")
SOPK_CMPK_WORDS = (
    (0xffff7fff, 0x00007fff, 0x13579bdf, 0xffffffff),
    (0xffff8000, 0x00008000, 0x13579bdf, 0x0000ffff),
    (0x00008000, 0xffff8000, 0x2468ace0, 0xffffffff),
    (0xffff8001, 0x00008001, 0x2468ace0, 0),
    (0xffff8001, 0x00008001, 0xfffffffe, 0xffffffff),
    (0xffff8001, 0x00008001, 0x2468ace0, 1),
)


def sopk_cmpk_cmovk() -> list[str]:
    body = ["s_mov_b32 s52, 0", "s_mov_b32 s53, 0"]
    for bit, op in enumerate(SOPK_CMPK_OPS):
        flag = 1 << bit
        body += [
            f"s_cmpk_{op}_i32 s44, 0x8000",
            f"s_cselect_b32 s63, {flag}, 0",
            "s_or_b32 s52, s52, s63",
            f"s_cmpk_{op}_u32 s45, 0x8000",
            f"s_cselect_b32 s63, {flag}, 0",
            "s_or_b32 s53, s53, s63",
        ]
    body += [
        "s_cmpk_eq_u32 s45, 0x8001",
        "s_cselect_b32 s63, 64, 0",
        "s_or_b32 s53, s53, s63",
        "s_mov_b32 s50, s46",
        "s_mov_b32 s51, s46",
        "s_cmpk_eq_i32 s47, 0xffff",
        "s_cmovk_i32 s50, 0xfffe",
        "s_cmovk_i32 s51, 0x7fff",
        "s_cmpk_eq_i32 s47, 1",
        "s_cmovk_i32 s51, 0x8000",
        "v_mov_b32_e32 v12, s52",
        "v_mov_b32_e32 v13, s53",
        "v_mov_b32_e32 v14, s50",
        "v_mov_b32_e32 v15, s51",
    ]
    return uncompressed_export(body)


def sopk_cmpk_cmovk_reference(words):
    s44, s45, s46, s47 = words
    signed44 = s44 - (1 << 32) if s44 >> 31 else s44
    preds = {
        "eq": lambda a, b: a == b,
        "lg": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
    }
    mask_i32 = sum((1 << bit) for bit, op in enumerate(SOPK_CMPK_OPS) if preds[op](signed44, -0x8000))
    mask_u32 = sum((1 << bit) for bit, op in enumerate(SOPK_CMPK_OPS) if preds[op](s45, 0x8000))
    mask_u32 |= (1 << len(SOPK_CMPK_OPS)) * (s45 == 0x8001)
    take = s47 == 0xffffffff
    first = 0xfffffffe if take else s46
    second = {-1: 0x7fff, 1: -32768}.get(s47 - (1 << 32) if s47 >> 31 else s47, s46)
    return (mask_i32, mask_u32, first, second % (1 << 32))


def rounding():
    """Round scaled samples up and to nearest even, and a constant tie through the VOP3 encoding."""
    return with_constants([
        "v_mul_f32_e32 v18, s44, v12",
        "v_ceil_f32_e32 v12, v18",
        "v_mul_f32_e32 v19, s44, v13",
        "v_rndne_f32_e32 v13, v19",
        "v_rndne_f32_e64 v14, s45",
        "v_ceil_f32_e64 v15, -s46",
    ])


def select(wave64=False):
    """Compare into lane masks and select; LLVM names the masks by wavefront width."""
    vcc, first, second = ("vcc", "s[52:53]", "s[54:55]") if wave64 else ("vcc_lo", "s52", "s53")
    return with_constants([
        f"v_cmp_nlt_f32_e64 {second}, v12, v13",
        f"v_cndmask_b32_e64 v14, 0, 1.0, {second}",
        f"v_cmp_eq_f32_e32 {vcc}, v12, v13",
        f"v_cndmask_b32_e32 v15, s47, v15, {vcc}",
        f"v_cmp_lt_f32_e32 {vcc}, s44, v12",
        "v_sub_f32_e32 v18, 1.0, v12",
        f"v_cmp_ge_f32_e64 {first}, v13, s45",
        f"v_cndmask_b32_e64 v13, v13, s46, {first}",
        f"v_cndmask_b32_e32 v12, v12, v18, {vcc}",
    ])


def three_way():
    """Median, minimum and maximum of three operands, each result one of its inputs."""
    return with_constants([
        "v_med3_f32 v15, s47, v13, v12",
        "v_max3_f32 v14, v12, v14, v13",
        "v_med3_f32 v12, v12, s44, s45",
        "v_min3_f32 v13, |s46|, v13, v12",
    ])


def absolute_value():
    program = assembly(False)
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = [
        "v_sub_f32_e32 v12, 0.5, v12",
        "v_sub_f32_e64 v12, |v12|, 0",
    ]
    return program


def unpack_bytes():
    # Each output takes a different byte of a constant word, in e32 and e64 forms.
    return with_constants([
        "v_cvt_f32_ubyte0_e32 v12, s44",
        "v_cvt_f32_ubyte1_e64 v13, s45",
        "v_cvt_f32_ubyte2_e32 v14, s46",
        "v_cvt_f32_ubyte3_e64 v15, s47 clamp",
    ])


def offset_nibbles():
    # The low four bits of each constant word as a signed offset in sixteenths.
    return with_constants([
        "v_cvt_off_f32_i4_e32 v12, s44",
        "v_cvt_off_f32_i4_e64 v13, s45",
        "v_cvt_off_f32_i4_e32 v14, s46",
        "v_cvt_off_f32_i4_e64 v15, s47 clamp",
    ])


def legacy_multiply_add():
    # gfx10 allows two scalar operands per VOP3 instruction, so two constants move to VGPRs.
    return uncompressed_export([
        "v_mov_b32_e32 v18, s46",
        "v_mov_b32_e32 v19, s47",
        "v_mad_legacy_f32 v12, s44, s45, v18",
        "v_mad_legacy_f32 v13, s45, s44, -v19",
        "v_mad_legacy_f32 v14, |v18|, s47, s44",
        "v_mad_legacy_f32 v15, s44, v19, v18 clamp",
    ])


def ceiling():
    return with_constants([
        "v_ceil_f32_e32 v12, s44",
        "v_ceil_f32_e64 v13, -|s45|",
        "v_ceil_f32_e64 v14, |s46|",
        "v_ceil_f32_e64 v15, |s47| clamp",
    ])


def int_minmax_pair(encoding):
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        f"v_min_i32_{encoding} v12, s44, v18",
        f"v_max_i32_{encoding} v13, s44, v18",
        f"v_min_u32_{encoding} v14, s44, v18",
        f"v_max_u32_{encoding} v15, s44, v18",
    ])


def int_minmax_triple(kind):
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v19, s46",
        f"v_min3_{kind} v12, s44, v18, v19",
        f"v_max3_{kind} v13, s44, v18, v19",
        f"v_med3_{kind} v14, s44, v18, v19",
        f"v_med3_{kind} v15, s46, v18, v18",
    ])


def cvt_int_fixture(encoding):
    if encoding == "e32":
        return with_constants([
            "v_cvt_f32_i32_e32 v12, s44",
            "v_cvt_f32_u32_e32 v13, s45",
            "v_mov_b32_e32 v18, s46",
            "v_cvt_f32_i32_e32 v14, v18",
            "v_cvt_f32_u32_e32 v15, s47",
        ])
    return with_constants([
        "v_cvt_f32_i32_e64 v12, s44",
        "v_cvt_f32_u32_e64 v13, s45 clamp",
        "v_mov_b32_e32 v18, s46",
        "v_cvt_f32_i32_e64 v14, v18 clamp",
        "v_cvt_f32_u32_e64 v15, s47",
    ])


def cvt_int_literals():
    return with_constants([
        "v_cvt_f32_i32_e32 v12, -7",
        "v_cvt_f32_u32_e32 v13, 25",
        "v_cvt_f32_i32_e64 v14, -100 clamp",
        "v_cvt_f32_u32_e64 v15, 50 clamp",
    ])


def frexp_mant_values():
    return with_constants([
        "v_frexp_mant_f32_e32 v12, s44",
        "v_frexp_mant_f32_e64 v13, -|s45|",
        "v_frexp_mant_f32_e64 v14, |s46|",
        "v_frexp_mant_f32_e64 v15, s47 clamp",
    ])


def ldexp_values():
    return with_constants([
        "v_mov_b32_e32 v18, s44",
        "v_ldexp_f32 v12, v18, s45",
        "v_ldexp_f32 v13, -|s44|, 2",
        "v_ldexp_f32 v14, s46, -2",
        "v_ldexp_f32 v15, s47, 3 clamp",
    ])
# v_sin_f32 and v_cos_f32 take revolutions, sin(2 * pi * x). D3D bounds sin and cos only to 0.0008
# absolute in [-100 pi, 100 pi], so each result is quantized to 1/32 steps by exact operations
# (times 32, plus 0.5, floor, times 1/32) and every reference lies at least 0.1 step from a boundary.
TRIG_BODY = [
    "v_sin_f32_e32 v12, s44",
    "v_cos_f32_e32 v13, s45",
    "v_sin_f32_e64 v14, -|s46|",
    "v_cos_f32_e64 v15, -s47 clamp",
] + [line for reg in ("v12", "v13", "v14", "v15") for line in (
    f"v_mul_f32_e32 {reg}, 0x42000000, {reg}",
    f"v_add_f32_e32 {reg}, 0.5, {reg}",
    f"v_floor_f32_e32 {reg}, {reg}",
    f"v_mul_f32_e32 {reg}, 0x3d000000, {reg}",
)]
# Each case covers all four quadrants across its channels and one input of millions of turns, where
# multiplying the unreduced input by 2 pi in binary32 moves the quantized result by 20 or more steps.
TRIG_CASES = (
    (-4000000.25, 0.125, -0.375, 0.5),
    (-0.125, 4000000.25, 2.75, -0.125),
    (0.25, -1.625, -4000000.5, 0.0),
    (-0.375, 0.375, 0.0625, 2000000.75),
)


def trig_reference(case, swap=False):
    """The quantized four channels of TRIG_BODY, from a double-precision reduction of each input."""
    channels = []
    for kind, x, saturate in zip(("sin", "cos", "sin", "cos"), (case[0], case[1], -abs(case[2]), -case[3]),
                                 (False, False, False, True)):
        if swap:
            kind = "cos" if kind == "sin" else "sin"
        value = (math.sin if kind == "sin" else math.cos)(2 * math.pi * math.remainder(x, 1.0))
        if saturate:
            value = min(1.0, max(0.0, value))
        scaled = value * 32 + 0.5
        # Only a drawn reference needs the margin; the swapped values merely have to differ from it.
        assert swap or abs(scaled - round(scaled)) >= 0.1, (case, kind, value)
        channels.append(math.floor(scaled) / 32)
    return channels


def reciprocal_iflag():
    return with_constants([
        "v_rcp_iflag_f32_e32 v12, s44",
        "v_rcp_iflag_f32_e64 v13, -|s45|",
        "v_rcp_iflag_f32_e64 v14, |s46|",
        "v_rcp_iflag_f32_e64 v15, |s47| clamp",
    ])


# The gfx10 predicate number is a four-bit truth table. Bits 0..3 select
# less, equal, greater and unordered respectively, independent of the HLSL.
PREDICATES = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o",
              "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")


def predicate_group(first, sdwa=False):
    # Copy runtime constants into vector registers using supported operations.
    body = ["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0"]
    for channel, op in enumerate(PREDICATES[first:first + 4]):
        compare = (f"v_cmp_{op}_f32_sdwa vcc_lo, s44, v18 src0_sel:DWORD src1_sel:DWORD" if sdwa
                   else f"v_cmp_{op}_f32_e32 vcc_lo, s44, v18")
        body.extend([compare,
                     f"v_cndmask_b32_e32 v{12 + channel}, 0, v19, vcc_lo"])
    return with_constants(body)


# Three-bit truth table for signed and unsigned 32-bit integer comparisons (bits 0..2: <, ==, >).
INT_PREDICATES = ("f", "lt", "eq", "le", "gt", "ne", "ge", "t")


def int_predicate_group(kind, encoding, first, *, wave64=False):
    # Copy runtime constants with v_mov_b32 so all 32 bits survive unchanged.
    dst = ("vcc" if wave64 else "vcc_lo") if encoding == "e32" else ("s[52:53]" if wave64 else "s52")
    body = ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47"]
    for channel, op in enumerate(INT_PREDICATES[first:first + 4]):
        body.extend([f"v_cmp_{op}_{kind}_{encoding} {dst}, s44, v18",
                     f"v_cndmask_b32_{encoding} v{12 + channel}, 0, v19, {dst}"])
    return with_constants(body)


def unsigned_less_than(kind="u32"):
    return with_constants([
        "v_mov_b32_e32 v18, s44",
        "v_mov_b32_e32 v19, s45",
        "v_mov_b32_e32 v20, 1.0",
        f"v_cmp_lt_{kind}_e32 vcc_lo, s44, v19",
        "v_cndmask_b32_e32 v12, 0, v20, vcc_lo",
        f"v_cmp_lt_{kind}_e32 vcc_lo, s45, v18",
        "v_cndmask_b32_e32 v13, 0, v20, vcc_lo",
        f"v_cmp_lt_{kind}_e64 s52, v18, v19",
        "v_cndmask_b32_e64 v14, 0, v20, s52",
        f"v_cmp_lt_{kind}_e64 s52, v19, v18",
        "v_cndmask_b32_e64 v15, 0, v20, s52",
    ])


# 16-bit integer comparisons read only the low half of each source. L is s44, R is v18 (a copy of s45);
# other operands are an integer inline constant or a 16-bit literal. Each (kind, encoding) pair has two
# groups of four channels, so all six predicates appear with SGPR and VGPR operands, and the second group
# adds inline constants at both ends of their range and a literal with bit 15 set.
INT16_GROUPS = {
    "e32": ((("lt", "L", "R"), ("eq", "L", "R"), ("le", "L", "R"), ("gt", "L", "R")),
            (("ne", "L", "R"), ("ge", "L", "R"), ("lt", "-16", "R"), ("gt", "0x8000", "R"))),
    "e64": ((("lt", "L", "R"), ("eq", "L", "R"), ("le", "L", "R"), ("gt", "L", "R")),
            (("ne", "L", "R"), ("ge", "R", "L"), ("lt", "R", "64"), ("le", "R", "0x8000"))),
}


def int16_group(kind, encoding, group, *, wave64=False):
    dst = ("vcc" if wave64 else "vcc_lo") if encoding == "e32" else ("s[52:53]" if wave64 else "s52")
    body = ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47"]
    names = {"L": "s44", "R": "v18"}
    for channel, (op, a, b) in enumerate(INT16_GROUPS[encoding][group]):
        body.extend([f"v_cmp_{op}_{kind}_{encoding} {dst}, {names.get(a, a)}, {names.get(b, b)}",
                     f"v_cndmask_b32_{encoding} v{12 + channel}, 0, v19, {dst}"])
    return with_constants(body)


def int16_reference(kind, encoding, group, left, right):
    """Expected channels of int16_group for 32-bit words left (s44) and right (s45)."""
    def value(token):
        word = {"L": left, "R": right}.get(token)
        word = int(token, 0) if word is None else word
        low = word & 0xffff
        return low - 0x10000 if kind == "i16" and low & 0x8000 else low

    tests = {"lt": lambda a, b: a < b, "eq": lambda a, b: a == b, "le": lambda a, b: a <= b,
             "gt": lambda a, b: a > b, "ne": lambda a, b: a != b, "ge": lambda a, b: a >= b}
    return [float(tests[op](value(a), value(b))) for op, a, b in INT16_GROUPS[encoding][group]]


# v_cmp_class_f32 is true when the class of S0 is in the mask S1 (LLVM SIInstrFlags::ClassFlags).
# Each channel tests one property: +inf against its own bit, a quiet NaN against both NaN bits,
# -0 against the +0 bit and a positive denormal against its bit, so neither the sign of zero
# nor a denormal flushed to zero can pass.
CLASS_BITS = (0x7f800000, 0x7fc00000, 0x80000000, 0x00000001)
CLASS_BODY = [
    "v_mov_b32_e32 v19, 1.0",
    "v_cmp_class_f32_e64 vcc_lo, s44, 0x200",
    "v_cndmask_b32_e32 v12, 0, v19, vcc_lo",
    "v_cmp_class_f32_e64 vcc_lo, s45, 3",
    "v_cndmask_b32_e32 v13, 0, v19, vcc_lo",
    "v_mov_b32_e32 v18, 64",
    "v_cmp_class_f32_e32 vcc_lo, s46, v18",
    "v_cndmask_b32_e32 v14, 0, v19, vcc_lo",
    "v_cmp_class_f32_e64 s0, s47, 0x80",
    "v_cndmask_b32_e64 v15, 0, v19, s0",
]


def check_class_draw(directory, compiler, renderer, backend):
    source = directory / "class.hlsl"
    shader = directory / "class.cso"
    case = directory / "class.case"
    source.write_text(translate(with_constants(CLASS_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *CLASS_BITS)
    expected = [1.0, 1.0, 0.0, 1.0]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
    subprocess.run(draw, check=True)
    # A wrong class in any channel flips its result; the renderer must report each flip.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = 1.0 - wrong_values[k]
        case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"A wrong class result escaped detection in channel {k}")


# v_fract_f32 clamps x - floor(x) to 0x3f7fffff. For -2**-30 the unclamped difference rounds to 1.0;
# the other inputs are exact and stay below the clamp.
FRACT_INPUTS = (-2.0 ** -30, -0.25, 2.75, -2.0 ** -23)
FRACT_CLAMP = struct.unpack("<f", struct.pack("<I", 0x3f7fffff))[0]


def fract_values():
    return with_constants([
        "v_fract_f32_e32 v12, s44",
        "v_fract_f32_e64 v13, |s45|",
        "v_fract_f32_e64 v14, -|s46|",
        "v_fract_f32_e32 v15, s47",
    ])


RECIPROCAL_BODY = [
    "v_rcp_f32_e64 v12, |s44|",
    "v_rcp_f32_e64 v13, -|s45|",
    "v_rsq_f32_e64 v14, |s46|",
    "v_rsq_f32_e64 v15, |s47| clamp",
]


def check_reciprocal_draw(directory, compiler, renderer, backend):
    source = directory / "reciprocal.hlsl"
    shader = directory / "reciprocal.cso"
    case = directory / "reciprocal.case"
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -4.0, 8.0, -16.0, -0.25)
    case.write_bytes(constants + struct.pack("<4f", 0.25, -0.125, 0.25, 1.0) * 64)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for mutation in (False, True):
        body = RECIPROCAL_BODY.copy()
        if mutation:
            # 1 / abs(-16) is 1/16, while its reciprocal square root is 1/4.
            body[2] = body[2].replace("v_rsq", "v_rcp")
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        if mutation:
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("Substituting reciprocal for reciprocal square root escaped detection")
        else:
            subprocess.run(draw, check=True)
# VOP3 (e64) encodings that the translator already accepts but no other fixture draws: floor, fract, max, min and
# fmac with source modifiers, and the 16 comparisons writing an SGPR mask from negated operands. Negating both
# operands swaps the less and greater relations and keeps equality and unordered.
E64_ARITH = (
    (["v_floor_f32_e64 v12, -|s44|", "v_fract_f32_e64 v13, -s45",
      "v_max_f32_e64 v14, -s46, |s47|", "v_min_f32_e64 v15, s46, -|s47|"],
     [-3.0, 0.75, 4.0, -4.0], ((0, 2.0), (1, 0.25), (2, -3.0), (3, 3.0))),
    (["v_mov_b32_e32 v12, s46", "v_fmac_f32_e64 v12, -s44, |s45|",
      "v_mov_b32_e32 v13, 0", "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0"],
     [-0.125, 0.0, 0.0, 0.0], ((0, 6.125),)),
)
E64_ARITH_INPUTS = (2.5, 1.25, 3.0, -4.0)


def predicate_group_e64(first):
    body = ["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0"]
    for channel, op in enumerate(PREDICATES[first:first + 4]):
        body.extend([f"v_cmp_{op}_f32_e64 s0, -s44, -v18", f"v_cndmask_b32_e64 v{12 + channel}, 0, v19, s0"])
    return with_constants(body)


def run_pixel_batch(renderer, backend, shader, cases):
    command = [renderer, backend, "--check-pixel-batch", shader, *(path for path, _ in cases)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        result.check_returncode()
    expected = [f"case={i} status={status}" for i, (_, status) in enumerate(cases)]
    actual = [line for line in result.stdout.splitlines() if line.startswith("case=")]
    if actual != expected or result.returncode != int(any(status != "pass" for _, status in cases)):
        raise AssertionError(f"Expected {expected}\n{result.stdout}\n{result.stderr}")


def check_e64_draws(directory, compiler, renderer, backend):
    source = directory / "e64.hlsl"
    shader = directory / "e64.cso"
    def add_case(cases, constants, expected, status):
        path = directory / f"e64-{len(cases)}.case"
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((path, status))

    x, y, z, w = E64_ARITH_INPUTS
    assert [math.floor(-abs(x)), -y - math.floor(-y), max(-z, abs(w)), min(z, -abs(w))] == E64_ARITH[0][1]
    assert [-x * abs(y) + z, 0.0, 0.0, 0.0] == E64_ARITH[1][1]
    for body, expected, wrongs in E64_ARITH:
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *E64_ARITH_INPUTS)
        cases = []
        add_case(cases, constants, expected, "pass")
        for channel, value in wrongs:
            wrong_values = list(expected)
            wrong_values[channel] = value
            add_case(cases, constants, wrong_values, "mismatch")
        add_case(cases, constants, expected, "pass")
        run_pixel_batch(renderer, backend, shader, cases)
    inputs = ((-1, 1, 0), (-0.0, 0.0, 1), (1, -1, 2), (math.nan, 1, 3), (1, math.nan, 3))
    negated = {0: 2, 1: 1, 2: 0, 3: 3}
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group_e64(first)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            expected = [float((op >> negated[relation]) & 1) for op in range(first, first + 4)]
            add_case(cases, constants, expected, "pass")
            # A lowering that dropped the negations would follow the original relation.
            unnegated = [float((op >> relation) & 1) for op in range(first, first + 4)]
            if unnegated != expected:
                add_case(cases, constants, unnegated, "mismatch")
        run_pixel_batch(renderer, backend, shader, cases)


def bitwise_program():
    # Emit exact finite binary32 colors by placing result bits directly in their mantissas.
    return with_constants([
        "v_mov_b32_e32 v18, s46",
        "v_mov_b32_e32 v19, s47",
        "v_lshlrev_b32_e32 v20, s44, v18",
        "v_lshrrev_b32_e32 v20, s45, v20",
        "v_ashrrev_i32_e32 v21, 20, v19",
        "v_not_b32_e32 v22, v18",
        "v_bfi_b32 v23, 0x00700000, v18, v19",
        "v_xor_b32_e32 v22, v19, v22",
        *[instruction for index in range(4) for instruction in (
            f"v_and_b32_e32 v{12+index}, 0x007f0000, v{20+index}",
            f"v_or_b32_e32 v{12+index}, 0x3f800000, v{12+index}",
        )],
    ])


def check_bitwise_draw(directory, compiler, renderer, backend):
    shader = directory / "bits.cso"
    source = directory / "bits.hlsl"
    case = directory / "bits.case"
    for logical in (False, True):
        program = bitwise_program()
        if logical:
            program = [line.replace("v_ashrrev_i32", "v_lshrrev_b32") for line in program]
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        for left, right, a, b in ((36, 33, 0x40d10000, 0xc0250000), (35, 37, 0x3e270000, 0xbf620000)):
            words = (((a << (left % 32)) & 0xffffffff) >> (right % 32),
                     (b - 2**32) >> 20, ~a ^ b, (a & 0x00700000) | (b & ~0x00700000))
            colors = struct.pack("<4I", *[0x3f800000 | (value & 0x007f0000) for value in words])
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, left, right, a, b)
            case.write_bytes(constants + colors * 64)
            draw = [renderer, backend, "--check-pixel-case", shader, case]
            if logical:
                wrong = subprocess.run(draw, capture_output=True, text=True)
                if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                    raise AssertionError("Logical shift substituted for arithmetic shift escaped detection")
            else:
                subprocess.run(draw, check=True)


def check_fract_draw(directory, compiler, renderer, backend):
    source = directory / "fract.hlsl"
    shader = directory / "fract.cso"
    case = directory / "fract.case"
    source.write_text(translate(fract_values()), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *FRACT_INPUTS)
    a, b, c, d = FRACT_INPUTS
    expected = [min(single(x - math.floor(x)), FRACT_CLAMP) for x in (a, abs(b), -abs(c), d)]
    assert expected[0] == FRACT_CLAMP and single(FRACT_INPUTS[0] - math.floor(FRACT_INPUTS[0])) == 1.0
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    write_case(case, constants, tuple(map(half_rtz, expected)))
    subprocess.run(draw, check=True)
    # An unclamped fract exports 1.0 for the first input; the renderer must tell it from the clamp.
    write_case(case, constants, (1.0, *map(half_rtz, expected[1:])))
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("An unclamped fract result escaped detection")


# v_mad_f32 and v_mac_f32 round the product before the add and flush binary32 denormals.
# 4097 * 4097 = 2**24 + 2**13 + 1 rounds to 2**24 + 2**13, so the unfused sum is 0 where a fused one is 1.0.
# 2**-127 is a denormal input: flushed, the second sum is 0.5; preserved, it is 1.0.
MAD_INPUTS = (4097.0, -16785408.0, 2.0 ** -127, 2.0 ** 126)
MAD_BODY = [
    "v_mad_f32 v12, s44, s44, s45",
    "v_mov_b32_e32 v13, s47",
    "v_mad_f32 v13, s46, v13, 0.5",
    "v_mov_b32_e32 v15, s44",
    "v_mov_b32_e32 v14, s45",
    "v_mac_f32_e32 v14, s44, v15",
    "v_mad_f32 v15, 0.5, 4.0, 0.5",
]


def check_mad_draw(directory, compiler, renderer, backend):
    source = directory / "mad.hlsl"
    shader = directory / "mad.cso"
    case = directory / "mad.case"
    source.write_text(translate(with_constants(MAD_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *MAD_INPUTS)
    a, c, tiny, large = MAD_INPUTS
    unfused = single(single(a * a) + c)
    assert unfused == 0.0 and single(a * a + c) == 1.0
    assert single(tiny) == tiny and 0 < tiny < 2.0 ** -126 and single(tiny * large) == 0.5
    expected = [unfused, 0.5, unfused, 2.5]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, expected)) * 64)
    subprocess.run(draw, check=True)
    # A fused multiply-add or a denormal-preserving one exports 1.0; the renderer must tell each case apart.
    for wrong_values in ([1.0, 0.5, unfused, 2.5], [unfused, 1.0, unfused, 2.5], [unfused, 0.5, 1.0, 2.5]):
        case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, wrong_values)) * 64)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"A fused or denormal-preserving result escaped detection: {wrong_values}")


# v_madak_f32 computes S0 * S1 + K and v_madmk_f32 computes S0 * K + S1, with K a 32-bit literal. Unlike the
# fma forms they round the product before the add and flush binary32 denormals, like v_mad_f32.
# 4097 * 4097 - 16785408 is 0 unfused and 1.0 fused; 2^-127 * 2^126 + 0.5 is 0.5 with the denormal input flushed.
MAD_LITERAL_BODY = [
    "v_mov_b32_e32 v13, s44",
    "v_madak_f32 v12, s44, v13, 0xcb801000",
    "v_mov_b32_e32 v13, s45",
    "v_madmk_f32 v13, s44, 0x45800800, v13",
    "v_mov_b32_e32 v15, s47",
    "v_madak_f32 v14, s46, v15, 0x3f000000",
    "v_mov_b32_e32 v15, 0.5",
    "v_madmk_f32 v15, s46, 0x7e800000, v15",
]


def check_mad_literal_draw(directory, compiler, renderer, backend):
    source = directory / "mad_literal.hlsl"
    shader = directory / "mad_literal.cso"
    case = directory / "mad_literal.case"
    source.write_text(translate(with_constants(MAD_LITERAL_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *MAD_INPUTS)
    a, c, tiny, large = MAD_INPUTS
    # Exact rational arithmetic gives the fused result; the unfused one rounds the product to binary32 first.
    fused = float(Fraction(a) * Fraction(a) + Fraction(c))
    unfused = single(single(a * a) + c)
    assert fused == 1.0 and unfused == 0.0
    assert single(tiny) == tiny and 0 < tiny < 2.0 ** -126 and single(tiny * large) == 0.5
    expected = [unfused, unfused, 0.5, 0.5]
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    write_case(case, constants, tuple(map(half_rtz, expected)))
    subprocess.run(draw, check=True)
    # A reference that fuses the first two rows or keeps the denormal in the last two exports 1.0 there.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = 1.0
        write_case(case, constants, tuple(map(half_rtz, wrong_values)))
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"A fused or denormal-preserving literal result escaped detection in channel {k}")


def check_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "predicates.hlsl"
    shader = directory / "predicates.cso"
    # Isolate each negated relation so another predicate cannot supply the unordered test. An
    # unordered relational fcmp alone is not enough: the Windows 10 in-box WARP 10.0.19041
    # evaluates fcmp ult, ule, ugt and uge as their ordered forms.
    for op in ("nge", "nlg", "ngt", "nle", "nlt"):
        isolated = with_constants(["v_sub_f32_e64 v18, s45, 0",
                                   f"v_cmp_{op}_f32_e32 vcc_lo, s44, v18",
                                   "v_cndmask_b32_e32 v12, 0, 1.0, vcc_lo"])
        source.write_text(translate(isolated), encoding="ascii")
        listing = directory / f"{op}.ll"
        compile_shader(compiler, source, shader, listing=listing)
        dxil = listing.read_text(encoding="utf-8")
        if not any(text in dxil for text in ("@dx.op.isSpecialFloat.f32(i32 8,", "fcmp uno ", "fcmp ueq ")):
            raise AssertionError(f"{op.upper()} lacks an explicit unordered comparison in DXIL")
    inputs = (("lt", -1, 1, 0), ("signed-zero", -0.0, 0.0, 1), ("gt", 1, -1, 2),
              ("left-nan", math.nan, 1, 3), ("right-nan", 1, math.nan, 3))
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group(first)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for label, left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            pixel = tuple((op >> relation) & 1 for op in range(first, first + 4))
            case = directory / f"predicates-{first}-{label}.case"
            write_case(case, constants, pixel)
            cases.append((case, "pass"))
        if first == 12:
            # Keep the unordered oracle and replace the final NaN operand with an equal value.
            struct.pack_into("<f", constants, 16, 1.0)
            wrong = directory / f"predicates-{first}-finite-instead-of-nan.case"
            write_case(wrong, constants, pixel)
            cases.insert(-1, (wrong, "mismatch"))
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except AssertionError as error:
            channels = ", ".join(f"{channel}={op}" for channel, op in enumerate(PREDICATES[first:first + 4]))
            error.add_note(f"Predicate draw on {backend}, first={first}, channels {channels}")
            raise


def check_int_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "int_predicates.hlsl"
    shader = directory / "int_predicates.cso"
    # Normal and signed-zero IEEE-754 bit patterns avoid denormal/NaN canonicalization while covering
    # signed/unsigned order flips (-0.0 vs +0.0, +1.0 vs -1.0) and integer-vs-float order (-2.0 vs -1.0).
    pairs = (
        (0x80000000, 0x00000000),
        (0xbf800000, 0xbf800000),
        (0x3f800000, 0xbf800000),
        (0xc0000000, 0xbf800000),
    )

    def relation(left_u, right_u, kind):
        if kind == "i32":
            left_v = left_u - 0x100000000 if left_u & 0x80000000 else left_u
            right_v = right_u - 0x100000000 if right_u & 0x80000000 else right_u
        else:
            left_v, right_v = left_u, right_u
        return 0 if left_v < right_v else (1 if left_v == right_v else 2)

    for kind in ("i32", "u32"):
        for encoding in ("e32", "e64"):
            for first in (0, 4):
                source.write_text(translate(int_predicate_group(kind, encoding, first)), encoding="ascii")
                compile_shader(compiler, source, shader)
                cases = []
                for left_u, right_u in pairs:
                    rel = relation(left_u, right_u, kind)
                    constants = bytearray(256)
                    struct.pack_into("<IIff", constants, 12, left_u, right_u, 0.0, 1.0)
                    pixel = struct.pack("<4f", *((op >> rel) & 1 for op in range(first, first + 4)))
                    case = directory / f"int_predicates-{kind}-{encoding}-{first}-{left_u:08x}-{right_u:08x}.case"
                    case.write_bytes(constants + pixel * 64)
                    cases.append((case, "pass"))
                if (kind, encoding, first) == ("i32", "e32", 0):
                    # Negative control: the signed shader against the unsigned oracle on (0x80000000, 0x00000000).
                    constants = bytearray(256)
                    struct.pack_into("<IIff", constants, 12, 0x80000000, 0x00000000, 0.0, 1.0)
                    unsigned_rel = relation(0x80000000, 0x00000000, "u32")
                    wrong_pixel = struct.pack("<4f", *((op >> unsigned_rel) & 1 for op in range(4)))
                    wrong = directory / f"int_predicates-{kind}-{encoding}-{first}-unsigned-reference.case"
                    wrong.write_bytes(constants + wrong_pixel * 64)
                    cases.insert(-1, (wrong, "mismatch"))
                try:
                    run_pixel_batch(renderer, backend, shader, cases)
                except AssertionError as error:
                    error.add_note(f"Integer predicate draw on {backend}, {kind} {encoding} first={first}")
                    raise


def check_unsigned_less_than_draws(directory, compiler, renderer, backend):
    source = directory / "unsigned_less_than.hlsl"
    shader = directory / "unsigned_less_than.cso"
    source.write_text(translate(unsigned_less_than()), encoding="ascii")
    compile_shader(compiler, source, shader)

    cases = []
    # Quieting the first word of the NaN pair would reverse the unsigned ordering.
    pairs = (("sign-boundary", 0x80000000, 0x7fffffff),
             ("equal", 0x7fffffff, 0x7fffffff),
             ("nan-payload", 0x7f800002, 0x7fc00001))
    for label, left, right in pairs:
        constants = bytearray(256)
        struct.pack_into("<II", constants, 12, left, right)
        order = (float(left < right), float(right < left))
        case = directory / f"unsigned-less-than-{label}.case"
        write_case(case, constants, order * 2)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    source.write_text(translate(unsigned_less_than("i32")), encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(cases[0][0], "mismatch")])


# Each word is a normal binary32 value, so no float move can flush or quiet it, and its upper half differs
# from the lower one wherever a 32-bit comparison would give another answer.
INT16_PAIRS = (
    (0x3f808000, 0x40000000),  # 0x8000 vs 0: signed less, unsigned greater
    (0x40000005, 0x3f800005),  # equal low halves; the whole words compare greater
    (0xbf80ffff, 0x3f800001),  # -1 vs 1 signed, 0xffff vs 1 unsigned
    (0x3f807fff, 0xbf808000),  # 0x7fff vs 0x8000; right equals the 0x8000 literal
    (0x3f80fff0, 0x3f800040),  # -16 vs 64; right equals the inline constant 64
    (0x3f800040, 0x3f80fff0),  # 64 vs -16; right equals the inline constant -16
)


def check_int16_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "int16_predicates.hlsl"
    shader = directory / "int16_predicates.cso"
    case = directory / "int16_predicates.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]

    def build(program):
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)

    def write_case(left, right, expected, path=case):
        constants = bytearray(256)
        struct.pack_into("<IIff", constants, 12, left, right, 0.0, 1.0)
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        return path

    for kind in ("i16", "u16"):
        for encoding in ("e32", "e64"):
            for group in (0, 1):
                build(int16_group(kind, encoding, group))
                cases = []
                for left, right in INT16_PAIRS:
                    path = directory / f"int16_predicates-{kind}-{encoding}-{group}-{left:08x}-{right:08x}.case"
                    write_case(left, right, int16_reference(kind, encoding, group, left, right), path)
                    cases.append((path, "pass"))
                try:
                    run_pixel_batch(renderer, backend, shader, cases)
                except AssertionError as error:
                    error.add_note(f"16-bit predicate draw on {backend}, {kind} {encoding} group {group}")
                    raise

    def must_mismatch(program, left, right, expected, message):
        build(program)
        write_case(left, right, expected)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(message)

    left, right = INT16_PAIRS[0]
    must_mismatch(int16_group("i16", "e32", 0), left, right, int16_reference("u16", "e32", 0, left, right),
                  "A signed 16-bit comparison matched the unsigned reference")
    left, right = INT16_PAIRS[1]
    swapped = [line.replace("v_cmp_eq_i16", "v_cmp_ne_i16") for line in int16_group("i16", "e64", 0)]
    must_mismatch(swapped, left, right, int16_reference("i16", "e64", 0, left, right),
                  "Replacing eq with ne escaped detection")
    widened = [line.replace("_u16_", "_u32_") for line in int16_group("u16", "e32", 0)]
    must_mismatch(widened, left, right, int16_reference("u16", "e32", 0, left, right),
                  "A 32-bit comparison matched the 16-bit reference")


# The SDWA comparisons with whole-dword selections repeat the 16-predicate draws, including NaN on either side.
# For each group a reference that inverts the unordered (NaN) result must report a mismatch.
def check_sdwa_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "sdwa_predicates.hlsl"
    shader = directory / "sdwa_predicates.cso"
    inputs = (("lt", -1, 1, 0), ("signed-zero", -0.0, 0.0, 1), ("gt", 1, -1, 2),
              ("left-nan", math.nan, 1, 3), ("right-nan", 1, math.nan, 3))
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group(first, sdwa=True)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for label, left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            expected = [float((op >> relation) & 1) for op in range(first, first + 4)]
            case = directory / f"sdwa_predicates-{first}-{label}.case"
            case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
            cases.append((case, "pass"))
        # Keep the final NaN input and invert its reference; a passing case follows the control.
        wrong = directory / f"sdwa_predicates-{first}-inverted-unordered.case"
        wrong.write_bytes(constants + struct.pack("<4f", *(1.0 - value for value in expected)) * 64)
        cases.insert(-1, (wrong, "mismatch"))
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except AssertionError as error:
            channels = ", ".join(f"{channel}={op}" for channel, op in enumerate(PREDICATES[first:first + 4]))
            error.add_note(f"SDWA predicate draw on {backend}, first={first}, channels {channels}")
            raise


def half_rtz(x):
    bits, = struct.unpack("<H", struct.pack("<e", x))
    rounded, = struct.unpack("<e", struct.pack("<H", bits))
    return struct.unpack("<e", struct.pack("<H", bits - (abs(rounded) > abs(x))))[0]


# v_cvt_f32_f16 widens the binary16 in the low half of S0; the high half holds junk that must be ignored.
# The second channel applies -|x| to -5.0, the third is the largest finite half and the fourth is the
# smallest subnormal half, 2**-24, scaled by 2**24 so a flushed subnormal exports 0 instead of 1.0.
HALF_WORDS = (0xdead3c00, 0x0000c500, 0x00007bff, 0xbeef0001)
HALF_BODY = [
    "v_cvt_f32_f16_e32 v12, s44",
    "v_cvt_f32_f16_e64 v13, -|s45|",
    "v_cvt_f32_f16_e32 v14, s46",
    "v_cvt_f32_f16_e32 v15, s47",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]


def half_value(word):
    return struct.unpack("<e", struct.pack("<H", word & 0xffff))[0]


def check_half_draw(directory, compiler, renderer, backend):
    source = directory / "half.hlsl"
    shader = directory / "half.cso"
    case = directory / "half.case"
    source.write_text(translate(with_constants(HALF_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *HALF_WORDS)
    first, second, third, fourth = map(half_value, HALF_WORDS)
    expected = [first, -abs(second), third, fourth * 2.0 ** 24]
    assert expected == [1.0, -5.0, 65504.0, 1.0] and fourth == 2.0 ** -24
    # Reading the high half, dropping the negation, a wrong exponent and a flushed subnormal.
    wrong = [half_value(HALF_WORDS[0] >> 16), 5.0, 32752.0, 0.0]
    for x in (*expected, *wrong):
        assert half_rtz(x) == x, x
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    write_case(case, constants, tuple(expected))
    subprocess.run(draw, check=True)
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = wrong[k]
        write_case(case, constants, tuple(wrong_values))
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"A wrong half conversion escaped detection in channel {k}")

    # A separate export observes the literal and two scalar results as half encodings.
    body = [
        "v_cvt_f32_f16_e32 v12, 0x3c00",
        "s_ashr_i64 s[72:73], 0xf000, 2",
        "v_cvt_f32_f16_e32 v13, s72",
        "s_bfm_b64 s[72:73], 16, 0",
        "s_and_b64 s[72:73], s[72:73], 0x3c00",
        "v_cvt_f32_f16_e32 v14, s72",
        *HALF_BODY[3:],
    ]
    source.write_text(translate(with_constants(body)), encoding="ascii")
    compile_shader(compiler, source, shader)
    scalars = [half_value(0x3c00), half_value(0xf000 // 4), half_value((2**16 - 1) & 0x3c00), fourth * 2.0 ** 24]
    assert scalars == [1.0, 1.0, 1.0, 1.0]
    case.write_bytes(constants + struct.pack("<4f", *scalars) * 64)
    subprocess.run(draw, check=True)
    # A shift by zero leaves 0xf000 (-8192.0) in green.
    wrong_values = [scalars[0], half_value(0xf000), *scalars[2:]]
    case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("A wrong s_ashr_i64 count escaped detection in channel 1")


def pack_f16_pair(lo, hi):
    lo_bits = lo if isinstance(lo, int) else struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_bits = hi if isinstance(hi, int) else struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_bits | (hi_bits << 16)


def rne_f16(value):
    if math.isnan(value):
        return math.nan
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


# Packed binary16 VOP3P arithmetic: exercises all five v_pk_*_f16 instructions, op_sel/op_sel_hi,
# neg_lo/neg_hi, clamp, NaN minNum semantics, RNE tie-to-even rounding, and gfx1013 zero-extension
# of inline f16 constants (0.5 -> lo=0.5, hi=0.0).
VOP3P_BODY = [
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
    "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
    "v_pk_fma_f16 v24, v23, v21, s47 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
    "v_pk_min_f16 v25, v24, 0x40007e00",
    "v_pk_max_f16 v26, v24, 0.5 clamp",
    "v_cvt_f32_f16_e32 v12, v25",
    "v_pk_add_f16 v27, v25, 0 op_sel:[1,0]",
    "v_cvt_f32_f16_e32 v13, v27",
    "v_cvt_f32_f16_e32 v14, v26",
    "v_pk_add_f16 v28, v26, 0 op_sel:[1,0]",
    "v_cvt_f32_f16_e32 v15, v28",
]
VOP3P_SETS = (
    # Set 1: v24=(0.875, 0.25); channel 0 tests minNum(0.875, qNaN)=0.875, channel 3=0.25 proves inline 0.5 hi=0.0.
    (pack_f16_pair(1.5, 2.0), pack_f16_pair(0.5, -1.0), pack_f16_pair(0.5, 1.0), pack_f16_pair(0.5, 0.5)),
    # Set 2: v24=(2.5, -1.0); channel 2 clamps 2.5 -> 1.0 and channel 3 clamps -1.0 -> 0.0.
    (pack_f16_pair(2.0, 3.0), pack_f16_pair(1.0, 0.0), pack_f16_pair(1.0, 2.0), pack_f16_pair(0.5, 1.0)),
    # Set 3: exact halfway ties at 1.0 + 2^-11 (-> even 1.0) and -(1.0 + 2^-10) - 2^-11 (-> even -(1.0 + 2^-9),
    # distinguishing RNE from RTZ f32tof16 in channel 1).
    (pack_f16_pair(0xbc01, 0x3c00), pack_f16_pair(0x9000, 0x9000), pack_f16_pair(4.0, 4.0), pack_f16_pair(0.25, 0.25)),
)


def vop3p_reference(words):
    def split16(w):
        return half_value(w), half_value(w >> 16)

    def min_num(a, b):
        return b if math.isnan(a) else (a if math.isnan(b) else min(a, b))

    def max_num(a, b):
        return b if math.isnan(a) else (a if math.isnan(b) else max(a, b))

    def sat(x):
        return 0.0 if math.isnan(x) or x <= 0.0 else (1.0 if x >= 1.0 else x)

    s44, s45, s46, s47 = (split16(w) for w in words)
    v22 = (rne_f16(s44[1] + (-s45[0])), rne_f16(s44[0] + s45[1]))
    v23 = (rne_f16(v22[0] * s46[0]), rne_f16((-v22[1]) * s46[1]))
    v24 = (rne_f16(v23[0] * s47[1] + s47[0]), rne_f16(v23[1] * s47[0] + s47[1]))
    v25 = (rne_f16(min_num(v24[0], math.nan)), rne_f16(min_num(v24[1], 2.0)))
    v26 = (rne_f16(sat(max_num(v24[0], 0.5))), rne_f16(sat(max_num(v24[1], 0.0))))
    return [v25[0], v25[1], v26[0], v26[1]]


def check_vop3p_draw(directory, compiler, renderer, backend):
    source = directory / "vop3p.hlsl"
    shader = directory / "vop3p.cso"
    case = directory / "vop3p.case"
    program = with_constants(VOP3P_BODY)
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for words in VOP3P_SETS:
        expected = vop3p_reference(words)
        assert all(not math.isnan(x) and half_rtz(x) == x for x in expected), expected
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    # Negative mutations against Set 1 and Set 2 must produce pixel_mismatch.
    for words, old, new, label in (
        (VOP3P_SETS[0],
         "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
         "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1]", "neg_lo"),
        (VOP3P_SETS[0],
         "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
         "v_pk_mul_f16 v23, v22, v20", "neg_hi"),
        (VOP3P_SETS[0],
         "v_pk_min_f16 v25, v24, 0x40007e00",
         "v_pk_max_f16 v25, v24, 0x40007e00", "min vs max"),
        (VOP3P_SETS[0],
         "v_pk_max_f16 v26, v24, 0.5 clamp",
         "v_pk_max_f16 v26, v24, 0.5 op_sel_hi:[1,0] clamp", "inline hi zero-extension"),
        (VOP3P_SETS[1],
         "v_pk_max_f16 v26, v24, 0.5 clamp",
         "v_pk_max_f16 v26, v24, 0.5", "clamp"),
    ):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        case.write_bytes(constants + struct.pack("<4f", *vop3p_reference(words)) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


# v_cvt_f16_f32 rounds to nearest even and, on GFX10, writes only the low half of the destination. Channels 0
# and 1 read the whole word: v12 starts as 1.0, so after the conversion it is 1 + h * 2^-23 when the high half
# is kept, and (v12 - 1) * 2^23 splits into the low and high bytes of the binary16 encoding h. Channel 2
# widens -|S0| back to binary32 and channel 3 widens a subnormal result scaled by 2^24. Each set lists wrong
# references (channel, value) for the property it isolates.
NARROW_BODY = [
    "v_mov_b32_e32 v12, 1.0",
    "v_cvt_f16_f32_e32 v12, s44",
    "v_subrev_f32_e32 v12, 1.0, v12",
    "v_mul_f32_e32 v12, 0x4b000000, v12",
    "v_mul_f32_e32 v13, 0x3b800000, v12",
    "v_floor_f32_e32 v13, v13",
    "v_fmac_f32_e32 v12, 0xc3800000, v13",
    "v_cvt_f16_f32_e64 v14, -|s45|",
    "v_cvt_f32_f16_e32 v14, v14",
    "v_cvt_f16_f32_e32 v15, s46",
    "v_cvt_f32_f16_e32 v15, v15",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]
NARROW_SETS = (
    # Ties to even in the normal range and at the smallest subnormal.
    ((1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 1.5 * 2.0 ** -24), ((0, 1.0), (2, -(1 + 2.0 ** -10)), (3, 1.0))),
    # The largest value below the overflow tie, -0, and the tie between zero and 2^-24.
    ((65520 - 2.0 ** -8, -0.0, 2.0 ** -25), ((1, 124.0), (2, 0.0), (3, 1.0))),
    # The overflow tie gives infinity; rounding carries a subnormal into the smallest normal; a negative subnormal.
    ((65520.0, 2.0 ** -14 - 2.0 ** -25, -3 * 2.0 ** -24), ((1, 123.0), (2, -1023 * 2.0 ** -24), (3, 3.0))),
    # A negative subnormal encoding, the negation modifier and a binary32 subnormal input.
    ((-3 * 2.0 ** -24, 3.0, 2.0 ** -149), ((1, 0.0), (2, 3.0))),
)


def narrow_reference(x):
    if abs(x) >= 65520:
        return 0xfc00 if x < 0 else 0x7c00
    return struct.unpack("<H", struct.pack("<e", x))[0]


def check_narrow_draw(directory, compiler, renderer, backend):
    source = directory / "narrow.hlsl"
    shader = directory / "narrow.cso"
    case = directory / "narrow.case"
    source.write_text(translate(with_constants(NARROW_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for inputs, wrongs in NARROW_SETS:
        words = [struct.unpack("<I", struct.pack("<f", x))[0] for x in inputs]
        assert [struct.unpack("<f", struct.pack("<I", w))[0] for w in words] == list(inputs)
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *words)
        h = narrow_reference(inputs[0])
        expected = [float(h & 0xff), float(h >> 8), -abs(half_value(narrow_reference(-abs(inputs[1])))),
                    half_value(narrow_reference(inputs[2])) * 2.0 ** 24]
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            result = subprocess.run(draw, capture_output=True, text=True)
            if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
                raise AssertionError(f"A wrong narrowing escaped detection for {inputs} in channel {channel}")


def arithmetic():
    """Add, multiply, reversed subtraction and moves on the sample and constant words 8 to 11."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[44:47], s[44:47], 0x20",
        "s_waitcnt lgkmcnt(0)",
    ]
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = [
        "v_mov_b32_e32 v18, v12",
        "v_mul_f32_e32 v18, s44, v18",
        "v_add_f32_e64 v12, v18, s45 clamp",
        "v_mov_b32_e64 v19, s46",
        "v_subrev_f32_e32 v19, v13, v19",
        "v_mul_f32_e64 v19, v19, -2.0",
        "v_subrev_f32_e64 v13, s47, v19 clamp",
        "v_add_f32_e32 v14, -0.5, v14",
        "v_mov_b32_e32 v20, 0x3e800000",
        "v_mul_f32_e32 v20, v20, v18",
        "v_mov_b32_e32 v15, v20",
    ]
    return program


def arithmetic_case(directory, *constants):
    """Expected colors for arithmetic(). Each step must be exact in binary32, so neither the
    rounding of add, subtract and multiply nor fusion can change the result."""
    def exact(value):
        rounded, = struct.unpack("<f", struct.pack("<f", value))
        assert rounded == value, value
        return rounded
    scale, bias, pivot, offset = map(exact, constants)
    case = bytearray(256)
    struct.pack_into("<4f", case, 32, scale, bias, pivot, offset)
    ramp = (0, 16, 37, 64, 101, 153, 219, 254)
    pixels = []
    for y in ramp:
        for x in ramp:
            r, g, b = (struct.unpack("<f", struct.pack("<f", c / 255))[0] for c in (x, y, 128))
            scaled = exact(scale * r)
            red = min(1, max(0, exact(scaled + bias)))
            green = min(1, max(0, exact(exact(exact(pivot - g) * -2.0) - offset)))
            color = (red, green, exact(-0.5 + b), exact(0.25 * scaled))
            pixels.append(tuple(map(half_rtz, color)))
    return write_case(directory / "arithmetic.case", case, pixels)


def curve_case(directory, gain):
    gamma, floor, mix = 2.0, .125, .25
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, gamma, gain, floor, mix)
    pixels = []
    for y in RAMP:
        for x in RAMP:
            g = y / 255
            r = gain * math.sqrt(max(floor, x / 255)) - (1 - g) + mix * g
            pixels.append((half_rtz(min(1, max(0, r))), half_rtz(g), half_rtz(128/255), 1))
    return write_case(directory / "curve.case", case, pixels)


def shaping_case(directory, gain):
    scale, floor, mix = 2.0, .125, .25
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, scale, gain, floor, mix)
    pixels = []
    for y in RAMP:
        for x in RAMP:
            g = y / 255
            scaled = scale * g
            r = (gain * math.sqrt(max(floor, x / 255))
                 + 1.0 / math.sqrt(scale + g)
                 + mix * (scaled - math.floor(scaled))
                 - math.floor(scaled))
            pixels.append((half_rtz(min(1, max(0, r))), half_rtz(g), half_rtz(128/255), 1))
    return write_case(directory / "shaping.case", case, pixels)


def truncate_case(directory: Path, bias: float, green: float, blue: float, alpha: float) -> Path:
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, bias, green, blue, alpha)
    # modf returns the integer component as a float and preserves the sign of zero.
    green = math.modf(green)[1]
    blue = math.modf(-abs(blue))[1]
    alpha = max(0.0, min(1.0, math.modf(alpha)[1]))
    pixels = []
    for _ in RAMP:
        for x in RAMP:
            pixels.append((math.modf(bias + x / 255)[1], green, blue, alpha))
    return write_case(directory / "truncate.case", case, pixels)


def sopk_case(directory, red: float, green: float) -> Path:
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, red, green, 0, 1)

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    results = (bits(red) * 3, bits(green) - 2)
    expected = [struct.unpack("<f", struct.pack("<I", value & 0xffffffff))[0]
                for value in results]
    for _ in RAMP:
        for x in RAMP:
            case.extend(struct.pack("<4f", half_rtz(expected[0]), half_rtz(expected[1]), half_rtz(128 / 255), 1))
    path = directory / "sopk.case"
    path.write_bytes(case)
    return path


def rndne(x):
    """Round a binary32 value to the nearest integer, ties to even (Python's round on a float)."""
    return float(round(x))


def rounding_case(directory, scale, tie, bias):
    """Expected colors for rounding(). The scale is a power of two, so each product is exact and
    only the rounding mode decides the result. The constants keep every result away from -0."""
    assert math.frexp(scale)[0] == 0.5, scale
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, scale, tie, bias, 0)
    for y in RAMP:
        for x in RAMP:
            r, g = single(x / 255), single(y / 255)
            color = (math.ceil(scale * r), rndne(scale * g), rndne(single(tie)), math.ceil(-single(bias)))
            case.extend(struct.pack("<4f", *map(half_rtz, map(float, color))))
    path = directory / "rounding.case"
    path.write_bytes(case)
    return path


def absolute_case(directory):
    case = bytearray(256)
    pixels = []
    for y in RAMP:
        for x in RAMP:
            red = abs(0.5 - x / 255)
            pixels.append((half_rtz(red), half_rtz(y / 255), half_rtz(128 / 255), 1))
    return write_case(directory / "absolute.case", case, pixels)


def single(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def select_case(directory, fold, limit, capped, unequal):
    """Expected colors for select(): every step is exact in binary32, so no device tolerance applies."""
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, fold, limit, capped, unequal)
    ramp = (0, 16, 37, 64, 101, 153, 219, 254)
    pixels = []
    for y in ramp:
        for x in ramp:
            r, g = single(x / 255), single(y / 255)
            color = (single(1 - r) if fold < r else r, capped if g >= limit else g,
                     0.0 if r < g else 1.0, 1.0 if r == g else unequal)
            pixels.append(tuple(map(half_rtz, color)))
    return write_case(directory / "select.case", case, pixels)


# v_cube{id,sc,tc,ma}_f32 follow LLVM's ConstantFoldAMDGCNCubeIntrinsic. The direction sets cover all
# six faces, the y/x and z/x ties (the later axis in z, y, x order wins) and -0 on the major axis,
# which selects the positive face. Each set has one wrong reference for the property it isolates.
CUBE_BODY = [
    "v_mov_b32_e32 v18, s45",
    "v_mov_b32_e32 v19, s46",
    "v_cubeid_f32 v12, s44, v18, v19",
    "v_cubesc_f32 v13, s44, v18, v19",
    "v_cubetc_f32 v14, s44, v18, v19",
    "v_cubema_f32 v15, s44, v18, v19",
]
CUBE_SETS = (
    ((2.0, -2.0, 1.0), (0, 1.0)),
    ((-1.5, 0.25, -1.5), (0, 1.0)),
    ((-3.0, 1.0, 2.0), (3, 6.0)),
    ((0.0, 0.0, -0.0), (0, 5.0)),
    ((1.0, 0.5, -0.25), (1, -0.25)),
    ((0.25, 1.0, 0.5), (2, -0.5)),
)


def cube_reference(x, y, z):
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        face, sc, tc, ma = (5.0, -x, -y, z) if z < 0 else (4.0, x, -y, z)
    elif abs(y) >= abs(x):
        face, sc, tc, ma = (3.0, x, -z, y) if y < 0 else (2.0, x, z, y)
    else:
        face, sc, tc, ma = (1.0, z, -y, x) if x < 0 else (0.0, -z, -y, x)
    return [face, sc, tc, ma + ma]


def check_cube_draw(directory, compiler, renderer, backend):
    source = directory / "cube.hlsl"
    shader = directory / "cube.cso"
    case = directory / "cube.case"
    source.write_text(translate(with_constants(CUBE_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    faces = set()
    for direction, (channel, wrong_value) in CUBE_SETS:
        constants = bytearray(256)
        struct.pack_into("<3f", constants, 12, *direction)
        expected = cube_reference(*direction)
        faces.add(expected[0])
        assert all(half_rtz(x) == x for x in (*expected, wrong_value)) and wrong_value != expected[channel]
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        write_case(case, constants, tuple(expected))
        subprocess.run(draw, check=True)
        wrong_values = list(expected)
        wrong_values[channel] = wrong_value
        write_case(case, constants, tuple(wrong_values))
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"A wrong cube result escaped detection for {direction}")
    assert faces == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert math.copysign(1.0, cube_reference(0.0, 0.0, -0.0)[3]) < 0


SCALAR_BFE64_OPS = ("s_bfe_u64", "s_bfe_i64")


def scalar_bfe64(op, src="s[44:45]", control="s46"):
    return uncompressed_export([
        "s_cmp_eq_u32 s47, 0",
        f"{op} s[44:45], {src}, {control}",
        "s_cselect_b32 s48, 1, 0",
        "v_mov_b32_e32 v12, s44",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s48",
        "v_mov_b32_e32 v15, 0",
    ])


def check_scalar_bfe64_draw(directory, compiler, renderer, backend):
    word = 0x81234567_89abcdef
    inputs = (
        (word, 0x000005),  # empty field on nonzero input
        (0, 0x400000),  # zero result from a non-empty field
        (word, 0x010000),
        (word, 0x02001f),  # negative two-bit field spanning the dwords
        (word, 0x200020),
        (1 << 40, 0x3f0001),  # only the high result dword sets SCC
        (word, 0x400000),
        (word, 0x7f003c),  # width beyond the source at a nonzero offset
        (word, 0xff7fffc0),  # ignored control bits, width 127, offset 0
        (word, 0x01003f),
        (0x7f800001, 0x200000),  # preserve raw bits of a signaling NaN
    )
    cases = {}
    for op in SCALAR_BFE64_OPS:
        cases[op] = []
        for i, (value, control) in enumerate(inputs):
            offset = control & 63
            width = min((control >> 16) & 127, 64 - offset)
            field = (value >> offset) & ((1 << width) - 1)
            if op == "s_bfe_i64" and width and (field >> (width - 1)):
                field -= 1 << width
            scc = int(field != 0)
            constants = bytearray(256)
            # With s47=scc, the preceding comparison seeds the opposite condition.
            struct.pack_into("<4I", constants, 12, value & 0xffffffff, value >> 32, control, scc)
            pixel = struct.pack("<4I", field & 0xffffffff, (field >> 32) & 0xffffffff, scc, 0)
            path = directory / f"{op}-{i}.case"
            path.write_bytes(constants + pixel * 64 + b"\x01")  # UINT target, four raw words
            cases[op].append((path, "pass"))

    for op, wrong_op in (SCALAR_BFE64_OPS, SCALAR_BFE64_OPS[::-1]):
        source = directory / f"{op}.hlsl"
        shader = directory / f"{op}.cso"
        source.write_text(translate_int(scalar_bfe64(op)), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong_case, _ = cases[wrong_op][3]
        run_pixel_batch(renderer, backend, shader, [*cases[op], (wrong_case, "mismatch")])


S_CSELECT_B64 = "s_cselect_b64 s[44:45], s[44:45], s[46:47]"


def scalar_select64():
    return uncompressed_export([
        "s_cmp_eq_u32 s44, 0",
        S_CSELECT_B64,
        "s_cselect_b64 s[48:49], 0, -1",
        "v_mov_b32_e32 v12, s44",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s48",
        "v_mov_b32_e32 v15, s49",
    ])


def check_scalar_select64_draw(directory, compiler, renderer, backend):
    source = directory / "scalar-select64.hlsl"
    shader = directory / "scalar-select64.cso"
    program = scalar_select64()
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)

    # Select a zero pair with SCC set and a nonzero pair with SCC clear to
    # distinguish preservation from recomputing SCC from the selected words.
    other = (0x87654321, 0x43218765)
    cases = []
    for i, (lo, hi) in enumerate(((0, 0), (0x89abcdef, 0x12345678), (0, 0x76543210))):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, lo, hi, *other)
        expected = (*other, 0xffffffff, 0xffffffff) if lo else (lo, hi, 0, 0)
        case = directory / f"scalar-select64-{i}.case"
        write_case(case, constants, tuple(as_float(word) for word in expected), integer_rt=True)
        cases.append(case)
    run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases])

    program[program.index(S_CSELECT_B64)] = "s_cselect_b64 s[44:45], s[46:47], s[44:45]"
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(case, "mismatch") for case in cases])


SCALAR64_SETS = (
    # (a_lo, a_hi, b_lo, b_hi, sh_lo, sh_hi, bfm_w, bfm_off, bfe_src, bfe_ctrl_a, bfe_ctrl_b, alt)
    (0x89abcdef, 0x91234567, 0x55aa55aa, 0x7fffffff, 5, 37, 12, 28, 0x9a345f50, 0x00080004, 0x00100018, 0x55aa55aa),
    (0x0000a5a5, 0x0f0f0000, 0xffff5a5a, 0xf0f0ffff, 0, 100, 40, 16, 0x9a345f50, 0x00000000, 0x00000000, 0x00000000),
)


def scalar64():
    """Exercise 64-bit SOP1/SOP2 scalar ALU and 32-bit BFE, packing 40 output bits into exact binary16 floats."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[52:55], s[44:47], 0x10",
        "s_buffer_load_dwordx4 s[56:59], s[44:47], 0x20",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], 0x30",
        "s_waitcnt lgkmcnt(0)",
    ]

    body = [
        "s_mov_b32 s64, 0",
        "s_mov_b32 s65, 0",
        "s_mov_b32 s66, 0",
        "s_mov_b32 s67, 0",
        "s_mov_b32 s78, 0",
        "s_mov_b32 s79, 0",
        "s_mov_b32 s82, s63",
        "s_mov_b32 s83, 0",
    ]

    def capture_scc(bit_index):
        body.extend([
            "s_mov_b32 s70, 0",
            "s_mov_b32 s71, 0",
            f"s_bfm_b64 s[74:75], 1, {bit_index}",
            "s_cmov_b64 s[70:71], s[74:75]",
            "s_or_b64 s[66:67], s[66:67], s[70:71]",
        ])

    def mix_acc(pair):
        body.extend([
            "s_lshl_b64 s[68:69], s[64:65], 3",
            "s_lshr_b64 s[64:65], s[64:65], 61",
            "s_or_b64 s[64:65], s[64:65], s[68:69]",
            f"s_xor_b64 s[64:65], s[64:65], {pair}",
        ])

    # Bit 0: s_and_b64 + two consecutive s_cmov_b64 (inline -1 sign-extended and SCC preserved).
    body.extend([
        "s_and_b64 s[72:73], s[52:53], s[54:55]",
        "s_cmov_b64 s[78:79], -1",
    ])
    capture_scc(0)
    mix_acc("s[72:73]")
    mix_acc("s[78:79]")

    # Bit 1: s_or_b64 (full 64-bit union saved in s[84:85] + SCC probe on s[82:83]).
    body.append("s_or_b64 s[84:85], s[52:53], s[54:55]")
    mix_acc("s[84:85]")
    body.append("s_or_b64 s[72:73], s[82:83], 0")
    capture_scc(1)

    # Bit 2: s_xor_b64 (register pair and zero-extended 32-bit hex literal).
    body.append("s_xor_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_xor_b64 s[72:73], s[82:83], 0x55aa55aa")
    capture_scc(2)
    mix_acc("s[72:73]")

    # Bit 3: s_andn2_b64 (register pair + SCC probe).
    body.append("s_andn2_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_andn2_b64 s[72:73], s[82:83], 0xf0f0f0f")
    capture_scc(3)

    # Bit 4: s_orn2_b64 (register pair + inline -1 proving 64-bit all-ones inversion).
    body.append("s_orn2_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_orn2_b64 s[72:73], s[82:83], -1")
    capture_scc(4)

    # Bit 5: s_not_b64 + s_brev_b64 (s_brev_b64 placed before capture_scc to verify SCC preservation).
    body.append("s_not_b64 s[72:73], s[52:53]")
    mix_acc("s[72:73]")
    body.extend([
        "s_not_b64 s[72:73], s[84:85]",
        "s_brev_b64 s[86:87], s[52:53]",
    ])
    capture_scc(5)
    mix_acc("s[86:87]")

    # Bit 6: s_lshl_b64 across small/zero and >=32/wrapped shift counts.
    body.append("s_lshl_b64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_lshl_b64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.append("s_lshl_b64 s[72:73], s[82:83], s57")
    capture_scc(6)

    # Bit 7: s_lshr_b64 across small/zero and >=32/wrapped shift counts.
    body.append("s_lshr_b64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_lshr_b64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.append("s_lshr_b64 s[72:73], s[82:83], s56")
    capture_scc(7)

    # Bit 8: s_ashr_i64 (signed) + s_bfm_b64 (preserving SCC, plus w=0 and w=32 boundaries).
    body.append("s_ashr_i64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_ashr_i64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.extend([
        "s_ashr_i64 s[72:73], s[82:83], s56",
        "s_bfm_b64 s[86:87], s58, s59",
    ])
    capture_scc(8)
    mix_acc("s[86:87]")
    body.append("s_bfm_b64 s[88:89], 0, 5")
    mix_acc("s[88:89]")
    body.append("s_bfm_b64 s[88:89], 32, 16")
    mix_acc("s[88:89]")

    # Bit 9: s_bfe_u32 (SCC observed via s_cmov_b64 into acc) and s_bfe_i32 (SCC observed in bit 9).
    body.extend([
        "s_mov_b32 s78, 0",
        "s_mov_b32 s79, 0",
        "s_bfe_u32 s72, s60, s61",
        "s_cmov_b64 s[78:79], 0x13572468",
        "s_bfe_u32 s73, s60, s62",
    ])
    mix_acc("s[72:73]")
    mix_acc("s[78:79]")
    body.extend([
        "s_bfe_i32 s72, s60, s61",
        "s_bfe_i32 s73, s60, s62",
    ])
    capture_scc(9)
    mix_acc("s[72:73]")

    # s_nand_b64, s_nor_b64 and s_xnor_b64: the register pair, then an SCC probe. Channel 3 exports
    # only SCC bits 0-9, so each probe's SCC reaches acc through s_cmov_b64 of a distinct literal.
    # nand(-1, -1) and nor(alt, -1) are zero, and xnor(alt, -1) = alt is zero only in Set 2.
    for op, probe, marker in (("s_nand_b64", "-1, -1", "0x2468ace0"),
                              ("s_nor_b64", "s[82:83], -1", "0x13579bdf"),
                              ("s_xnor_b64", "s[82:83], -1", "0x0fedcba9")):
        body.append(f"{op} s[72:73], s[52:53], s[54:55]")
        mix_acc("s[72:73]")
        body.extend([
            "s_mov_b32 s78, 0",
            "s_mov_b32 s79, 0",
            f"{op} s[72:73], {probe}",
            f"s_cmov_b64 s[78:79], {marker}",
        ])
        mix_acc("s[72:73]")
        mix_acc("s[78:79]")

    # Fold all 64 bits of acc into bits [29:0] of s64: acc ^ (acc >> 30) ^ (acc >> 60), both shifts of the
    # original acc, so every bit reaches the export. Repeating x ^= x >> 30 instead gives x ^ (x >> 60)
    # and drops bits 30-59. Then export 4 x 10-bit fields as 1 + k/1024.
    body.extend([
        "s_lshr_b64 s[68:69], s[64:65], 30",
        "s_lshr_b64 s[70:71], s[64:65], 60",
        "s_xor_b64 s[64:65], s[64:65], s[68:69]",
        "s_xor_b64 s[64:65], s[64:65], s[70:71]",
    ])
    for channel, (src, ctrl) in enumerate((("s64", "0xa0000"), ("s64", "0xa000a"),
                                           ("s64", "0xa0014"), ("s66", "0xa0000"))):
        body.extend([
            f"s_bfe_u32 s80, {src}, {ctrl}",
            "s_mov_b32 s81, 0",
            "s_lshl_b64 s[80:81], s[80:81], 13",
            "s_or_b64 s[80:81], s[80:81], 0x3f800000",
            f"v_mov_b32_e32 v{12 + channel}, s80",
        ])

    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = body
    return program


def scalar64_reference(program, words):
    """Independent Python ISA reference for scalar64() using wide integers."""
    mask32, mask64 = 0xffffffff, 0xffffffffffffffff
    regs = {}
    scc = None
    channels = {}

    def read32(tok):
        if tok.startswith("s"):
            return regs[tok] & mask32
        return int(tok, 0) & mask32

    def read64(tok, signed=False):
        if tok.startswith("s["):
            lo, hi = map(int, tok[2:-1].split(":"))
            return (regs[f"s{lo}"] & mask32) | ((regs[f"s{hi}"] & mask32) << 32)
        if tok.startswith("0x"):
            v = int(tok, 16)
            return v | (0xffffffff00000000 if signed and (v & 0x80000000) else 0)
        return int(tok, 0) & mask64

    def write64(tok, val):
        lo, hi = map(int, tok[2:-1].split(":"))
        val &= mask64
        regs[f"s{lo}"] = val & mask32
        regs[f"s{hi}"] = (val >> 32) & mask32

    for line in program:
        op, _, rest = line.partition(" ")
        ops = [t.strip() for t in rest.split(",")] if rest else []
        if op == "s_buffer_load_dwordx4":
            lo = int(ops[0][2:-1].split(":")[0])
            start = (0 if ops[2] == "null" else int(ops[2], 16)) // 4
            for j in range(4):
                regs[f"s{lo + j}"] = words[start + j]
        elif op == "s_mov_b32" and ops[0].startswith("s") and not ops[1].startswith("s2"):
            regs[ops[0]] = read32(ops[1])
        elif op == "s_and_b64" and ops[0].startswith("s["):
            v = read64(ops[1]) & read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_or_b64":
            v = read64(ops[1]) | read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_xor_b64":
            v = read64(ops[1]) ^ read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_andn2_b64":
            v = read64(ops[1]) & (~read64(ops[2]) & mask64)
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_orn2_b64":
            v = read64(ops[1]) | (~read64(ops[2]) & mask64)
            write64(ops[0], v)
            scc = v != 0
        elif op in ("s_nand_b64", "s_nor_b64", "s_xnor_b64"):
            a, b = read64(ops[1]), read64(ops[2])
            v = ~{"s_nand_b64": a & b, "s_nor_b64": a | b, "s_xnor_b64": a ^ b}[op] & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_not_b64":
            v = ~read64(ops[1]) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_brev_b64":
            x = read64(ops[1])
            v = int(f"{x:064b}"[::-1], 2)
            write64(ops[0], v)
        elif op == "s_cmov_b64":
            assert scc is not None
            if scc:
                write64(ops[0], read64(ops[1]))
        elif op == "s_lshl_b64":
            v = (read64(ops[1]) << (read32(ops[2]) & 63)) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_lshr_b64":
            v = read64(ops[1]) >> (read32(ops[2]) & 63)
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_ashr_i64":
            x = read64(ops[1], signed=True)
            if x >= (1 << 63):
                x -= 1 << 64
            v = (x >> (read32(ops[2]) & 63)) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_bfm_b64":
            w = read32(ops[1]) & 63
            off = read32(ops[2]) & 63
            v = (((1 << w) - 1) << off) & mask64
            write64(ops[0], v)
        elif op in ("s_bfe_u32", "s_bfe_i32"):
            src = read32(ops[1])
            ctrl = read32(ops[2])
            off = ctrl & 31
            w = (ctrl >> 16) & 127
            if w == 0:
                v = 0
            elif op == "s_bfe_u32":
                v = (src >> off) & ((1 << min(w, 32)) - 1)
            else:
                if off + w < 32:
                    raw = (src >> off) & ((1 << w) - 1)
                    if raw & (1 << (w - 1)):
                        raw -= 1 << w
                else:
                    raw = (src - (1 << 32) if src & 0x80000000 else src) >> off
                v = raw & mask32
            regs[ops[0]] = v
            scc = v != 0
        elif op == "v_mov_b32_e32" and ops[0] in ("v12", "v13", "v14", "v15"):
            bits = regs[ops[1]] & mask32
            val, = struct.unpack("<f", struct.pack("<I", bits))
            assert half_rtz(val) == val, hex(bits)
            channels[ops[0]] = val
    return tuple(channels[f"v{12 + c}"] for c in range(4)), regs["s66"] & 0x3ff


def scalar64_case(directory, values):
    constants = bytearray(256)
    struct.pack_into("<12I", constants, 16, *values)
    words = struct.unpack("<64I", constants)
    color, _ = scalar64_reference(scalar64(), words)
    return write_case(directory / "scalar64.case", constants, tuple(color))
# Scalar ALU producers in the scalar() fixture. Each SCC result sets its own bit of s60 through
# s_cselect_b32; each result value is mixed into the hash in s61. a to d are constant words 3 to 6.
SCALAR_STEPS = (
    ("s_add_u32 s62, s44, s45", True), ("s_sub_u32 s62, s46, s47", True),
    ("s_add_i32 s62, s46, s45", True), ("s_sub_i32 s62, s46, s45", True),
    ("s_min_u32 s62, s45, s44", True), ("s_min_i32 s62, s44, s45", True),
    ("s_max_u32 s62, s44, s45", True), ("s_max_i32 s62, s45, s44", True),
    ("s_and_b32 s62, s44, s47", True), ("s_xor_b32 s62, s44, s45", True),
    ("s_lshr_b32 s62, s46, 31", True),
    ("s_cmp_eq_u32 s44, s45", False), ("s_cmp_lg_u32 s44, s45", False),
    ("s_cmp_gt_i32 s45, s44", False), ("s_cmp_ge_i32 s44, s45", False),
    ("s_cmp_lt_i32 s44, s45", False), ("s_cmp_le_i32 s45, s44", False),
    ("s_cmp_gt_u32 s44, s45", False), ("s_cmp_ge_u32 s45, s44", False),
    ("s_cmp_lt_u32 s45, s44", False), ("s_cmp_le_u32 s44, s45", False),
    ("s_cmp_eq_i32 s47, -1", False), ("s_cmp_lg_i32 s46, 0x80000000", False),
)
# Results that leave SCC unused or unchanged; s_cselect_b32 here selects between two registers.
SCALAR_VALUES = ("s_mul_i32 s62, s44, s46", "s_or_b32 s62, s45, s46", "s_andn2_b32 s62, s44, s45",
                 "s_orn2_b32 s62, s47, s44", "s_not_b32 s62, s46", "s_lshl_b32 s62, s44, s47",
                 "s_ashr_i32 s62, s46, s44", "s_nand_b32 s62, s44, s45", "s_nor_b32 s62, s45, s46",
                 "s_xnor_b32 s62, s46, s47", "s_cmp_lt_u32 s44, s45", "s_cselect_b32 s62, s46, s47")
# Two constant sets; every SCC bit differs between them, so both arms of each select are drawn.
SCALAR_WORDS = ((0xfffffff0, 0x20, 0x80000000, 1), (5, 5, 0x7fffffff, 0xffffffff))


def scalar_body():
    body = ["s_mov_b32 s60, 0", "s_movk_i32 s61, 0xff80"]
    for bit, (step, value) in enumerate(SCALAR_STEPS):
        flag = 1 << bit
        body += [step, f"s_cselect_b32 s63, {flag if flag <= 64 else hex(flag)}, 0", "s_or_b32 s60, s60, s63"]
        if value:
            body += ["s_mul_i32 s61, s61, 31", "s_add_u32 s61, s61, s62"]
    for step in SCALAR_VALUES:
        body.append(step)
        if step.startswith("s_cselect") or not step.startswith("s_cmp"):
            body += ["s_mul_i32 s61, s61, 31", "s_add_u32 s61, s61, s62"]
    # Fold the 32-bit hash to 17 bits and spread 23 flag bits and 17 hash bits over four channels,
    # ten bits each, exported as 1 + k / 1024, which binary16 holds exactly.
    body += ["s_lshr_b32 s63, s61, 17", "s_xor_b32 s61, s61, s63", "s_and_b32 s61, s61, 0x1ffff",
             "s_and_b32 s56, s60, 0x3ff",
             "s_lshr_b32 s57, s60, 10", "s_and_b32 s57, s57, 0x3ff",
             "s_lshr_b32 s58, s60, 20", "s_lshl_b32 s63, s61, 3", "s_or_b32 s58, s58, s63",
             "s_and_b32 s58, s58, 0x3ff",
             "s_lshr_b32 s59, s61, 7"]
    for channel in range(4):
        body += [f"s_lshl_b32 s{56 + channel}, s{56 + channel}, 13",
                 f"s_or_b32 s{56 + channel}, s{56 + channel}, 1.0",
                 f"v_mov_b32_e32 v{12 + channel}, s{56 + channel}"]
    return body


PACK_HALVES = ("ll", "lh", "hh")


def scalar_carry_chain(subtract, *, initialize_scc=True):
    low, high = ("s_sub_u32", "s_subb_u32") if subtract else ("s_add_u32", "s_addc_u32")
    setup = [f"{low} s62, s44, s46", "s_cselect_b32 s63, 1, 0"] if initialize_scc else [
        "s_mov_b32 s62, 0", "s_mov_b32 s63, 0",
    ]
    return uncompressed_export([
        *setup,
        f"{high} s45, s45, s47", "s_cselect_b32 s64, 1, 0",
        "v_mov_b32_e32 v12, s62", "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s63", "v_mov_b32_e32 v15, s64",
    ])


def check_scalar_carry_chain_draw(directory, compiler, renderer, backend):
    source, shader = (directory / name for name in ("carry-chain.hlsl", "carry-chain.cso"))
    for subtract in (False, True):
        program = scalar_carry_chain(subtract)
        source.write_text(translate_int(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        pairs = ((0, 0), (0x100000000, 1), (0, 0x100000000), (0, 1),
                 (0, 0xffffffff00000001), (0xffffffff00000000, 0xffffffff00000001)) if subtract else (
                     (0, 0), (0xffffffff, 1), (0xffffffff00000000, 0x100000000),
                     (0xffffffffffffffff, 1))
        cases = []
        for index, (a, b) in enumerate(pairs):
            low_a, low_b = a & 0xffffffff, b & 0xffffffff
            low = low_a - low_b if subtract else low_a + low_b
            wide = a - b if subtract else a + b
            words = (wide & 0xffffffff, (wide >> 32) & 0xffffffff,
                     not 0 <= low < 1 << 32, not 0 <= wide < 1 << 64)
            constants = bytearray(256)
            struct.pack_into("<QQ", constants, 12, a, b)
            # Every case keeps its own file; the last one is also the mutation control below.
            case = directory / f"carry-chain-{int(subtract)}-{index}.case"
            write_case(case, constants, tuple(as_float(w) for w in words), integer_rt=True)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        # The final case wraps only because of the incoming SCC.
        old, new = ("s_subb_u32", "s_sub_u32") if subtract else ("s_addc_u32", "s_add_u32")
        source.write_text(translate_int([line.replace(old, new) for line in program]), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Dropping SCC from {old} escaped detection")


def scalar_leading_bit_values(width):
    ops = ("s_flbit_i32_b32", "s_flbit_i32") if width == 32 else ("s_flbit_i32_b64", "s_flbit_i32_i64")
    src = "s44" if width == 32 else "s[44:45]"
    body = []
    for i, op in enumerate(ops):
        body.extend([
            "s_cmp_eq_u32 s44, s45", f"{op} s62, {src}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{14 + i}, s63",
            f"s_cmp_eq_u32 s62, s{46 + i}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{12 + i}, s63",
        ])
    return with_constants(body)


def check_scalar_leading_bit_draw(directory, compiler, renderer, backend):
    source, shader = (directory / name for name in ("leading-bit.hlsl", "leading-bit.cso"))
    for width in (32, 64):
        source.write_text(translate(scalar_leading_bit_values(width)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        cases = []
        for index, (lo, hi) in enumerate(((0, 0), (0xffffffff, 0xffffffff), (0x80000000, 0x80000000),
                                          (0, 1), (0x80000000, 0), (1, 0), (0xfffffffe, 0xffffffff))):
            word = lo if width == 32 else lo | (hi << 32)
            bits = f"{word:0{width}b}"
            unsigned = bits.find("1")
            signed = bits.find("0" if bits[0] == "1" else "1")
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, lo, hi, unsigned & 0xffffffff, signed & 0xffffffff)
            case = directory / f"leading-bit-{width}-{index}.case"
            case.write_bytes(constants + struct.pack("<4f", 1, 1, lo == hi, lo == hi) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    # For -2, the signed search must cross into the low word; the unsigned search returns zero.
    program = [line.replace("s_flbit_i32_i64", "s_flbit_i32_b64") for line in scalar_leading_bit_values(64)]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Unsigned leading-bit scan matched the signed reference")


def scalar_bit_count_values(width):
    src = "s44" if width == 32 else "s[44:45]"
    body = []
    for bit in (0, 1):
        # Seed SCC from a runtime word. The reference always seeds the opposite of the expected SCC.
        body.extend([
            f"s_cmp_eq_u32 s{46 + bit}, 1",
            f"s_bcnt{bit}_i32_b{width} s62, {src}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{14 + bit}, s63",
            "s_lshl_b32 s62, s62, 13", "s_or_b32 s62, s62, 0x3f800000",
            f"v_mov_b32_e32 v{12 + bit}, s62",
        ])
    return with_constants(body)


def check_scalar_bit_count_draw(directory, compiler, renderer, backend):
    source, shader = (directory / name for name in ("bit-count.hlsl", "bit-count.cso"))
    for width in (32, 64):
        program = scalar_bit_count_values(width)
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        cases = []
        for index, (lo, hi) in enumerate(((0, 0), (0xffffffff, 0xffffffff), (0x1234abcd, 0x5678ef90),
                                          (0, 0x80000000))):
            word = lo if width == 32 else lo | (hi << 32)
            ones = word.bit_count()
            zeros = width - ones
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, lo, hi, zeros == 0, ones == 0)
            case = directory / f"bit-count-{width}-{index}.case"
            case.write_bytes(constants + struct.pack("<4f", 1 + zeros / 1024, 1 + ones / 1024,
                                                      zeros != 0, ones != 0) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    # The last input has one bit set in the high word and none in the low word.
    program = [line.replace("s_bcnt1_i32_b64 s62, s[44:45]", "s_bcnt1_i32_b32 s62, s44")
               for line in scalar_bit_count_values(64)]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Dropping the high word matched the original reference")


def scalar_mul_hi_values(signed="i32"):
    return with_constants([
        "s_cmp_eq_u32 s44, s45",
        f"s_mul_hi_{signed} s62, s44, s45",
        "s_mul_hi_u32 s63, s44, s45",
        "s_cselect_b32 s64, 0x3f800000, 0",
        "v_mov_b32_e32 v14, s64",
        "s_cmp_eq_u32 s62, s46", "s_cselect_b32 s64, 0x3f800000, 0",
        "v_mov_b32_e32 v12, s64",
        "s_cmp_eq_u32 s63, s47", "s_cselect_b32 s64, 0x3f800000, 0",
        "v_mov_b32_e32 v13, s64", "v_mov_b32_e32 v15, 0",
    ])


def check_scalar_mul_hi_draw(directory, compiler, renderer, backend):
    source, shader = (directory / name for name in ("mul-hi.hlsl", "mul-hi.cso"))
    source.write_text(translate(scalar_mul_hi_values()), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    cases = []
    for index, (a, b) in enumerate(((0, 0x80000000), (0x7fffffff, 0x7fffffff), (0x80000000, 3),
                                    (3, 0x80000000), (0x1234abcd, 0x5678ef90), (0xffffffff, 0xffffffff))):
        signed_a, signed_b = struct.unpack("<ii", struct.pack("<II", a, b))
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, a, b, (signed_a * signed_b >> 32) & 0xffffffff, a * b >> 32)
        case = directory / f"mul-hi-{index}.case"
        case.write_bytes(constants + struct.pack("<4f", 1, 1, a == b, 0) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    # For the final inputs, signed high is zero and unsigned high is 0xfffffffe.
    source.write_text(translate(scalar_mul_hi_values("u32")), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Unsigned high product matched the signed reference")


def scalar_pack_values(halves):
    return with_constants([
        "s_cmp_eq_u32 s47, 1", f"s_pack_{halves}_b32_b16 s44, s44, s45",
        "s_cselect_b32 s62, 0x3f800000, 0", "v_mov_b32_e32 v13, s62",
        "s_cmp_eq_u32 s44, s46", "s_cselect_b32 s62, 0x3f800000, 0",
        "v_mov_b32_e32 v12, s62", "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0",
    ])


def check_scalar_pack_draw(directory, compiler, renderer, backend):
    source, shader = (directory / name for name in ("pack.hlsl", "pack.cso"))
    for halves in PACK_HALVES:
        source.write_text(translate(scalar_pack_values(halves)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        # The third pair sets every bit of each half LL and LH select, so a mask that drops any one bit
        # of a selected half (0xfffd for the low half, 0xfffe0000 for the high half) breaks the equality.
        cases = []
        for index, words in enumerate(((0x1234abcd, 0x5678ef90), (0xffff0000, 0x0000ffff),
                                       (0x0000ffff, 0xffffffff))):
            flag = index % 2
            selected = [struct.unpack("<HH", struct.pack("<I", word))[half == "h"]
                        for word, half in zip(words, halves)]
            constants = bytearray(256)
            struct.pack_into("<IIHHI", constants, 12, *words, *selected, flag)
            case = directory / f"pack-{halves}-{index}.case"
            case.write_bytes(constants + struct.pack("<4f", 1, flag, 0, 0) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
        if halves == "lh":
            # Compile LL against the unchanged first LH reference; only the selected high half differs.
            source.write_text(translate(scalar_pack_values("ll")), encoding="ascii")
            mutant = directory / "wrong-pack.cso"
            compile_shader(compiler, source, mutant, ieee=False)
            wrong = subprocess.run([renderer, backend, "--check-pixel-case", mutant, cases[0][0]],
                                   capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("LL matched the LH reference")


def sext_values():
    body = []
    for index, width in enumerate((8, 16)):
        body.extend(["s_cmp_eq_u32 s47, 1", f"s_sext_i32_i{width} s62, s44",
                     "s_cselect_b32 s63, 0x3f800000, 0",
                     f"v_mov_b32_e32 v{14 + index}, s63",
                     f"s_cmp_eq_u32 s62, s{45 + index}", "s_cselect_b32 s63, 0x3f800000, 0",
                     f"v_mov_b32_e32 v{12 + index}, s63"])
    return with_constants(body)


def check_sext_draw(directory, compiler, renderer, backend):
    source = directory / "sext.hlsl"
    shader = directory / "sext.cso"
    case = directory / "sext.case"
    source.write_text(translate(sext_values()), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for index, word in enumerate((0xabcd807f, 0xabcd7f80, 0xabcd0000, 0x1234ffff)):
        flag = index % 2
        raw = word.to_bytes(4, "little")
        expected = [int.from_bytes(raw[:size], "little", signed=True) for size in (1, 2)]
        constants = bytearray(256)
        struct.pack_into("<IiiI", constants, 12, word, *expected, flag)
        case.write_bytes(constants + struct.pack("<4f", 1, 1, flag, flag) * 64)
        subprocess.run(draw, check=True)
    # Replacing both signed references with their zero-extended values must change both channels.
    struct.pack_into("<II", constants, 16, 255, 65535)
    case.write_bytes(constants + struct.pack("<4f", 0, 0, flag, flag) * 64)
    subprocess.run(draw, check=True)


SOP1_WORDS = (
    (0, 0x2468ace1, 0x12400009, 3),
    (0xfffffffb, 0x2468ace1, 0x12400009, 35),
    (5, 0x11223344, 0x81400008, 0),
    (0x80000000, 0x55667788, 0x81400009, 31),
)


def sop1_scalar_ops() -> list[str]:
    return uncompressed_export([
        "s_abs_i32 s50, s44",
        "s_mov_b32 s51, s46",
        "s_cmov_b32 s51, s45",
        "s_mov_b32 s52, s46",
        "s_bitset0_b32 s52, s47",
        "s_bitset1_b32 s52, 31",
        "s_mov_b32 s53, s46",
        "s_cmov_b32 s53, 0x12345678",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
        "v_mov_b32_e32 v14, s52",
        "v_mov_b32_e32 v15, s53",
    ])


def sop1_scalar_ops_reference(words, set_bit=31):
    s44, s45, s46, s47 = words
    signed44 = s44 - (1 << 32) if s44 >> 31 else s44
    abs44 = (-signed44 if signed44 < 0 else signed44) & 0xffffffff
    scc = abs44 != 0
    cmov_reg = s45 if scc else s46
    bitset = (s46 & ~(1 << (s47 & 31))) | (1 << set_bit)
    cmov_lit = 0x12345678 if scc else s46
    return (abs44, cmov_reg, bitset & 0xffffffff, cmov_lit)


# s_bfm_b32/s_bfm_b64: count words s44 (S0) and s45 (S1 offset) feed both widths; s46/s47 give a second 32-bit mask.
BFM_WORDS = (
    (0, 0, 1, 31),
    (31, 1, 32, 32),
    (32, 31, 33, 0xffffffff),
    (63, 33, 64, 63),
    (64, 5, 0xffffffe1, 0xffffffc3),
)


def bfm_scalar_ops() -> list[str]:
    return uncompressed_export([
        "s_bfm_b32 s50, s44, s45",
        "s_bfm_b32 s51, s46, s47",
        "s_bfm_b64 s[52:53], s44, s45",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
        "v_mov_b32_e32 v14, s52",
        "v_mov_b32_e32 v15, s53",
    ])


def bfm_scalar_ops_reference(words):
    """Counts are masked to 5 (b32) or 6 (b64) bits; the mask is (1 << count) - 1, shifted left by the offset."""
    s44, s45, s46, s47 = words
    b32 = lambda count, offset: (((1 << (count & 31)) - 1) << (offset & 31)) & 0xffffffff
    b64 = (((1 << (s44 & 63)) - 1) << (s45 & 63)) & 0xffffffffffffffff
    return (b32(s44, s45), b32(s46, s47), b64 & 0xffffffff, b64 >> 32)


def scalar():
    return with_constants(scalar_body())


def scalar_reference(body, words):
    """Run body on Python integers; returns the four exported channels and every select's arm."""
    mask = 0xffffffff
    regs = dict(zip(("s44", "s45", "s46", "s47"), words))
    scc, arms, out = None, [], {}

    def signed(x):
        return x - (1 << 32) if x >> 31 else x

    def read(token):
        if token in regs:
            return regs[token]
        if token == "1.0":
            return 0x3f800000
        return int(token, 0) & mask

    for line in body:
        op, operands = line.split(" ", 1)
        args = operands.split(", ")
        if op.startswith("v_mov_b32"):
            out[args[0]] = regs[args[1]]
            continue
        x, *rest = map(read, args[1:] if not op.startswith("s_cmp") else args)
        y = rest[0] if rest else None
        if op.startswith("s_cmp_"):
            relation, kind = op[6:8], op[9]
            left, right = (signed(x), signed(y)) if kind == "i" else (x, y)
            scc = {"eq": left == right, "lg": left != right, "gt": left > right, "ge": left >= right,
                   "lt": left < right, "le": left <= right}[relation]
            continue
        name = op[2:]
        if name == "mov_b32":
            result = x
        elif name == "movk_i32":
            result = (x - 0x10000 if x & 0x8000 else x) & mask
        elif name == "cselect_b32":
            assert scc is not None
            arms.append(scc)
            result = x if scc else y
        elif name in ("add_u32", "sub_u32"):
            wide = x + y if name == "add_u32" else x - y
            result, scc = wide & mask, not 0 <= wide <= mask
        elif name in ("add_i32", "sub_i32"):
            wide = signed(x) + signed(y) if name == "add_i32" else signed(x) - signed(y)
            result, scc = wide & mask, not -2**31 <= wide < 2**31
        elif name == "mul_i32":
            result = (x * y) & mask
        elif name in ("min_u32", "max_u32", "min_i32", "max_i32"):
            key = signed if name.endswith("i32") else (lambda v: v)
            scc = key(x) < key(y) if name.startswith("min") else key(x) > key(y)
            result = x if scc else y
        elif name == "not_b32":
            result = ~x & mask
            scc = result != 0
        else:
            result = {"and_b32": lambda: x & y, "or_b32": lambda: x | y, "xor_b32": lambda: x ^ y,
                      "andn2_b32": lambda: x & ~y, "orn2_b32": lambda: x | ~y,
                      "nand_b32": lambda: ~(x & y), "nor_b32": lambda: ~(x | y), "xnor_b32": lambda: ~(x ^ y),
                      "lshl_b32": lambda: x << (y & 31), "lshr_b32": lambda: x >> (y & 31),
                      "ashr_i32": lambda: signed(x) >> (y & 31)}[name]() & mask
            scc = result != 0
        regs[args[0]] = result
    channels = []
    for n in range(12, 16):
        value = struct.unpack("<f", struct.pack("<I", out[f"v{n}"]))[0]
        assert 1 <= value < 2 and half_rtz(value) == value, value
        channels.append(value)
    return channels, arms


def scalar_case(directory, words):
    """Constant words 3 to 6 as raw 32-bit patterns, then the uniform color expected at every texel."""
    case = bytearray(256)
    struct.pack_into("<4I", case, 12, *words)
    channels, _ = scalar_reference(scalar_body(), words)
    case.extend(struct.pack("<4f", *channels) * 64)
    path = directory / "scalar.case"
    path.write_bytes(case)
    return path


# Mutations a draw must detect: signedness of a shift and of a comparison, and the sources of a subtraction.
SCALAR_MUTATIONS = (("s_ashr_i32", "s_lshr_b32"), ("s_cmp_lt_i32", "s_cmp_lt_u32"),
                    ("s_sub_u32 s62, s46, s47", "s_sub_u32 s62, s47, s46"),
                    ("s_nand_b32 s62, s44, s45", "s_and_b32 s62, s44, s45"),
                    ("s_nor_b32 s62, s45, s46", "s_or_b32 s62, s45, s46"),
                    ("s_xnor_b32 s62, s46, s47", "s_xor_b32 s62, s46, s47"))


def scalar_detecting_words(old, new):
    """The first constant set whose reference changes under the mutation."""
    body = scalar_body()
    changed = [line.replace(old, new) for line in body]
    assert changed != body, old
    return next(words for words in SCALAR_WORDS
                if scalar_reference(changed, words)[0] != scalar_reference(body, words)[0])


def check_scalar_controls():
    body = scalar_body()
    program = scalar()
    source = translate(program)
    for expected in ("= asuint(params[0].w) + asuint(params[1].x);", " < asuint(params[0].w)) ? 1u : 0u;",
                     "= (asuint(params[1].y) < asuint(params[1].z)) ? 2u : 0u;",
                     "= ((~(asuint(params[1].y) ^ asuint(params[1].x)) & (asuint(params[1].y) ^ salu",
                     "= asuint(min(asint(asuint(params[0].w)), asint(asuint(params[1].x))));",
                     "= (asint(asuint(params[1].x)) > asint(asuint(params[0].w)));", "? 8192u : 0u;",
                     "= (asint(asuint(params[1].z)) == asint(4294967295u));", "? 2097152u : 0u;",
                     "= asuint(params[0].w) & ~asuint(params[1].x);", "= ~asuint(params[1].y);",
                     "= asuint(asint(asuint(params[1].y)) >> (asuint(params[0].w) & 31u));",
                     "= ~(asuint(params[0].w) & asuint(params[1].x));",
                     "= ~(asuint(params[1].x) | asuint(params[1].y));",
                     "= ~(asuint(params[1].y) ^ asuint(params[1].z));",
                     "= 4294967168u * 31u;", "? asuint(params[1].y) : asuint(params[1].z);",
                     " | 1065353216u;", "= asfloat(salu"):
        assert expected in source, expected
    # Both arms of every flag select are drawn across the two constant sets.
    flags = [scalar_reference(body, words)[1] for words in SCALAR_WORDS]
    assert len(flags[0]) == len(SCALAR_STEPS) + 1
    assert all(a != b for a, b in zip(flags[0][:-1], flags[1][:-1])), flags
    for old, new in SCALAR_MUTATIONS:
        scalar_detecting_words(old, new)
    # SCC exists only after a tracked write; the exec-mask operations overwrite it.
    base = assembly(False)
    wqm = base.index("s_wqm_b64 exec, exec")
    andexec = base.index("s_and_b64 exec, exec, s[50:51]")
    translate(base[:andexec] + ["s_cmp_eq_u32 1, 1", "s_cselect_b32 s62, 1, 0"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cselect_b32 s62, 1, 0"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_cselect_b32 s62, 1, 0"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_cselect_b32 s62, 1, 0"] + base[wqm + 1:])
    # A scalar result is not a known constant, so it cannot supply a sampler word or address.
    for old, new in (("s_mov_b32 s40, 1", "s_add_u32 s40, 0, 1"), ("s_mov_b32 s21, 0", "s_and_b32 s21, 0, 0")):
        must_reject(translate, [line.replace(old, new) for line in base])
    # Neighbouring forms that the lowerer does not implement, operand kinds it cannot read, and
    # destinations other than plain scalar registers.
    first = "s_add_u32 s62, s44, s45"
    for subtract in (False, True):
        must_reject(translate_int, scalar_carry_chain(subtract, initialize_scc=False))
    for new in ("s_absdiff_i32 s62, s44, s45",
                "s_nand_b32 s62, v12, s45", "s_nor_b32 v62, s44, s45", "s_xnor_b32 s62, s44",
                "s_bfe_u32 s62, v12, s45", "s_lshl5_add_u32 s62, s44, s45",
                "s_quadmask_b32 s62, s44", "s_bitset0_b64 s[62:63], s44",
                "s_brev_b32 s62, s44", "s_and_b64 s[62:63], s[44:46], s[46:47]",
                "s_add_u32 s62, s[44:45], s45", "s_add_u32 s62, s44", "s_add_u32 s62, s44, s45, s46",
                "s_add_u32 s62, -s44, s45", "s_add_u32 s62, s28, s45", "s_add_u32 s62, s20, s45",
                "s_add_u32 s62, s90, s45", "s_add_u32 s62, vcc_lo, s45", "s_add_u32 s62, exec_lo, s45",
                "s_add_u32 s62, v12, s45", "s_add_u32 m0, s44, s45", "s_add_u32 vcc_lo, s44, s45",
                "s_add_u32 s62, s44, |s45|",
                "s_cmp_eq_u64 s[44:46], s[46:47]", "s_bitcmp1_b32 v12, s45",
                "s_cmpk_eq_u32 s44, 0x10000", "s_cmpk_eq_u32 s44, -1", "s_cmpk_eq_f32 s44, 0x10",
                "s_cmp_lt_i32 s44", "s_cmp_lt_f32 s44, s45"):
        changed = [line.replace(first, new) for line in program]
        assert changed != program, new
        must_reject(translate, changed)
    flag = program.index(first) + 1
    for new in ("s_cselect_b64 s[62:63], s44, s45", "s_cmovk_i32 s63, 0x10000", "s_cmovk_i32 vcc_lo, 0x1",
                "s_cselect_b32 s63, s28, 0", "s_cselect_b32 vcc_lo, 1, 0"):
        must_reject(translate, program[:flag] + [new] + program[flag + 1:])
    # Before lgkmcnt(0) the constant words are still loading: neither read nor overwrite them.
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
    for line in ("s_add_u32 s62, s44, 1", "s_add_u32 s45, 1, 1", "s_cmp_eq_u32 s46, 0",
                 "s_cselect_b32 s47, 1, 0"):
        must_reject(translate, program[:load] + ["s_cmp_eq_u32 1, 1", line] + program[load:])
    translate(program[:load] + ["s_cmp_eq_u32 1, 1", "s_cselect_b32 s62, 1, 0"] + program[load:])

    sop1_program = sop1_scalar_ops()
    sop1_source = translate_int(sop1_program)
    for expected in ("(asint(asuint(params[0].w)) < 0 ? (0u - asuint(params[0].w)) : asuint(params[0].w))",
                     "& ~(1u << (", "| (1u << (31u & 31u))", "? 305419896u :"):
        assert expected in sop1_source, expected
    sop1_refs = [sop1_scalar_ops_reference(words) for words in SOP1_WORDS]
    assert any(r[0] == 0 for r in sop1_refs) and any(r[0] == 0x80000000 for r in sop1_refs)
    assert any(r[3] == 0x12400009 for r in sop1_refs) and any(r[3] == 0x12345678 for r in sop1_refs)
    must_reject(translate, base[:andexec] + ["s_mov_b32 s51, 0", "s_cmov_b32 s51, s40"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_mov_b32 s51, 0", "s_cmov_b32 s51, s40"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_mov_b32 s51, 0", "s_cmov_b32 s51, s40"] + base[wqm + 1:])
    for uninit in ("s_cmov_b32 s60, s44", "s_bitset0_b32 s60, 1", "s_bitset1_b32 s60, 1"):
        must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1", uninit] + base[andexec:])
    sop1_wait = sop1_program.index("s_waitcnt lgkmcnt(0)", sop1_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    for bad in ("s_abs_i32 s50, s44", "s_cmov_b32 s44, 1", "s_bitset0_b32 s44, 1", "s_bitset1_b32 s44, 1"):
        must_reject(translate_int, sop1_program[:sop1_wait] + ["s_cmp_eq_u32 1, 1", bad] + sop1_program[sop1_wait:])
    for old, new in (("s_abs_i32 s50, s44", "s_abs_i32 v12, s44"),
                     ("s_abs_i32 s50, s44", "s_abs_i32 s106, s44"),
                     ("s_cmov_b32 s51, s45", "s_cmov_b32 vcc_lo, s45"),
                     ("s_cmov_b32 s51, s45", "s_cmov_b32 s106, s45"),
                     ("s_bitset0_b32 s52, s47", "s_bitset0_b32 v12, 1"),
                     ("s_bitset1_b32 s52, 31", "s_bitset1_b32 s106, 1"),
                     ("s_bitset0_b32 s52, s47", "s_bitset0_b64 s[52:53], 1"),
                     ("s_bitset1_b32 s52, 31", "s_bitset1_b64 s[52:53], 1"),
                     ("s_bitset0_b32 s52, s47", "s_bitset0_b32 s52, v12")):
        changed = [line.replace(old, new) for line in sop1_program]
        assert changed != sop1_program, new
        must_reject(translate_int, changed)

    # s_bfm_b32: the lowering masks both counts to five bits and leaves SCC untouched.
    bfm_program = bfm_scalar_ops()
    bfm_source = translate_int(bfm_program)
    assert ("= ((1u << (asuint(params[0].w) & 31u)) - 1u) << (asuint(params[1].x) & 31u);" in bfm_source), bfm_source
    assert "s_bfm_b64(" in bfm_source
    refs = [bfm_scalar_ops_reference(words) for words in BFM_WORDS]
    # Counts 0, 1, 31, 32 and 33 for b32 and 63 and 64 for b64 are all covered with distinct results.
    assert [r[0] for r in refs] == [0, 0xfffffffe, 0, 0xfffffffe, 0]
    assert [r[1] for r in refs] == [0x80000000, 0, 0x80000000, 0, 8]
    assert refs[1][2:] == (0xfffffffe, 0) and refs[0][2:] == (0, 0)
    assert refs[2][2:] == (0x80000000, 0x7fffffff) and refs[3][2:] == (0, 0xfffffffe) and refs[4][2:] == (0, 0)
    with_scc = ["s_cmp_eq_u32 s44, s44"]
    scc_program = bfm_program[:]
    at = scc_program.index("v_mov_b32_e32 v12, s50")
    scc_program[at:at] = with_scc + ["s_bfm_b32 s54, 4, 8", "s_cselect_b32 s55, s44, s45"]
    scc_source = translate_int(scc_program)
    assert re.search(r"= scc\d+ \? asuint\(params\[0\]\.w\) : asuint\(params\[1\]\.x\);", scc_source), scc_source
    # Without a tracked SCC the select is rejected even after s_bfm_b32 (it does not create one).
    at = bfm_program.index("v_mov_b32_e32 v12, s50")
    must_reject(translate_int, bfm_program[:at] + ["s_bfm_b32 s54, 4, 8", "s_cselect_b32 s55, s44, s45"] + bfm_program[at:])
    wait = bfm_program.index("s_waitcnt lgkmcnt(0)", bfm_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    # Before the s_waitcnt, s44..s47 are still pending scalar loads: reading them, or overwriting s44, must be refused.
    for bad in ("s_bfm_b32 s50, s44, 1", "s_bfm_b32 s44, 1, 1", "s_bfm_b32 s50, 1, s47"):
        must_reject(translate_int, bfm_program[:wait] + [bad] + bfm_program[wait:])
    for old, new in (("s_bfm_b32 s50, s44, s45", "s_bfm_b32 v12, s44, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, v12, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s44, v12"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s44"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s44, s45, s46"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s106, s44, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s[50:51], s44, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s[44:45], s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s90, s45")):
        changed = [line.replace(old, new) for line in bfm_program]
        assert changed != bfm_program, new
        must_reject(translate_int, changed)


# SDWA moves and half widening. In SDWA_PRESERVE_BODY each move writes a 16-bit field into WORD_0 of a register
# holding 1.0 and keeps its high half, so the word is 1 + f * 2^-23; (v - 1) * 2^23 is split into the low and high
# bytes of f. Channels 0 and 1 read WORD_1 of S0; channels 2 and 3 read BYTE_2 of S1 sign-extended to 16 bits.
# SDWA_PAD_BODY writes BYTE_0 of S0 to BYTE_3 and WORD_0 of S2 to WORD_1 with zero padding, and widens each half
# of S1, the low one through -|x|. Each constant set has wrong references (channel, value) for one selection.
def sdwa_readback(value, high):
    return [line.format(r=value, h=high) for line in SDWA_READBACK]


SDWA_READBACK = ["v_subrev_f32_e32 {r}, 1.0, {r}", "v_mul_f32_e32 {r}, 0x4b000000, {r}", "v_mul_f32_e32 {h}, 0x3b800000, {r}", "v_floor_f32_e32 {h}, {h}", "v_fmac_f32_e32 {r}, 0xc3800000, {h}"]
SDWA_PRESERVE_BODY = [
    "v_mov_b32_e32 v12, 1.0",
    "v_mov_b32_sdwa v12, s44 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
    *sdwa_readback("v12", "v13"),
    "v_mov_b32_e32 v14, 1.0",
    "v_mov_b32_sdwa v14, sext(s45) dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_2",
    *sdwa_readback("v14", "v15"),
]
SDWA_PAD_BODY = [
    "v_mov_b32_sdwa v12, s44 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:BYTE_0",
    "v_cvt_f32_f16_sdwa v13, s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
    "v_cvt_f32_f16_sdwa v14, -|s45| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
    "v_mov_b32_sdwa v15, s46 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
]
SDWA_SETS = (
    # A wrong WORD_0 selection; a zero-extended byte.
    (SDWA_PRESERVE_BODY, (0x12345678, 0x00c00000, 0), [52.0, 18.0, 192.0, 255.0], ((0, 120.0), (1, 86.0), (3, 0.0))),
    (SDWA_PRESERVE_BODY, (0xfedc0000, 0x007f0000, 0), [220.0, 254.0, 127.0, 0.0], ((0, 0.0), (3, 255.0))),
    # BYTE_1 instead of BYTE_0; the other half of S1 in channels 1 and 2; WORD_1 instead of WORD_0 of S2.
    (SDWA_PAD_BODY, (0x0000c340, 0xc5003c00, 0x3f80c0a0), [2.0, -5.0, -1.0, -5.0], ((0, -128.0), (1, 1.0), (2, -5.0), (3, 1.0))),
    (SDWA_PAD_BODY, (0x000047c0, 0x38004900, 0x00004120), [-2.0, 0.5, -10.0, 10.0], ((0, 32768.0), (1, 10.0), (2, -0.5), (3, 0.0))),
)


def sdwa_reference(body, words):
    """Selections and widening on the integer words, independent of the lowering."""
    x, y, z = words
    if body is SDWA_PRESERVE_BODY:
        first = x >> 16
        byte = (y >> 16) & 0xff
        second = (byte - 0x100 if byte & 0x80 else byte) & 0xffff
        return [float(first & 0xff), float(first >> 8), float(second & 0xff), float(second >> 8)]
    as_float = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
    return [as_float((x & 0xff) << 24), half_value(y >> 16), -abs(half_value(y)), as_float((z & 0xffff) << 16)]


def check_sdwa_draw(directory, compiler, renderer, backend):
    for name, body in (("sdwa_preserve", SDWA_PRESERVE_BODY), ("sdwa_pad", SDWA_PAD_BODY)):
        source = directory / f"{name}.hlsl"
        shader = directory / f"{name}.cso"
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
    case = directory / "sdwa.case"
    for body, words, expected, wrongs in SDWA_SETS:
        assert sdwa_reference(body, words) == expected, (words, expected)
        shader = directory / ("sdwa_preserve.cso" if body is SDWA_PRESERVE_BODY else "sdwa_pad.cso")
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *words)
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        for channel, value in wrongs:
            assert value != expected[channel]
            wrong_values = list(expected)
            wrong_values[channel] = value
            case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            result = subprocess.run(draw, capture_output=True, text=True)
            if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
                raise AssertionError(f"A wrong SDWA selection escaped detection for {words} in channel {channel}")


# SDWA moves and half widening. In SDWA_PRESERVE_BODY each move writes a 16-bit field into WORD_0 of a register
# holding 1.0 and keeps its high half, so the word is 1 + f * 2^-23; (v - 1) * 2^23 is split into the low and high
# bytes of f. Channels 0 and 1 read WORD_1 of S0; channels 2 and 3 read BYTE_2 of S1 sign-extended to 16 bits.
# SDWA_PAD_BODY writes BYTE_0 of S0 to BYTE_3 and WORD_0 of S2 to WORD_1 with zero padding, and widens each half
# of S1, the low one through -|x|. Each constant set has wrong references (channel, value) for one selection.
def sdwa_arith(op, dst, operands, dst_sel="DWORD", unused="UNUSED_PAD", clamp=False):
    sels = " src0_sel:DWORD src1_sel:DWORD" if len(operands) == 2 else " src0_sel:DWORD"
    return (f"v_{op}_f32_sdwa {dst}, {', '.join(operands)}{' clamp' if clamp else ''}"
            f" dst_sel:{dst_sel} dst_unused:{unused}{sels}")


SDWA_ARITH = (
    ([
        "v_mov_b32_e32 v18, s45",
        sdwa_arith("add", "v12", ["s44", "v18"]),
        sdwa_arith("sub", "v13", ["-s44", "v18"]),
        sdwa_arith("subrev", "v14", ["s46", "v18"], clamp=True),
        sdwa_arith("mul", "v15", ["-|s46|", "v18"]),
    ], (1.5, 2.25, -3.0, 0.0), [3.75, -3.75, 1.0, -6.75], ((0, 0.75), (1, -0.75), (2, 5.25), (3, 6.75))),
    ([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v19, s47",
        sdwa_arith("max", "v12", ["s44", "v18"]),
        sdwa_arith("min", "v13", ["0.5", "v18"]),
        sdwa_arith("mul_legacy", "v14", ["s46", "v19"]),
        sdwa_arith("floor", "v15", ["-s45"]),
    ], (-0.75, 2.5, 0.0, math.inf), [2.5, 0.5, 0.0, -3.0], ((0, -0.75), (1, 2.5), (2, -0.0), (3, -2.0))),
    ([
        sdwa_arith("ceil", "v12", ["s44"]),
        sdwa_arith("fract", "v13", ["-s45"]),
        sdwa_arith("trunc", "v14", ["|s46|"]),
        "v_mov_b32_e32 v19, 0",
        sdwa_arith("add", "v15", ["s47", "v19"], dst_sel="BYTE_3"),
    ], (-1.25, 2.75, -2.5, 0x3f800040), [-1.0, 0.25, 2.0, 2.0], ((0, -2.0), (1, 0.75), (2, -2.0), (3, 1.0))),
    ([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v12, 1.0",
        sdwa_arith("mul", "v12", ["s44", "v18"], "WORD_0", "UNUSED_PRESERVE"),
        *sdwa_readback("v12", "v13"),
        "v_mov_b32_e32 v14, 1.0",
        sdwa_arith("floor", "v14", ["s46"], "BYTE_2", "UNUSED_PRESERVE"),
        sdwa_arith("mul", "v15", ["s47", "v18"], "WORD_1"),
    ], (0x3f801234, 1.0, 8388613.0, 0xc0a03fff), [52.0, 18.0, 0.51953125, 1.9921875],
     ((0, 128.0), (1, 63.0), (2, 1.0), (3, -5.0))),
)


def sdwa_arith_reference(index, inputs):
    """Python binary32 arithmetic on the constant words, independent of the lowering."""
    as_float = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
    bits = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    x, y, z, w = (as_float(v) if isinstance(v, int) else v for v in inputs)
    if index == 0:
        return [x + y, -x - y, min(max(y - z, 0.0), 1.0), -abs(z) * y]
    if index == 1:
        return [max(x, y), min(0.5, y), 0.0 if z == 0 or w == 0 else z * w, float(math.floor(-y))]
    if index == 2:
        return [float(math.ceil(x)), -y - math.floor(-y), float(math.trunc(abs(z))), as_float((bits(w + 0.0) & 0xff) << 24)]
    product = bits(x * y) & 0xffff
    return [float(product & 0xff), float(product >> 8),
            as_float((0x3f800000 & ~0xff0000) | ((bits(float(math.floor(z))) & 0xff) << 16)), as_float((bits(w * y) & 0xffff) << 16)]


def check_sdwa_arith_draws(directory, compiler, renderer, backend):
    case = directory / "sdwa_arith.case"
    for index, (body, inputs, expected, wrongs) in enumerate(SDWA_ARITH):
        assert sdwa_arith_reference(index, inputs) == expected, (index, sdwa_arith_reference(index, inputs))
        source = directory / f"sdwa_arith{index}.hlsl"
        shader = directory / f"sdwa_arith{index}.cso"
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in inputs]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            result = subprocess.run(draw, capture_output=True, text=True)
            if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
                raise AssertionError(f"A wrong SDWA arithmetic result escaped detection in program {index}, channel {channel}")


def check_constant_descriptor_bounds() -> None:
    for offset in (0, 28, 32):
        program = [line.replace("s[20:21], 0x40", f"s[20:21], {offset:#x}")
                   for line in curve()]
        if offset == 32:
            assert lower(program, 20, 24, [1, 2, 3, 4], offset) == translate(curve())
        else:
            try:
                lower(program, 20, 24, [1, 2, 3, 4], offset)
            except ValueError as error:
                assert "descriptors overlap" in str(error), error
            else:
                raise AssertionError(f"Constant descriptor at {offset} overlaps texture 0")


def check_scalar_constant_reads() -> None:
    program = with_scalar_constant_reads()
    source = translate(program)
    for field in ("params[0].x", "params[0].z", "params[15].w"):
        assert f"= {field};" in source

    load = "s_buffer_load_dword s62, s[44:47], 0x8"
    for bad in (load.replace("0x8", "0x9"), load.replace("0x8", "0x100"), load.replace("0x8", "s60"),
                load.replace("s[44:47]", "s[28:31]"), load.replace("s[44:47]", "s[44:46]")):
        must_reject(translate, [bad if line == load else line for line in program])

    first = program.index("s_buffer_load_dword s61, s[44:47], null")
    wait = program.index("s_waitcnt lgkmcnt(0)", first)
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    must_reject(translate, program[:descriptor + 1] + [load] + program[descriptor + 1:])
    must_reject(translate, program[:wait] + program[wait + 1:])  # Result read before the wait.
    must_reject(translate, program[:wait] + [load] + program[wait:])  # Outstanding write to the same SGPR.
    must_reject(translate, program[:first] + ["s_mov_b32 s46, 0"] + program[first:])
    must_reject(translate, [line.replace("s[44:47]", "s[46:49]") for line in program])

    alias = [line.replace("s63", "s46") for line in program]
    assert translate(alias) == source
    must_reject(translate, alias[:wait + 1] + [load] + alias[wait + 1:])


def check_affine_coordinates() -> None:
    program = affine_coordinates()
    final = "v_fma_f32 v8, 0.5, v8, 0"
    final_index = program.index(final)
    source = translate(program)
    assert f".Sample(samp, float2(alu{final_index}, uv.y))" in source

    exact_input = ["v_mov_b32_e32 v22, 0.5", *program]
    exact_input[final_index + 1] = final.replace("0.5", "v22")
    must_reject(translate, exact_input)
    mask_input = ["v_cmp_eq_f32_e64 s52, 0.5, 0.5", *program]
    mask_input[final_index + 1] = final.replace("0.5", "s52")
    must_reject(translate, mask_input)

    restore = program.index("s_and_b64 exec, exec, s[50:51]")
    exact_overwrite = "v_add_f32_e32 v8, -0.5, v8"
    must_reject(translate, program[:restore + 1] + [exact_overwrite] + program[restore + 1:])
    duplicate = [*program[:final_index], final.replace("v8, 0.5", "v22, 0.5"), *program[final_index:restore],
                 "v_add_f32_e32 v23, 0.0, v9", program[restore], exact_overwrite, *program[restore + 1:]]
    separate_coords = [line.replace("v[8:9]", "v[22:23]") for line in duplicate]
    assert f".Sample(samp, float2(alu{final_index}, alu{restore + 1}))" in translate(separate_coords)
    partial = program.index("v_interp_p2_f32_e32 v8, v1, attr0.x")
    must_reject(translate, program[:partial] + ["v_add_f32_e32 v8, 0.0, v8"] + program[partial:])
    wait = program.index("s_buffer_load_dwordx4 s[60:63], s[44:47], null") + 1
    must_reject(translate, program[:wait] + program[wait + 1:])

    for opcode in ("v_mul_legacy_f32_e32 v8, v9, v8", "v_fma_f32 v8, v9, v8, 0",
                   "v_fma_f32 v8, 0.5, |v8|, 0", "v_fma_f32 v8, 0.5, v8, 0 clamp"):
        changed = program.copy()
        changed[final_index] = opcode
        must_reject(translate, changed)


def check_scalar_helpers_with_vector_arithmetic() -> None:
    entry = assembly(False)
    entry_add = "v_add_f32_e32 v12, 0.0, v12"
    entry.insert(entry.index("s_waitcnt vmcnt(0)") + 1, entry_add)
    quad_add = "v_add_f32_e32 v8, 0.5, v8"
    scalar = "s_bfe_u32 s70, 1, 0x10000"
    for program, vector in ((entry, entry_add), (affine_coordinates(), quad_add)):
        position = program.index(vector)
        for index in (position, position + 1):
            source = translate(program[:index] + [scalar] + program[index:])
            assert "uint s_bfe_u32(" in source and " = s_bfe_u32(" in source


# v_fmaak_f32 computes S0 * S1 + K and v_fmamk_f32 computes S0 * K + S1, with K a 32-bit literal.
# Every product and sum is exact in binary32, so the reference does not depend on fusion. Swapping
# the literal with S1 changes each result.
LITERAL_FMA_INPUTS = (1.5, 2.5, -3.0, 0.75)
LITERAL_FMA_BODY = [
    "v_mov_b32_e32 v13, s45",
    "v_fmaak_f32 v12, s44, v13, 0x3e800000",
    "v_fmamk_f32 v13, s44, 0x40400000, v13",
    "v_mov_b32_e32 v15, s47",
    "v_fmaak_f32 v14, s46, v15, 0xbf000000",
    "v_fmamk_f32 v15, s46, 0x3f000000, v15",
]


def check_literal_fma_draw(directory, compiler, renderer, backend):
    source = directory / "literal_fma.hlsl"
    shader = directory / "literal_fma.cso"
    case = directory / "literal_fma.case"
    source.write_text(translate(with_constants(LITERAL_FMA_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *LITERAL_FMA_INPUTS)
    a, b, c, d = LITERAL_FMA_INPUTS
    expected = [a * b + 0.25, a * 3.0 + b, c * d - 0.5, c * 0.5 + d]
    swapped = [a * 0.25 + b, a * b + 3.0, c * -0.5 + d, c * d + 0.5]
    for x in (*expected, *swapped):
        assert single(x) == x == half_rtz(x), x
    assert all(x != y for x, y in zip(expected, swapped))
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    write_case(case, constants, tuple(expected))
    subprocess.run(draw, check=True)
    # Treating the literal as S1, or S1 as the literal, must change the exported color.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = swapped[k]
        write_case(case, constants, tuple(wrong_values))
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Swapped literal multiply-add operands escaped detection in channel {k}")
def check_scalar_pair_reads() -> None:
    program = with_scalar_pair_reads()
    source = translate(program)
    for field in ("(params[0].w * sample", "max(params[15].z, ", "(params[1].x + sample", "(params[1].y - sample",
                  "= params[1].z;", "(params[15].w * alu"):
        assert field in source, field

    load = PAIR_LOADS[1]
    for old, new in (("0x14", "0x15"), ("0x14", "0x16"),  # not dword-aligned
                     ("0x14", "0xfc"), ("0x14", "0x100"),  # the second word or both pass the 256 bound bytes
                     ("0x14", "s60"), ("0x14", "s60, 0x14"),  # register offset
                     ("s[62:63]", "s62"), ("s[62:63]", "s[62:62]"), ("s[62:63]", "s[62:64]"),
                     ("s[62:63]", "s[60:63]"), ("s[62:63]", "s[61:62]"), ("s[62:63]", "s[106:107]"),
                     ("s[44:47]", "s[28:31]"), ("s[44:47]", "s[44:45]"), ("s[44:47]", "s[44:46]"),
                     ("dwordx2", "dword"), ("dwordx2", "dwordx4"), ("s_buffer_load", "s_load"),
                     ("dwordx2 s[62:63]", "dwordx8 s[56:63]"), ("dwordx2 s[62:63]", "dwordx16 s[48:63]")):
        bad = load.replace(old, new)
        assert bad != load
        must_reject(translate, [bad if line == load else line for line in program])

    first = program.index(PAIR_LOADS[0])
    wait = program.index("s_waitcnt lgkmcnt(0)", first)
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    must_reject(translate, program[:descriptor + 1] + [load] + program[descriptor + 1:])  # Descriptor still loading.
    must_reject(translate, program[:wait] + program[wait + 1:])  # Results read before the wait.
    for word in ("s62", "s63"):  # Neither word of the pair can be read or written until the wait.
        must_reject(translate, program[:wait] + [f"v_mov_b32_e32 v20, {word}"] + program[wait:])
        translate(program[:wait + 1] + [f"v_mov_b32_e32 v20, {word}"] + program[wait + 1:])
        must_reject(translate, program[:wait] + [f"s_mov_b32 {word}, 0"] + program[wait:])
        must_reject(translate, program[:wait] + [f"s_buffer_load_dword {word}, s[44:47], 0x8"] + program[wait:])
    must_reject(translate, program[:wait] + [load] + program[wait:])
    must_reject(translate, program[:wait] + ["s_mov_b64 s[62:63], exec"] + program[wait:])
    translate(program[:wait + 1] + [load, "s_waitcnt lgkmcnt(0)"] + program[wait + 1:])  # Reloading after the wait.
    must_reject(translate, program[:first] + ["s_mov_b32 s46, 0"] + program[first:])

    # A pair may land in the descriptor's own registers, which ends the descriptor.
    alias = [line.replace("s[64:65]", "s[46:47]").replace("s64", "s46").replace("s65", "s47") for line in program]
    assert alias != program and translate(alias) == source
    must_reject(translate, alias[:wait + 1] + [load] + alias[wait + 1:])


# s_setreg_imm32_b32 writes hardware state. The program opens with the scratch base writes, which no lowered
# instruction reads. MODE[3:2] then selects round toward zero for an exact widening, restores round to nearest
# even for two narrowings, and selects round toward zero again before a second exact widening and the
# truncating pack. Channel 0 widens S0, channels 1 and 2 narrow and widen back (channel 2 a subnormal scaled
# by 2^24) and channel 3 widens -|S0|.
FLAT_SCRATCH_WRITES = [
    "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0",
    "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0x400",
]
HALF_RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
HALF_RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"
MODE_BODY = [
    HALF_RTZ,
    "v_cvt_f32_f16_e32 v12, s44",
    HALF_RNE,
    "v_cvt_f16_f32_e32 v13, s45",
    "v_cvt_f32_f16_e32 v13, v13",
    "v_cvt_f16_f32_e32 v14, s46",
    "v_cvt_f32_f16_e32 v14, v14",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    HALF_RTZ,
    "v_cvt_f32_f16_e64 v15, -|s47|",
]
# Each set: the half in S44, two binary32 inputs whose nearest-even and truncated halves differ, the half in
# S47, and the wrong references (channel, value) that truncation after the restore would give.
MODE_SETS = (
    ((0xdeadc500, 1 + 3 * 2.0 ** -11, 1.5 * 2.0 ** -24, 0xbeef7bff), ((1, 1 + 2.0 ** -10), (2, 1.0))),
    ((0x00003c01, -(2.0 ** -14 - 2.0 ** -25), 65519.0, 0x00000001), ((1, -1023 * 2.0 ** -24),)),
)


def mode_program():
    return FLAT_SCRATCH_WRITES + with_constants(MODE_BODY)


def check_mode_register_draw(directory, compiler, renderer, backend):
    source = directory / "mode_register.hlsl"
    shader = directory / "mode_register.cso"
    case = directory / "mode_register.case"
    source.write_text(translate(mode_program()), encoding="ascii")
    compile_shader(compiler, source, shader)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for (first, second, third, last), wrongs in MODE_SETS:
        words = [first, *(struct.unpack("<I", struct.pack("<f", x))[0] for x in (second, third)), last]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        narrowed = half_value(narrow_reference(third))
        expected = [half_value(first), half_value(narrow_reference(second)),
                    narrowed * 2.0 ** 24 if abs(narrowed) < 2.0 ** -14 else narrowed, -abs(half_value(last))]
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            case.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            result = subprocess.run(draw, capture_output=True, text=True)
            if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
                raise AssertionError(f"A truncated narrowing escaped detection in channel {channel}")


def check_mode_register_controls():
    """s_setreg_imm32_b32 is accepted for the scratch base and for MODE[3:2] as tracked rounding state."""
    program = mode_program()
    as_nops = [("s_nop 0" if line.startswith("s_setreg_imm32_b32 ") else line) for line in program]
    assert as_nops.count("s_nop 0") == 5
    # Every accepted write here leaves the lowered result unchanged: generated names carry the instruction
    # index, so the program must translate exactly like s_nop 0 at the same positions.
    assert translate(program) == translate(as_nops)
    plain = with_constants(NARROW_BODY)
    nop = translate(["s_nop 0"] + plain)
    # The scratch base is a whole 32-bit register with any immediate; a field of it is refused.
    for value in ("0", "0x400", "-1", "0x41", "0xfffffffe", "-2147483648", "4294967295"):
        for register in ("LO", "HI"):
            assert translate([f"s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_{register}), {value}"] + plain) == nop
    for write in ("hwreg(HW_REG_FLAT_SCR_LO, 0, 16), 0", "hwreg(HW_REG_FLAT_SCR_HI, 16, 16), 0",
                  "hwreg(HW_REG_FLAT_SCR_LO), s48", "hwreg(HW_REG_FLAT_SCR_LO), 0x100000000",
                  "hwreg(HW_REG_FLAT_SCR_LO), 4294967296", "hwreg(HW_REG_FLAT_SCR_HI), -2147483649",
                  "hwreg(HW_REG_MODE, 0, 2), 3", "hwreg(HW_REG_MODE, 0, 2), 0", "hwreg(HW_REG_MODE, 2, 1), 0",
                  "hwreg(HW_REG_MODE, 3, 2), 0", "hwreg(HW_REG_MODE, 0, 4), 12", "hwreg(HW_REG_MODE, 4, 4), 15",
                  "hwreg(HW_REG_MODE), 0", "hwreg(HW_REG_MODE), 0x2f0", "hwreg(HW_REG_STATUS, 2, 2), 0",
                  "hwreg(HW_REG_STATUS), 0", "hwreg(HW_REG_TRAPSTS, 2, 2), 3", "hwreg(HW_REG_LDS_ALLOC), 0",
                  "hwreg(HW_REG_SHADER_CYCLES), 0", "hwreg(HW_REG_GPR_ALLOC), 0", "hwreg(HW_REG_IB_STS), 0"):
        expect_value_error(translate, [f"s_setreg_imm32_b32 {write}"] + plain, "Unsupported hardware register write")
    # The scratch base stays unobservable: its readers are not lowered.
    for reader in ("scratch_load_dword v12, off, s48", "scratch_store_dword off, v12, s48 offset:16",
                   "flat_load_dword v12, v[12:13]", "s_getreg_b32 s48, hwreg(HW_REG_FLAT_SCR_LO)"):
        expect_value_error(translate, with_constants([*FLAT_SCRATCH_WRITES, reader]), "Unsupported instruction")
    # Nearest even is the entry state, so writing it first is no change. The directed modes are refused.
    assert translate([HALF_RNE] + plain) == nop
    assert translate([HALF_RTZ, HALF_RNE] + plain) == translate(["s_nop 0", "s_nop 0"] + plain)
    for value in ("1", "2", "4", "7", "-1", "0x103"):
        expect_value_error(translate, [f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {value}"] + plain,
                           "Unsupported binary16 rounding mode")
    # The register forms of the same access stay unsupported.
    expect_value_error(translate, ["s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s48"] + plain, "Unsupported instruction")
    expect_value_error(translate, ["s_getreg_b32 s48, hwreg(HW_REG_MODE, 2, 2)"] + plain, "Unsupported instruction")
    # Like the hints, nothing may sit between the color export and s_endpgm.
    must_reject(translate, plain[:-1] + [HALF_RNE, plain[-1]])
    must_reject(translate, plain[:-1] + [FLAT_SCRATCH_WRITES[0], plain[-1]])
    # The write is state, not a hint: under round toward zero the nearest-even narrowing and the packed
    # binary16 arithmetic are refused by name, the last write wins, and restoring nearest even first makes
    # the same programs translate exactly as without the writes.
    expect_value_error(translate, [HALF_RTZ] + plain, "under round toward zero: v_cvt_f16_f32_e32")
    expect_value_error(translate, [HALF_RNE, HALF_RTZ] + plain, "under round toward zero: v_cvt_f16_f32_e32")
    expect_value_error(translate, [line for line in program if line != HALF_RNE],
                       "under round toward zero: v_cvt_f16_f32_e32")
    packed = with_constants(VOP3P_BODY)
    expect_value_error(translate, [HALF_RTZ] + packed, "under round toward zero: v_pk_add_f16")
    assert translate([HALF_RTZ, HALF_RNE] + packed) == translate(["s_nop 0", "s_nop 0"] + packed)
    for operation in ("v_cvt_f16_f32_e64 v12, -|s44|", "v_cvt_f16_u16_e32 v12, v12", "v_add_f16_e32 v12, v12, v13",
                      "v_cvt_f32_f64_e32 v12, v[12:13]", "v_dot2_f32_f16 v12, v12, v13, v14",
                      "v_fma_mixlo_f16 v12, v12, v13, v14", "v_interp_p2_f16 v12, v1, attr0.x"):
        expect_value_error(translate, with_constants([HALF_RTZ, operation]), "under round toward zero")
    # The exact widening and the pack that always truncates do not depend on the mode.
    assert translate([HALF_RTZ] + with_constants(HALF_BODY)) == translate(["s_nop 0"] + with_constants(HALF_BODY))


HINT_NAMES = ("s_clause", "s_setprio", "s_sleep")


def scheduling_hints():
    # Hints the compiler places between instructions. native_program ends the code at s_endpgm,
    # so s_code_end padding never reaches the translator and is not part of the fixture.
    program = assembly(False)
    program.insert(program.index("s_waitcnt vmcnt(0)"), "s_clause 0x1")
    program.insert(1, "s_setprio 2")
    program.insert(program.index("s_waitcnt vmcnt(0)"), "v_nop")
    program.insert(program.index("s_waitcnt vmcnt(0)"), "s_sleep 1")
    return program


def check_scheduling_hints():
    """Scheduling hints change timing only. Generated names carry the instruction index, so
    each hint must translate exactly like s_nop 0 at the same position."""
    plain = assembly(False)
    hinted = scheduling_hints()
    as_nops = [("s_nop 0" if line.split(" ")[0] in (*HINT_NAMES, "v_nop") else line)
               for line in hinted]
    assert translate(hinted) == translate(as_nops)
    nop = translate(["s_nop 0"] + plain)
    for hint, limit in (("s_clause", 0x3f), ("s_setprio", 3), ("s_sleep", 0x7f)):
        assert translate([f"{hint} {limit}"] + plain) == nop
        assert translate([f"{hint} {hex(limit)}"] + plain) == nop
        must_reject(translate, [f"{hint} {limit + 1}"] + plain)
    assert translate(["v_nop"] + plain) == nop
    must_reject(translate, ["v_nop 0"] + plain)
    # s_code_end stays unknown, and nothing may follow the program end or sit between the color
    # export and s_endpgm.
    must_reject(translate, ["s_code_end"] + plain)
    must_reject(translate, plain + ["s_code_end"])
    must_reject(translate, plain + ["s_nop 0"])
    must_reject(translate, plain + ["v_nop"])
    must_reject(translate, plain[:-1] + ["s_setprio 0", plain[-1]])
    must_reject(translate, plain[:-1] + ["v_nop", plain[-1]])


def check_adapter_skip(directory: Path) -> None:
    compiler, renderer = "stub-dxc", "stub-offscreen"
    work = directory / "no-adapter"
    work.mkdir()

    def run_without_adapter(args: list[str | Path], *, check: bool = False) -> subprocess.CompletedProcess[None]:
        assert args[0] in (compiler, renderer)
        result = subprocess.CompletedProcess(args, 77 if args[0] == renderer else 0)
        if check:
            result.check_returncode()
        return result

    with patch.object(subprocess, "run", side_effect=run_without_adapter):
        assert check_render(work, compiler, renderer, "hardware") == 77


def attribute_color(rg):
    code = ["s_mov_b32 s30, s14", "s_mov_b32 m0, s30"]
    for phase in (1, 2):
        for i, channel in enumerate("wzxy"):
            code.append(f"v_interp_p{phase}_f32_e32 v{8+i}, v{phase-1}, attr0.{channel}")
    code.extend(["v_cvt_pkrtz_f16_f32_e32 v12, v8, v9", "v_cvt_pkrtz_f16_f32_e32 v13, v10, v11"])
    tail = "off, off" if rg else "v13, v13"
    return code + [f"exp mrt0 v12, v12, {tail} done compr vm", "s_endpgm"]


def translate_attribute(code):
    return lower(code, 8, 14, None)


def buffer_color(rg):
    # Use a copied low table address and an explicit zero high word.
    tail = "off, off" if rg else "v26, v26"
    return [
        "s_mov_b32 s30, s9", "s_mov_b32 s31, 0",
        "s_load_dwordx4 s[12:15], s[30:31], null", "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[4:7], s[12:15], 0x10", "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v20, s7", "v_mov_b32_e32 v21, s4",
        "v_cvt_pkrtz_f16_f32_e32 v25, v20, v21",
        "v_cvt_pkrtz_f16_f32_e32 v26, v21, v20",
        f"exp mrt0 v25, v25, {tail} done compr vm", "s_endpgm",
    ]


def unsigned_conversion(quad):
    prefix = buffer_color(False)[:6]  # word at byte 16 is in s4 after the wait
    if quad:
        prefix += ["s_mov_b64 s[40:41], exec", "s_wqm_b64 exec, exec",
                   "v_cvt_u32_f32_e32 v32, s4", "s_and_b64 exec, exec, s[40:41]"]
    else:
        prefix += ["v_mov_b32_e32 v32, s4", "v_cvt_u32_f32_e32 v32, v32"]
    # Every possible byte is an exact binary16 and binary32 value.
    return prefix + [f"v_cvt_f32_ubyte{i}_e32 v{20+i}, v32" for i in range(4)] + [
        "v_cvt_pkrtz_f16_f32_e32 v24, v20, v21", "v_cvt_pkrtz_f16_f32_e32 v25, v22, v23",
        "exp mrt0 v24, v24, v25, v25 done compr vm", "s_endpgm",
    ]


def translate_buffer_color(instructions):
    return lower(instructions, 9, 24, None, 0)


def three_way_case(directory, low, high, cap, middle):
    """Expected colors for three_way(); no rounding occurs before the half export."""
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, low, high, cap, middle)
    for y in RAMP:
        for x in RAMP:
            r, g, b = single(x / 255), single(y / 255), single(128 / 255)
            clamped = sorted((r, low, high))[1]
            color = (clamped, min(abs(cap), g, clamped), max(r, b, g), sorted((middle, g, r))[1])
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / "three_way.case"
    path.write_bytes(case)
    return path


def check_three_way_templates():
    """The HLSL forms must pick the expected operand for every operand order."""
    values = (-1.5, 0.0, 0.25, 0.25, 3.0)
    for a in values:
        for b in values:
            for c in values:
                pick = {name: eval(ALU[name][1].format("a", "b", "c"), {"min": min, "max": max},
                                   {"a": a, "b": b, "c": c}) for name in ("min3", "max3", "med3")}
                assert pick == {"min3": min(a, b, c), "max3": max(a, b, c),
                                "med3": sorted((a, b, c))[1]}, (a, b, c)
# Quad-local DPP16 and DPP8 integer/bitwise/shift/bit-scan fixture covering 23 VOP1/VOP2 DPP forms.
# Derives integer quad lane ID {0, 1, 2, 3} from interpolated uv.x/uv.y (v8/v9), verifies quad
# swizzles across lanes, folds all 23 ops into a 32-bit accumulator, and exports all 4 bytes as
# exact [0, 255] integers.
DPP_INT32_BODY = [
    "v_mov_b32_e32 v17, 0",
    "v_or_b32_dpp v18, v8, v17 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_or_b32_dpp v19, v9, v17 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_or_b32_dpp v20, v8, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_or_b32_dpp v21, v9, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_sub_f32_e32 v18, v20, v18",
    "v_sub_f32_e32 v19, v21, v19",
    "v_mul_f32_e32 v18, 0x42800000, v18",
    "v_mul_f32_e32 v19, 0x43000000, v19",
    "v_add_f32_e32 v20, v18, v19",
    "v_add_f32_e32 v20, 0x4b000000, v20",
    "v_mov_b32_e32 v17, 3",
    "v_and_b32_dpp v20, v20, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_sub_nc_u32_dpp v21, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_mul_i32_i24_dpp v21, v21, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_subrev_nc_u32_dpp v22, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_mul_i32_i24_dpp v22, v22, v22 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v23, v20, v20 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_sub_nc_u32_dpp v24, v20, v20 dpp8:[1,2,3,0,5,6,7,4]",
    "v_and_b32_dpp v24, v24, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_e32 v25, s44",
    "v_mov_b32_e32 v26, s45",
    "v_mov_b32_e32 v27, s46",
    "v_mov_b32_e32 v28, s47",
    "v_min_i32_dpp v29, v25, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_max_i32_dpp v30, v25, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_min_u32_dpp v31, v25, v26 dpp8:[3,2,1,0,7,6,5,4]",
    "v_max_u32_dpp v32, v25, v26 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_mul_u32_u24_dpp v33, v25, v26 dpp8:[1,0,3,2,5,4,7,6]",
    "v_mul_hi_u32_u24_dpp v34, v25, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_mul_hi_i32_i24_dpp v35, v25, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v36, v29, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_xnor_b32_dpp v37, v31, v32 dpp8:[2,3,0,1,6,7,4,5]",
    "v_not_b32_dpp v38, v27 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_lshlrev_b32_dpp v39, v28, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v40, v28, v26 dpp8:[1,0,3,2,5,4,7,6]",
    "v_ashrrev_i32_dpp v41, v28, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_bfrev_b32_dpp v42, v27 dpp8:[3,2,1,0,7,6,5,4]",
    "v_ffbl_b32_dpp v43, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_ffbh_u32_dpp v44, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_ffbh_i32_dpp v45, v25 dpp8:[2,3,0,1,6,7,4,5]",
    "v_add_nc_u32_dpp v30, v21, v22 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v23 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v33 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v34 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v35 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v36 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v37 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v38 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v39 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v40 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v41 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v42 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v43 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v44 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v45 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_e32 v46, 0xff",
    "v_mov_b32_e32 v47, 8",
    "v_mov_b32_e32 v48, 16",
    "v_mov_b32_e32 v49, 24",
    "v_and_b32_dpp v50, v30, v46 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v51, v47, v30 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_and_b32_dpp v51, v51, v46 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v52, v48, v30 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_and_b32_dpp v52, v52, v46 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v53, v49, v30 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_cvt_f32_f16_e32 v12, v50",
    "v_mul_f32_e32 v12, 0x4b800000, v12",
    "v_cvt_f32_f16_e32 v13, v51",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_cvt_f32_f16_e32 v14, v52",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_cvt_f32_f16_e32 v15, v53",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

DPP_INT32_SETS = (
    # Set 1: negative i24/i32 in s44, positive in s45, mixed bits in s46, shift >= 32 (37 -> 5).
    (0xff801234, 0x007f5678, 0x00108040, 37),
    # Set 2: extreme i24 (-2^23 * 2^23-1), zero input for ffbl/ffbh (-1), all-ones sign for ffbh_i32.
    (0xffffffff, 0x007fffff, 0x00000000, 13),
    # Set 3: negative * negative i24 (-2^23 * -2^23 -> +0x4000 high), MSB=1 for ffbh_u32=0.
    (0x00800000, 0x00800000, 0x80000001, 31),
)


def dpp_int32_reference(a, b, c, shift):
    m32 = 0xffffffff

    def i32(x):
        x &= m32
        return x - 0x100000000 if x & 0x80000000 else x

    def i24(x):
        x &= 0xffffff
        return x - 0x1000000 if x & 0x800000 else x

    def ffbl(x):
        x &= m32
        return m32 if x == 0 else (x & -x).bit_length() - 1

    def ffbh_u(x):
        x &= m32
        return m32 if x == 0 else 32 - x.bit_length()

    def ffbh_i(x):
        x &= m32
        m = (~x & m32) if (x & 0x80000000) else x
        return m32 if m == 0 else 32 - m.bit_length()

    s = shift & 31
    v21, v22, v23, v24 = 1, 4, 3, 1
    v29 = min(i32(a), i32(b)) & m32
    v30_max = max(i32(a), i32(b)) & m32
    v31 = min(a & m32, b & m32)
    v32 = max(a & m32, b & m32)
    v33 = ((a & 0xffffff) * (b & 0xffffff)) & m32
    v34 = (((a & 0xffffff) * (b & 0xffffff)) >> 32) & m32
    v35 = ((i24(a) * i24(b)) >> 32) & m32
    v36 = (v29 ^ v30_max) & m32
    v37 = (~(v31 ^ v32)) & m32
    v38 = (~c) & m32
    v39 = ((a & m32) << s) & m32
    v40 = (b & m32) >> s
    v41 = (i32(a) >> s) & m32
    v42 = int(f"{c & m32:032b}"[::-1], 2)
    v43 = ffbl(c)
    v44 = ffbh_u(c)
    v45 = ffbh_i(a)

    acc = (v21 + v22 + v23 + v24) & m32
    acc = (acc ^ v33) & m32
    acc = (acc + v34) & m32
    acc = (acc ^ v35) & m32
    acc = (acc + v36) & m32
    acc = (acc ^ v37) & m32
    acc = (acc + v38) & m32
    acc = (acc ^ v39) & m32
    acc = (acc + v40) & m32
    acc = (acc ^ v41) & m32
    acc = (acc + v42) & m32
    acc = (acc ^ v43) & m32
    acc = (acc + v44) & m32
    acc = (acc ^ v45) & m32
    return tuple(float((acc >> (8 * k)) & 0xff) for k in range(4))


def check_dpp_int32_draw(directory, compiler, renderer, backend):
    program = with_constants(DPP_INT32_BODY)
    source = directory / "dpp_int32.hlsl"
    shader = directory / "dpp_int32.cso"
    case = directory / "dpp_int32.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for values in DPP_INT32_SETS:
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dpp_int32_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    for set_idx, old, new, label in (
        (0, "v_sub_nc_u32_dpp v21, v20, v20 quad_perm:[1,0,3,2]",
         "v_sub_nc_u32_dpp v21, v20, v20 quad_perm:[2,3,0,1]", "AcrossX vs AcrossY"),
        (0, "dpp8:[1,2,3,0,5,6,7,4]", "dpp8:[3,0,1,2,7,4,5,6]", "cyclic dpp8 rotation"),
        (0, "v_ashrrev_i32_dpp v41, v28, v25", "v_lshrrev_b32_dpp v41, v28, v25", "ashrrev vs lshrrev"),
        (0, "v_mul_hi_i32_i24_dpp v35", "v_mul_hi_u32_u24_dpp v35", "mul_hi signed vs unsigned"),
        (0, "v_ffbh_i32_dpp v45, v25", "v_ffbh_u32_dpp v45, v25", "ffbh signed vs unsigned"),
    ):
        values = DPP_INT32_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dpp_int32_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


# Quad-local DPP16 (quad_perm) and DPP8 (dpp8) fixture covering all 20 supported VOP1/VOP2 DPP forms.
# Derives per-quad lane ID {0, 1, 2, 3} from interpolated uv.x (v8) and uv.y (v9) so every 2x2 quad
# carries 4 distinct values across all 4,096 pixels while exporting uniform exact binary16 results.
DPP_BODY = [
    "v_subrev_f32_dpp v18, v8, v8 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_subrev_f32_dpp v19, v9, v9 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_mul_f32_e32 v18, 0x42800000, v18",
    "v_fma_f32 v20, 0x43000000, v19, v18",
    "v_sub_f32_dpp v21, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_add_f32_dpp v21, |v21|, |v21| quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_sub_f32_dpp v22, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_max_f32_dpp v22, |v22|, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_add_f32_dpp v23, v20, v20 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_dpp v24, v20 dpp8:[1,2,3,0,5,6,7,4]",
    "v_sub_f32_e32 v25, v24, v20",
    "v_add_f32_e32 v26, 4.0, v25",
    "v_cmp_gt_f32_e32 vcc_lo, 0, v25",
    "v_cndmask_b32_dpp v27, v25, v26, vcc_lo quad_perm:[1,2,0,3] row_mask:0xf bank_mask:0xf",
    "v_add_f32_e32 v12, v21, v22",
    "v_add_f32_e32 v12, v12, v23",
    "v_add_f32_e32 v12, v12, v27",
    # DPP16 neg on src1: v23 is 3 in every lane, so v45 = 3 + -3 = 0 and leaves channel 0 unchanged;
    # a dropped src1 neg gives 6.
    "v_add_f32_dpp v45, v23, -v23 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_add_f32_e32 v12, v12, v45",
    "v_add_f32_e32 v12, s44, v12",
    "v_mov_b32_e32 v28, s45",
    "v_mov_b32_e32 v29, s46",
    "v_mov_b32_e32 v30, s47",
    "v_mul_f32_dpp v31, v20, v28 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf",
    "v_mul_legacy_f32_dpp v32, v20, v28 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_min_f32_dpp v33, -v31, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_cvt_f32_f16_dpp v34, -|v29| quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_e32 v13, v32",
    "v_fmac_f32_dpp v13, v33, v34 dpp8:[1,0,3,2,5,4,7,6]",
    "v_add_f32_e32 v35, s44, v30",
    "v_floor_f32_dpp v36, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_ceil_f32_dpp v37, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_trunc_f32_dpp v38, v35 dpp8:[3,2,1,0,7,6,5,4]",
    "v_fract_f32_dpp v39, v35 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_fma_f32 v14, 2.0, v37, v36",
    "v_fma_f32 v14, 4.0, v38, v14",
    "v_add_f32_e32 v14, v14, v39",
    "v_log_f32_dpp v40, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_exp_f32_dpp v41, v40 dpp8:[2,3,0,1,6,7,4,5]",
    "v_rcp_f32_dpp v42, v41 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_sqrt_f32_dpp v43, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_rsq_f32_dpp v44, v28 dpp8:[3,2,1,0,7,6,5,4]",
    "v_add_f32_e32 v15, v40, v41",
    "v_fma_f32 v15, 0x41800000, v42, v15",
    "v_add_f32_e32 v15, v15, v43",
    "v_fma_f32 v15, 4.0, v44, v15",
    "v_add_f32_e32 v15, 0.5, v15",
    "v_floor_f32_e32 v15, v15",
]

DPP_SETS = (
    # (a: f32, b: power-of-2 f32, h_bits: packed f16 in low half, c: f32)
    (0.25, 4.0, 0x4000, 1.5),
    (-2.25, 16.0, 0xbe00, 0.5),
)


def dpp_reference(a, b, h_bits, c):
    h = struct.unpack("<e", struct.pack("<H", h_bits & 0xffff))[0]
    ch0 = 8.0 + a
    ch1 = 3.0 * b * abs(h)
    x = a + c
    ch2 = math.floor(x) + 2.0 * math.ceil(x) + 4.0 * math.trunc(x) + (x - math.floor(x))
    log_b = math.log2(b)
    ch3 = math.floor(log_b + (2.0 ** log_b) + 16.0 / (2.0 ** log_b) + math.sqrt(b) + 4.0 / math.sqrt(b) + 0.5)
    return (half_rtz(ch0), half_rtz(ch1), half_rtz(ch2), half_rtz(ch3))


def check_dpp_draw(directory, compiler, renderer, backend):
    program = with_constants(DPP_BODY)
    source = directory / "dpp.hlsl"
    shader = directory / "dpp.cso"
    case = directory / "dpp.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    # The NVIDIA driver (32.0.16.2101, RTX 5080) refuses PSO creation for the dpp program
    # under -denorm preserve (E_OUTOFMEMORY), matching the recorded curve exception. Its rows
    # contain no binary32 subnormals, so record this fixture's mode as any and assert that
    # flushing leaves every expected word unchanged.
    compile_shader(compiler, source, shader, mode="any")
    for a, b, h_bits, c in DPP_SETS:
        constants = bytearray(256)
        struct.pack_into("<2fIf", constants, 12, a, b, h_bits, c)
        expected = dpp_reference(a, b, h_bits, c)
        assert all(flush_ftz(word) == word for word in expected), expected
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    for set_idx, old, new, label in (
        (0, "v_sub_f32_dpp v21, v20, v20 quad_perm:[1,0,3,2]",
         "v_sub_f32_dpp v21, v20, v20 quad_perm:[2,3,0,1]", "AcrossX vs AcrossY"),
        (0, "v_sub_f32_dpp v22, v20, v20 quad_perm:[2,3,0,1]",
         "v_sub_f32_dpp v22, v20, v20 quad_perm:[1,0,3,2]", "AcrossY vs AcrossX"),
        (0, "dpp8:[1,2,3,0,5,6,7,4]", "dpp8:[3,0,1,2,7,4,5,6]", "cyclic dpp8 rotation"),
        (0, "v_mul_f32_dpp v31, v20, v28 quad_perm:[3,3,3,3]",
         "v_mul_f32_dpp v31, v20, v28 quad_perm:[1,1,1,1]", "lane broadcast"),
        (0, "v_min_f32_dpp v33, -v31, v30", "v_min_f32_dpp v33, v31, v30", "DPP16 neg modifier"),
        (0, "v_add_f32_dpp v45, v23, -v23", "v_add_f32_dpp v45, v23, v23", "DPP16 src1 neg modifier"),
        (1, "v_floor_f32_dpp v36, v35", "v_ceil_f32_dpp v36, v35", "floor vs ceil DPP"),
    ):
        a, b, h_bits, c = DPP_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<2fIf", constants, 12, a, b, h_bits, c)
        expected = dpp_reference(a, b, h_bits, c)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader, mode="any")
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def pack_f16_pair(lo, hi):
    lo_b = struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_b = struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_b | (hi_b << 16)


def rne_f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


# Mixed-precision VOP3P FMA fixture covering v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16,
# op_sel / op_sel_hi, |.| / -, clamp, inline f16 constants, and untouched-half preservation.
FMA_MIX_BODY = [
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_mov_b32_e32 v22, 0x40003c00",
    "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
    "v_fma_mixlo_f16 v22, s44, 0.5, v21 op_sel:[1,1,0]",
    "v_fma_mix_f32 v23, v22, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
    "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
    "v_cvt_f32_f16_e32 v13, v22",
    "v_fma_mix_f32 v14, v22, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
    "v_fma_mix_f32 v15, v12, 1.0, 0.5 clamp",
]

FMA_MIX_SETS = (
    # Set 1: mixed f16/f32 with |.|, -, op_sel_hi, and untouched-half preservation on both mixlo & mixhi.
    (pack_f16_pair(1.5, -2.0), pack_f16_pair(0.5, -0.25), 0.75, 0.25),
    # Set 2: saturates both v_fma_mixhi_f16 clamp and v_fma_mix_f32 clamp to [0, 1].
    (pack_f16_pair(3.0, -4.0), pack_f16_pair(-0.5, 0.5), 2.5, -1.0),
    # Set 3: exact binary16 halfway ties (odd mantissa -> round up under RNE vs down under RTZ).
    (pack_f16_pair(2.0 + 2 ** -9, 1.0), pack_f16_pair(0.5, 0.25), 1.0 + 2 ** -11 + 2 ** -12, 2 ** -11),
)


def fma_mix_reference(a_bits, b_bits, c_val, d_val):
    a_lo = struct.unpack("<e", struct.pack("<H", a_bits & 0xffff))[0]
    a_hi = struct.unpack("<e", struct.pack("<H", (a_bits >> 16) & 0xffff))[0]
    b_lo = struct.unpack("<e", struct.pack("<H", b_bits & 0xffff))[0]
    b_hi = struct.unpack("<e", struct.pack("<H", (b_bits >> 16) & 0xffff))[0]
    v12 = single(abs(a_hi) * (-b_lo) + c_val)
    lo = rne_f16(single(a_lo * 0.5 + d_val))
    hi = rne_f16(min(max(single(-abs(b_hi) * 2.0 + c_val), 0.0), 1.0))
    v15 = min(max(single(v12 + 0.5), 0.0), 1.0)
    return (half_rtz(v12), half_rtz(lo), half_rtz(hi), half_rtz(v15))


def check_fma_mix_draw(directory, compiler, renderer, backend):
    program = with_constants(FMA_MIX_BODY)
    source = directory / "fma_mix.hlsl"
    shader = directory / "fma_mix.cso"
    case = directory / "fma_mix.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for a_bits, b_bits, c_val, d_val in FMA_MIX_SETS:
        constants = bytearray(256)
        struct.pack_into("<2I2f", constants, 12, a_bits, b_bits, c_val, d_val)
        expected = fma_mix_reference(a_bits, b_bits, c_val, d_val)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    # Mutations verified against Set 1 (modifiers, half selection, preservation) and Set 2 (clamp).
    for set_idx, old, new, label in (
        (0, "v_fma_mix_f32 v12, |s44|, -v19, v20", "v_fma_mix_f32 v12, s44, -v19, v20", "abs modifier"),
        (0, "v_fma_mix_f32 v12, |s44|, -v19, v20", "v_fma_mix_f32 v12, |s44|, v19, v20", "neg modifier"),
        (0, "op_sel:[1,1,0] op_sel_hi:[1,0,0]", "op_sel:[1,1,0]", "op_sel_hi half select"),
        (0, "v_fma_mixhi_f16 v22, -|v19|", "v_fma_mixlo_f16 v22, -|v19|", "mixhi half preservation"),
        (1, "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
         "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0]", "mixhi clamp"),
    ):
        a_bits, b_bits, c_val, d_val = FMA_MIX_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<2I2f", constants, 12, a_bits, b_bits, c_val, d_val)
        expected = fma_mix_reference(a_bits, b_bits, c_val, d_val)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def sopc_group() -> list[str]:
    body = [
        "s_mov_b32 s70, 0",
        "s_mov_b32 s71, 0",
        "s_mov_b32 s72, 1",
        "s_mov_b32 s73, 0",
    ]
    for op in (
        "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
        "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    ):
        body.extend([f"{op} s44, s45", "s_cmov_b64 s[70:71], s[72:73]"])
    body.extend([
        "s_cmp_eq_u64 s[44:45], s[46:47]", "s_cmov_b64 s[70:71], s[72:73]",
        "s_cmp_lg_u64 s[44:45], s[46:47]", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp0_b32 s44, 5", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp1_b32 s44, 5", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp0_b64 s[44:45], 37", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp1_b64 s[44:45], 37", "s_cmov_b64 s[70:71], s[72:73]",
    ])
    return with_constants(body)


def vop2_integer_group() -> list[str]:
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        "v_lshrrev_b32_e32 v12, s44, v18",
        "v_ashrrev_i32_e32 v13, s44, v18",
        "v_lshlrev_b32_e32 v14, s44, v18",
        "v_and_b32_e32 v15, s44, v18",
        "v_or_b32_e32 v12, s44, v18",
        "v_xor_b32_e32 v13, s44, v18",
        "v_add_nc_u32_e32 v14, s44, v18",
        "v_sub_nc_u32_e32 v15, s44, v18",
        "v_subrev_nc_u32_e32 v16, s44, v18",
    ])


# Bodies for uncompressed_export(): s44 = A and s45 = B from the constants, v18 = B, and each channel v12..v15
# holds its own result. Group 3 uses inline constants and register-first operands.
NC_INT_BODIES = {
    1: ["v_add_nc_u32_e64 v12, s44, v18", "v_sub_nc_u32_e64 v13, s44, v18",
        "v_subrev_nc_u32_e32 v14, s44, v18", "v_subrev_nc_u32_e64 v15, s44, v18"],
    2: ["v_add_nc_i32 v12, s44, v18", "v_sub_nc_i32 v13, s44, v18",
        "v_add_nc_u32_e32 v14, s44, v18", "v_sub_nc_u32_e32 v15, s44, v18"],
    3: ["v_subrev_nc_u32_e64 v12, 64, v18", "v_add_nc_i32 v13, 64, v18",
        "v_sub_nc_u32_e64 v14, s44, 64", "v_subrev_nc_u32_e64 v15, v18, s44"],
}
# Reference results by group, from the operand order A (s44) and B (s45): sub is src0 - src1 and subrev is
# src1 - src0, both modulo 2**32. Both follow the translator's float subrev convention; see the docs.
NC_INT_RESULTS = {
    1: lambda a, b: (a + b, a - b, b - a, b - a),
    2: lambda a, b: (a + b, a - b, a + b, a - b),
    3: lambda a, b: (b - 64, 64 + b, a - 64, a - b),
}
# (A, B) as binary32 words; the inputs and every result are normal numbers or zero (asserted at the draw).
NC_INT_ROWS = (
    (0xC0000000, 0x7F000000),  # unsigned wrap
    (0x7F000000, 0x3F800000),  # signed overflow
    (0x40400000, 0x3E800000),  # sub differs from subrev
    (0x3E800000, 0x40400000),  # borrow
    (0x80800000, 0x80800000),  # high-bit wrap, sub is zero
)


def nc_integer_group(group):
    return ["v_mov_b32_e32 v18, s45", *NC_INT_BODIES[group]]


def sop2_lshl_add_group() -> list[str]:
    return with_constants([
        "s_lshl1_add_u32 s60, s44, s45",
        "s_cselect_b32 s71, 1, 0",
        "s_lshl2_add_u32 s61, s44, s45",
        "s_cselect_b32 s72, 2, 0",
        "s_lshl3_add_u32 s62, s44, s45",
        "s_cselect_b32 s73, 4, 0",
        "s_lshl4_add_u32 s63, s44, s45",
        "s_cselect_b32 s74, 8, 0",
        "v_mov_b32_e32 v12, s60",
        "v_mov_b32_e32 v13, s61",
        "v_mov_b32_e32 v14, s62",
        "v_mov_b32_e32 v15, s63",
    ])


def pack_f16_pair(lo, hi):
    lo_b = struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_b = struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_b | (hi_b << 16)


# VOP3P dot-product fixture covering all 7 RDNA 2 (gfx1030) dot-product forms:
# v_dot2_f32_f16 (op_sel, op_sel_hi, neg_lo, neg_hi, clamp), v_dot2_{i,u}32_{i,u}16,
# v_dot4_{i,u}32_{i,u}8, and v_dot8_{i,u}32_{i,u}4 (wrapping and saturating clamp).
DOT_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_dot2_f32_f16 v22, v18, 0x40003800, 0.5 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[1,0,1] neg_hi:[0,1,0]",
    "v_dot2_f32_f16 v12, v18, 0x3c003800, v22 clamp",
    "v_dot2_i32_i16 v23, v19, v20, v21 op_sel:[0,1,0] op_sel_hi:[1,0,1] clamp",
    "v_dot2_u32_u16 v24, v19, v20, v21 clamp",
    "v_dot4_i32_i8 v25, v19, v20, v21 clamp",
    "v_dot4_u32_u8 v26, v19, v20, v21",
    "v_dot8_i32_i4 v27, v19, v20, v21",
    "v_dot8_u32_u4 v28, v19, v20, v21 clamp",
    "v_dot8_u32_u4 v29, v23, 0x12341234, 0",
    "v_dot8_u32_u4 v29, v24, 0x43214321, v29",
    "v_cvt_f32_f16_e32 v13, v29",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_dot8_u32_u4 v30, v25, 0x12341234, 0",
    "v_dot8_u32_u4 v30, v26, 0x43214321, v30",
    "v_cvt_f32_f16_e32 v14, v30",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_dot8_u32_u4 v31, v27, 0x12341234, 0",
    "v_dot8_u32_u4 v31, v28, 0x43214321, v31",
    "v_cvt_f32_f16_e32 v15, v31",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

DOT_SETS = (
    # Set 1: mixed signs, interior v_dot2_f32_f16 value, non-overflowing signed & unsigned integer dots.
    (pack_f16_pair(-0.5, 1.0), 0xfffc0005, 0x0003fffe, 0x00000010),
    # Set 2: positive signed saturation (INT32_MAX) and unsigned saturation (UINT32_MAX).
    (pack_f16_pair(2.0, 2.0), 0x7fff7fff, 0x7fff7fff, 0x7ffffff0),
    # Set 3: negative signed saturation (INT32_MIN) and high-end clamp on v_dot2_f32_f16.
    (pack_f16_pair(-2.0, 2.0), 0x8000ffff, 0x7fff7fff, 0x80000010),
    # Set 4: v_dot8_u32_u4 unsigned saturation (UINT32_MAX) vs unwrapped v_dot4_u32_u8 wraparound.
    (pack_f16_pair(0.5, -1.0), 0xffffffff, 0xffffffff, 0xfffffe00),
)


def dot_reference(f16_bits, x, y, acc):
    def sext(val, bits):
        mask = (1 << bits) - 1
        v = val & mask
        return v - (1 << bits) if v & (1 << (bits - 1)) else v

    def sat_i32(v):
        return max(-0x80000000, min(0x7fffffff, v)) & 0xffffffff

    def sat_u32(v):
        return min(0xffffffff, v) & 0xffffffff

    def nibble_dot(a, b):
        return sum(((a >> (4 * k)) & 0xf) * ((b >> (4 * k)) & 0xf) for k in range(8))

    a_lo = struct.unpack("<e", struct.pack("<H", f16_bits & 0xffff))[0]
    a_hi = struct.unpack("<e", struct.pack("<H", (f16_bits >> 16) & 0xffff))[0]
    v22 = single((-a_hi) * 0.5 + a_lo * (-2.0) + (-0.5))
    v12 = min(max(single(a_lo * 0.5 + a_hi * 1.0 + v22), 0.0), 1.0)

    acc_s = sext(acc, 32)
    v23 = sat_i32(sext(x, 16) * sext(y >> 16, 16) + sext(x >> 16, 16) * sext(y, 16) + acc_s)
    v24 = sat_u32((x & 0xffff) * (y & 0xffff) + ((x >> 16) & 0xffff) * ((y >> 16) & 0xffff) + acc)
    v25 = sat_i32(sum(sext(x >> (8 * k), 8) * sext(y >> (8 * k), 8) for k in range(4)) + acc_s)
    v26 = (sum(((x >> (8 * k)) & 0xff) * ((y >> (8 * k)) & 0xff) for k in range(4)) + acc) & 0xffffffff
    v27 = (sum(sext(x >> (4 * k), 4) * sext(y >> (4 * k), 4) for k in range(8)) + acc_s) & 0xffffffff
    v28 = sat_u32(nibble_dot(x, y) + acc)

    c1 = nibble_dot(v23, 0x12341234) + nibble_dot(v24, 0x43214321)
    c2 = nibble_dot(v25, 0x12341234) + nibble_dot(v26, 0x43214321)
    c3 = nibble_dot(v27, 0x12341234) + nibble_dot(v28, 0x43214321)
    assert c1 < 1024 and c2 < 1024 and c3 < 1024, (c1, c2, c3)
    return (half_rtz(v12), float(c1), float(c2), float(c3))


def check_dot_draw(directory, compiler, renderer, backend):
    program = with_constants(DOT_BODY)
    source = directory / "dot.hlsl"
    shader = directory / "dot.cso"
    case = directory / "dot.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for values in DOT_SETS:
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dot_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    for set_idx, old, new, label in (
        (0, "neg_lo:[1,0,1]", "neg_lo:[0,0,1]", "dot2_f32_f16 neg_lo"),
        (0, "v_dot2_i32_i16 v23", "v_dot2_u32_u16 v23", "dot2 signed vs unsigned"),
        (1, "v_dot4_i32_i8 v25, v19, v20, v21 clamp", "v_dot4_i32_i8 v25, v19, v20, v21", "dot4_i32_i8 clamp"),
        (0, "v_dot8_i32_i4 v27", "v_dot8_u32_u4 v27", "dot8 signed vs unsigned"),
        (3, "v_dot8_u32_u4 v28, v19, v20, v21 clamp", "v_dot8_u32_u4 v28, v19, v20, v21", "dot8_u32_u4 clamp"),
    ):
        values = DOT_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dot_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")
# Packed 16-bit integer VOP3P fixture covering all 14 forms: add/sub/min/max (i16/u16),
# mul_lo_u16, mad (i16/u16), and lshlrev/lshrrev/ashrrev shifts, plus op_sel/op_sel_hi, clamp,
# and inline -1. Exports all four bytes of the packed accumulator as exact [0, 255] integers.
PK_INT16_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_pk_add_i16 v22, v18, v19 op_sel:[1,0] op_sel_hi:[0,1] clamp",
    "v_pk_add_u16 v23, v18, v19 clamp",
    "v_pk_sub_i16 v24, v18, v19 clamp",
    "v_pk_sub_u16 v25, v18, v19 clamp",
    "v_pk_mad_i16 v26, v18, v19, v20 clamp",
    "v_pk_mad_u16 v27, v18, v19, v20 clamp",
    "v_pk_min_i16 v28, v22, v24",
    "v_pk_max_i16 v29, v28, v26",
    "v_pk_min_u16 v28, v23, v25",
    "v_pk_max_u16 v28, v28, v27",
    "v_pk_mul_lo_u16 v29, v29, v28",
    "v_pk_add_i16 v29, v29, -1",
    "v_pk_lshlrev_b16 v22, v21, v29 op_sel:[0,1] op_sel_hi:[1,0]",
    "v_pk_lshrrev_b16 v23, v21, v28",
    "v_pk_ashrrev_i16 v24, v21, v18",
    "v_pk_add_u16 v30, v22, v23",
    "v_pk_sub_u16 v30, v30, v24",
    "v_pk_add_u16 v30, v30, v25",
    "v_pk_add_u16 v30, v30, v26",
    "v_pk_add_u16 v30, v30, v29",
    "v_pk_lshlrev_b16 v25, 8, v30 op_sel_hi:[0,1]",
    "v_pk_lshrrev_b16 v25, 8, v25 op_sel_hi:[0,1]",
    "v_pk_lshrrev_b16 v26, 8, v30 op_sel_hi:[0,1]",
    "v_pk_add_u16 v27, v25, 0 op_sel:[1,0] op_sel_hi:[0,0]",
    "v_pk_add_u16 v28, v26, 0 op_sel:[1,0] op_sel_hi:[0,0]",
    "v_cvt_f32_f16_e32 v12, v25",
    "v_mul_f32_e32 v12, 0x4b800000, v12",
    "v_cvt_f32_f16_e32 v13, v26",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_cvt_f32_f16_e32 v14, v27",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_cvt_f32_f16_e32 v15, v28",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

PK_INT16_SETS = (
    # Set 1: mixed signed/unsigned lanes + shift counts >= 16 (testing & 15 masking and ashrrev sign fill).
    (0xf0080015, 0x0010fffe, 0x0007fff9, 0x00130012),
    # Set 2: saturates all 6 clamping ops (signed +/-32767/-32768 and unsigned 0/65535 boundaries).
    (0x70009000, 0x7000a000, 0x4000c000, 0x00020003),
    # Set 3: signed/unsigned extremes (INT16_MIN/MAX, UINT16_MAX) and max 4-bit shift count 15.
    (0x80007fff, 0xffff0001, 0x0002fffd, 0x000f0001),
)


def pk_int16_reference(w18, w19, w20, w21):
    def u16(w, hi):
        return (w >> 16) & 0xffff if hi else w & 0xffff

    def i16(w, hi):
        v = u16(w, hi)
        return v - 0x10000 if v >= 0x8000 else v

    def pack(lo, hi):
        return (lo & 0xffff) | ((hi & 0xffff) << 16)

    def sat_i(v):
        return max(-32768, min(32767, v)) & 0xffff

    def sat_u(v):
        return max(0, min(65535, v)) & 0xffff

    v22 = pack(sat_i(i16(w18, 1) + i16(w19, 0)), sat_i(i16(w18, 0) + i16(w19, 1)))
    v23 = pack(sat_u(u16(w18, 0) + u16(w19, 0)), sat_u(u16(w18, 1) + u16(w19, 1)))
    v24 = pack(sat_i(i16(w18, 0) - i16(w19, 0)), sat_i(i16(w18, 1) - i16(w19, 1)))
    v25 = pack(sat_u(u16(w18, 0) - u16(w19, 0)), sat_u(u16(w18, 1) - u16(w19, 1)))
    v26 = pack(sat_i(i16(w18, 0) * i16(w19, 0) + i16(w20, 0)), sat_i(i16(w18, 1) * i16(w19, 1) + i16(w20, 1)))
    v27 = pack(sat_u(u16(w18, 0) * u16(w19, 0) + u16(w20, 0)), sat_u(u16(w18, 1) * u16(w19, 1) + u16(w20, 1)))
    v28 = pack(min(i16(v22, 0), i16(v24, 0)), min(i16(v22, 1), i16(v24, 1)))
    v29 = pack(max(i16(v28, 0), i16(v26, 0)), max(i16(v28, 1), i16(v26, 1)))
    v28 = pack(min(u16(v23, 0), u16(v25, 0)), min(u16(v23, 1), u16(v25, 1)))
    v28 = pack(max(u16(v28, 0), u16(v27, 0)), max(u16(v28, 1), u16(v27, 1)))
    v29 = pack((u16(v29, 0) * u16(v28, 0)) & 0xffff, (u16(v29, 1) * u16(v28, 1)) & 0xffff)
    v29 = pack((i16(v29, 0) - 1) & 0xffff, (i16(v29, 1) - 1) & 0xffff)
    v22 = pack((u16(v29, 1) << (u16(w21, 0) & 15)) & 0xffff, (u16(v29, 0) << (u16(w21, 1) & 15)) & 0xffff)
    v23 = pack(u16(v28, 0) >> (u16(w21, 0) & 15), u16(v28, 1) >> (u16(w21, 1) & 15))
    v24 = pack((i16(w18, 0) >> (u16(w21, 0) & 15)) & 0xffff, (i16(w18, 1) >> (u16(w21, 1) & 15)) & 0xffff)
    v30 = pack((u16(v22, 0) + u16(v23, 0)) & 0xffff, (u16(v22, 1) + u16(v23, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) - u16(v24, 0)) & 0xffff, (u16(v30, 1) - u16(v24, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v25, 0)) & 0xffff, (u16(v30, 1) + u16(v25, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v26, 0)) & 0xffff, (u16(v30, 1) + u16(v26, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v29, 0)) & 0xffff, (u16(v30, 1) + u16(v29, 1)) & 0xffff)
    return tuple(float((v30 >> (8 * k)) & 0xff) for k in range(4))


def check_pk_int16_draw(directory, compiler, renderer, backend):
    program = with_constants(PK_INT16_BODY)
    source = directory / "pk_int16.hlsl"
    shader = directory / "pk_int16.cso"
    case = directory / "pk_int16.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for values in PK_INT16_SETS:
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = pk_int16_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)
    for set_idx, old, new, label in (
        (0, "v_pk_ashrrev_i16 v24, v21, v18", "v_pk_lshrrev_b16 v24, v21, v18", "ashrrev sign extension"),
        (0, "v_pk_min_i16 v28, v22, v24", "v_pk_min_u16 v28, v22, v24", "signed vs unsigned min"),
        (1, "v_pk_sub_u16 v25, v18, v19 clamp", "v_pk_sub_u16 v25, v18, v19", "sub_u16 clamp"),
        (0, "op_sel:[1,0] op_sel_hi:[0,1] clamp", "op_sel_hi:[1,1] clamp", "op_sel cross-lane swap"),
        (1, "v_pk_mad_i16 v26, v18, v19, v20 clamp", "v_pk_mad_i16 v26, v18, v19, v20", "mad_i16 clamp"),
    ):
        values = PK_INT16_SETS[set_idx]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = pk_int16_reference(*values)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def check_f16_denorm_controls():
    """CPU-side checks behind F16_DENORM_MODE: references and the lowering it relies on."""
    assert F16_DENORM_MODE == "preserve"
    assert f16_denorm_words(flush=False) == (0x84018001, 0x03ff0001, 0x80000200, 0x05fe0002)
    assert f16_denorm_words(flush=True) == (0x84018000, 0x00000000, 0x80000000, 0x00000000)
    body = translate(uncompressed_export(F16_DENORM_BODY)).split("main(", 1)[1]
    # The mode argument holds only while binary16 values are built and narrowed from their encodings;
    # a driver conversion or a native half type would make the result depend on the device's f16 mode.
    for token in ("f32tof16", "f16tof32", "half", "min16float", "float16_t"):
        assert token not in body, token
    # Presence only: the guarantee is that halves go through these helpers, not how many calls appear.
    assert "widen16(" in body and "pack16_rne(" in body, body


def check_nc_integer_lowering():
    """Static checks for the 32-bit no-carry add/sub forms of nc_integer_group(1..3)."""
    # v18 is written by the mov at index 20, and the four group instructions follow at 21..24. A and B are
    # s44 (params[0].w) and s45. sub is src0 - src1; subrev swaps the operands and is src1 - src0.
    a, b = "asuint(params[0].w)", "asuint(alu20)"
    expected = {
        1: (f"{a} + {b}", f"{a} - {b}", f"{b} - {a}", f"{b} - {a}"),
        2: (f"{a} + {b}", f"{a} - {b}", f"{a} + {b}", f"{a} - {b}"),
        3: (f"{b} - 64u", f"64u + {b}", f"{a} - 64u", f"{a} - {b}"),
    }
    for group, expressions in expected.items():
        program = uncompressed_export(nc_integer_group(group))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, group
        source = translate_int(program)
        for offset, expression in enumerate(expressions, start=21):
            statement = f"    precise float alu{offset} = asfloat({expression});"
            assert statement in source.splitlines(), (group, statement)
    assert set(NC_INT_OPS) <= ACCEPTED, set(NC_INT_OPS) - ACCEPTED
    # sub and subrev with the same operands must not lower alike.
    sub_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_sub_nc_u32_e64 v12, s44, v18"]))
    rev_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_subrev_nc_u32_e64 v12, s44, v18"]))
    assert sub_source != rev_source
    # A float-written inline constant on an integer form is taken as its binary32 bits, as for the #495 forms.
    float_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_add_nc_u32_e64 v12, 1.0, v18"]))
    assert "1065353216u" in float_source, float_source
    neg_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_add_nc_u32_e64 v12, -1.0, v18"]))
    assert "3212836864u" in neg_source, neg_source
    # Each refused spelling is paired with its nearest accepted one, so a refusal for another reason shows up.
    for bad, good in (
        ("v_add_nc_u32_e64 v12, s44, v18 clamp", "v_add_nc_u32_e64 v12, s44, v18"),
        ("v_subrev_nc_u32_e64 v12, s44, v18 clamp", "v_subrev_nc_u32_e64 v12, s44, v18"),
        ("v_add_nc_i32 v12, s44, v18 clamp", "v_add_nc_i32 v12, s44, v18"),
        ("v_sub_nc_i32 v12, s44, v18 clamp", "v_sub_nc_i32 v12, s44, v18"),
        ("v_add_nc_i32_e64 v12, s44, v18", "v_add_nc_i32 v12, s44, v18"),
        ("v_sub_nc_u32_e64 v12, abs(v18), v18", "v_sub_nc_u32_e64 v12, v18, v18"),
        ("v_add_nc_u32_e64 v12, s44, -v18", "v_add_nc_u32_e64 v12, s44, v18"),
        ("v_subrev_nc_u32_dpp v12, v18, v18 row_shl:1 row_mask:0xf bank_mask:0xf",
         "v_subrev_nc_u32_e32 v12, v18, v18"),
        ("v_subrev_nc_u32_sdwa v12, v18, v18 src0_sel:DWORD src1_sel:DWORD",
         "v_subrev_nc_u32_e32 v12, v18, v18"),
    ):
        translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))


def check_controls(directory):
    # A syntactically valid DPP VGPR must already have a tracked numeric value.
    must_reject(translate, with_constants([
        "v_mov_b32_dpp v12, v255 dpp8:[1,2,3,0,5,6,7,4]",
    ]))
    # Fast-path quad swizzles must hoist QuadRead* into an unconditional local before any ternary.
    dpp_ternary = translate(with_constants([
        "v_cmp_gt_f32_e32 vcc_lo, 0, v12",
        "v_cndmask_b32_dpp v13, v12, v13, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
        "v_mul_legacy_f32_dpp v12, v12, v13 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    ]))
    assert dpp_ternary.count("QuadReadAcrossX(") == 1 and dpp_ternary.count("QuadReadLaneAt(") == 1
    assert not any("?" in line and "QuadRead" in line for line in dpp_ternary.splitlines())
    check_f16_denorm_controls()
    program = bitwise_program()
    for before, after in (("s44, v18", "s44, |v18|"), ("20, v19", "20, v19 clamp"),
                          ("0x00700000, v18, v19", "0x00700000, v18")):
        altered = [line.replace(before, after) for line in program]
        assert altered != program
        must_reject(translate, altered)
    program = buffer_color(True)
    assert "float2 main(" in translate_buffer_color(program)
    assert "SamplerState" not in translate_buffer_color(program)
    must_reject(lambda p: lower(p, 8, 24, None, 0), program)
    must_reject(lambda p: lower(p, 9, 24, None, 32), program)
    must_reject(lambda p: lower(p, 9, 24, None, 0, second_texture_offset=32), program)
    must_reject(lambda p: lower(p, 9, 24, [0] * 4, 0), program)
    must_reject(lambda p: lower(p, 20, 24, None), assembly(False))
    must_reject(translate_buffer_color, [s for s in program if s != "s_waitcnt lgkmcnt(0)"])
    must_reject(translate_buffer_color, [s.replace("off, off", "off, v26") for s in program])
    check_scheduling_hints()
    check_mode_register_controls()
    check_constant_descriptor_bounds()
    check_scalar_constant_reads()
    check_affine_coordinates()
    check_scalar_helpers_with_vector_arithmetic()
    check_adapter_skip(directory)
    check_scalar_pair_reads()
    # Whole-dword SDWA comparisons lower exactly as the e32 forms; byte and word selections are refused.
    for first in range(0, 16, 4):
        assert translate(predicate_group(first, sdwa=True)) == translate(predicate_group(first)), first
    must_reject(translate, with_constants(["v_sub_f32_e64 v18, s45, 0",
        "v_cmp_lt_f32_sdwa vcc_lo, s44, v18 src0_sel:WORD_1 src1_sel:DWORD"]))
    sopc_source = translate(sopc_group())
    assert set(SOPC_OPS) <= ACCEPTED
    for expected in (
        "asint(asuint(params[0].w)) == asint(asuint(params[1].x))",
        "asint(asuint(params[0].w)) < asint(asuint(params[1].x))",
        "asuint(params[0].w) == asuint(params[1].x)",
        "asuint(params[0].w) < asuint(params[1].x)",
        "asuint(params[0].w) == asuint(params[1].y) && asuint(params[1].x) == asuint(params[1].z)",
        "asuint(params[0].w) != asuint(params[1].y) || asuint(params[1].x) != asuint(params[1].z)",
        "((asuint(params[0].w) & (1u << (5u & 31u))) == 0u)",
        "((asuint(params[0].w) & (1u << (5u & 31u))) != 0u)",
        "(((37u & 32u) != 0u ? (asuint(params[1].x) & (1u << (37u & 31u))) : (asuint(params[0].w) & (1u << (37u & 31u)))) == 0u)",
        "(((37u & 32u) != 0u ? (asuint(params[1].x) & (1u << (37u & 31u))) : (asuint(params[0].w) & (1u << (37u & 31u)))) != 0u)",
    ):
        assert expected in sopc_source, expected
    for bad_sopc in (
        "s_cmp_gt_u64 s[44:45], s[46:47]",
        "s_cmp_eq_i64 s[44:45], s[46:47]",
        "s_cmp_eq_u64 s[65:66], s[46:47]",
        "s_cmp_eq_u32 s106, s44",
        "s_cmp_eq_u64 s[105:106], s[46:47]",
    ):
        must_reject(translate, with_constants([bad_sopc]))
    vop2_source = translate(vop2_integer_group())
    assert set(VOP2_INT_OPS) <= ACCEPTED
    for expected in (
        "(asuint(alu20) >> (asuint(params[0].w) & 31u))",
        "asuint(asint(asuint(alu20)) >> (asuint(params[0].w) & 31u))",
        "(asuint(alu20) << (asuint(params[0].w) & 31u))",
        "(asuint(params[0].w) & asuint(alu20))",
        "(asuint(params[0].w) | asuint(alu20))",
        "(asuint(params[0].w) ^ asuint(alu20))",
        "(asuint(params[0].w) + asuint(alu20))",
        "(asuint(params[0].w) - asuint(alu20))",
        "(asuint(alu20) - asuint(params[0].w))",
    ):
        assert expected in vop2_source, expected
    for bad_vop2 in (
        "v_sub_nc_u32_e32 v12, abs(v18), v18",
        "v_subrev_nc_u32_e32 v12, abs(v18), v18",
        "v_and_b32_e32 v12, -v18, v18",
        "v_and_b32_e32 v12, abs(v18), v18",
        "v_and_b32_e32 v12, v106, v18",
        "v_and_b32_e32 v12, s44, v106",
    ):
        must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", bad_vop2]))
    check_nc_integer_lowering()
    lshl_add_source = translate(sop2_lshl_add_group())
    assert set(SOP2_LSHL_ADD_OPS) <= ACCEPTED
    for expected in (
        "(asuint(params[0].w) << 1u) + asuint(params[1].x)",
        "(asuint(params[0].w) << 2u) + asuint(params[1].x)",
        "(asuint(params[0].w) << 3u) + asuint(params[1].x)",
        "(asuint(params[0].w) << 4u) + asuint(params[1].x)",
        "asuint(params[0].w) >> 31u != 0u",
        "asuint(params[0].w) >> 30u != 0u",
        "asuint(params[0].w) >> 29u != 0u",
        "asuint(params[0].w) >> 28u != 0u",
    ):
        assert expected in lshl_add_source, expected
    for bad_lshl_add in (
        "s_lshl0_add_u32 s62, s44, s45",
        "s_lshl5_add_u32 s62, s44, s45",
        "s_lshl1_add_u32 s62, s44",
        "s_lshl2_add_u32 s62, s44, s45, s46",
        "s_lshl3_add_u32 s62, v12, s45",
        "s_lshl4_add_u32 s62, s44, v13",
        "s_lshl1_add_u32 v12, s44, s45",
    ):
        must_reject(translate, with_constants([bad_lshl_add]))
    rgb = rgb_sample()
    translate(rgb)
    for old, new in (("v[12:14]", "v[12:15]"), ("dmask:0x7", "dmask:0xb")):
        must_reject(translate, [line.replace(old, new) for line in rgb])
    wait = rgb.index("s_waitcnt vmcnt(0)")
    must_reject(translate, rgb[:wait] + rgb[wait+1:])
    must_reject(translate, rgb[:wait] + ["v_mov_b32_e32 v14, 1.0"] + rgb[wait:])
    translate(rgb[:wait] + ["v_mov_b32_e32 v15, 1.0"] + rgb[wait:])
    lod_zero = lod_zero_sample()
    assert ".SampleLevel(samp, float2(uv.x, uv.y), 0.0f)" in translate(lod_zero)
    must_reject(translate, [line.replace("image_sample_lz ", "image_sample ") for line in lod_zero])
    mixed_masks = [line.replace("image_sample ", "image_sample_lz ") for line in assembly(False)]
    restore = mixed_masks.pop(mixed_masks.index("s_and_b64 exec, exec, s[50:51]"))
    phase_two = next(i for i, line in enumerate(mixed_masks) if line.startswith("v_interp_p2"))
    mixed_masks.insert(phase_two, restore)
    must_reject(translate, mixed_masks)
    # v_cvt_f32_f16 widens the low half through widen16 and applies modifiers after widening.
    body = translate(with_constants(HALF_BODY)).split("float4 main", 1)[1]
    assert body.count("widen16(asuint(") == 4 and "(-abs(widen16(" in body, body
    must_reject(translate, with_constants(["v_cvt_f32_f16_e32 v12, 1.0"]))
    must_reject(translate, with_constants(["v_cvt_f32_f16_sdwa v12, s44 src0_sel:WORD_1"]))
    check_scalar_controls()
    # v_cvt_f16_f32 keeps the destination's high half and narrows the modified source through narrow16.
    body = translate(with_constants(NARROW_BODY)).split("float4 main", 1)[1]
    assert body.count("narrow16(") == 3 and body.count("& 0xffff0000u) | narrow16(") == 3, body
    assert "narrow16((-abs(" in body, body
    must_reject(translate, with_constants(["v_cvt_f16_f32_e64 v12, s44 clamp"]))
    must_reject(translate, with_constants(["v_cvt_f16_f32_sdwa v12, s44 dst_sel:WORD_1"]))
    # An inline integer keeps its bit pattern, and a destination holding interpolation state is refused.
    body = translate(with_constants(["v_cvt_f16_f32_e32 v12, -1", "v_cvt_f16_f32_e64 v13, -|0.5|"])).split("float4 main", 1)[1]
    assert "narrow16(asfloat(4294967295u))" in body and "narrow16((-abs(0.5f)))" in body, body
    for register in ("v0", "v1", "v8", "v9"):
        must_reject(translate, with_constants([f"v_cvt_f16_f32_e32 {register}, s44"]))
    # v_cvt_f32_u32/v_cvt_f32_i32 lower the plain e32 encoding without modifiers and the e64
    # encoding with its optional clamp; every other modifier or variant is refused.
    program = uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000))
    body = translate(program).split("float4 main", 1)[1]
    assert body.count("float(asuint(") == 2 and body.count("float(asint(") == 2, body
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert modern != program and translate(modern) == translate(program)
    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    must_reject(translate, program[:wait] + ["v_cvt_f32_u32_e32 v19, s45"] + program[wait:])
    program = uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True))
    body = translate(program).split("float4 main", 1)[1]
    assert body.count("float(asuint(") == 2 and body.count("float(asint(") == 2, body
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert modern != program and translate(modern) == translate(program)
    program = uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True, clamp=True))
    body = translate(program).split("float4 main", 1)[1]
    assert body.count("saturate(float(asuint(") == 2 and body.count("saturate(float(asint(") == 2, body
    # The e64 pending-load guard mirrors the e32 one; the modifier list pins the assembler-measured
    # refusals: omod in every probed spelling (LLVM 15 rejects it, so it cannot round-trip), abs/neg
    # on the integer source, the DPP/SDWA variants, and clamp on the modifier-free e32 encoding.
    must_reject(translate, program[:wait] + ["v_cvt_f32_u32_e64 v19, s45"] + program[wait:])
    for bad in ("v_cvt_f32_u32_e32 v12, s44 clamp", "v_cvt_f32_i32_e32 v12, s44 clamp",
                "v_cvt_f32_u32_e64 v12, |s44|", "v_cvt_f32_i32_e64 v12, -s44",
                "v_cvt_f32_u32_e64 v12, s44 omod:1", "v_cvt_f32_i32_e64 v12, s44 omod:2",
                "v_cvt_f32_u32_e64 v12, s44 omod:3", "v_cvt_f32_i32_e64 v12, s44 clamp omod:2",
                "v_cvt_f32_u32_e64 v12, s44 omod(2)", "v_cvt_f32_i32_e64 v12, s44 mul:2",
                "v_cvt_f32_u32_dpp v12, v13 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                "v_cvt_f32_i32_sdwa v12, v13 dst_sel:DWORD src0_sel:DWORD"):
        must_reject(translate, with_constants([bad]))
    # v_cvt_u32_f32_e32/v_cvt_i32_f32_e32 lower only the plain e32 encoding, through one helper each.
    program = uncompressed_export(cvt_float_body(0x3fc00000, 0xc0300000))
    full = translate_int(program)
    body = full.split("uint4 main", 1)[1]
    assert body.count("cvt_u32_f32(") == 2 and body.count("cvt_i32_f32(") == 2, body
    assert full.count("uint cvt_u32_f32(uint bits)") == 1 and full.count("uint cvt_i32_f32(uint bits)") == 1
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert modern != program and translate_int(modern) == full
    assert "asfloat(1077936128u)" in translate_int(uncompressed_export(["v_cvt_u32_f32_e32 v12, 0x40400000"]))
    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    expect_value_error(translate_int, program[:wait] + ["v_cvt_u32_f32_e32 v19, s45"] + program[wait:],
                       "Constant read before load completion")
    for bad in ("v_cvt_u32_f32_e64 v12, s44", "v_cvt_i32_f32_e64 v12, s44",
                "v_cvt_u32_f32_e64 v12, |s44|", "v_cvt_i32_f32_e64 v12, -s44",
                "v_cvt_u32_f32_e64 v12, s44 clamp", "v_cvt_i32_f32_e64 v12, s44 omod:2",
                "v_cvt_u32_f32_dpp v12, v13 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                "v_cvt_i32_f32_sdwa v12, v13 dst_sel:DWORD src0_sel:DWORD",
                "v_cvt_u32_f32_e32 v12, 0.5", "v_cvt_u32_f32_e32 v12, 1", "v_cvt_i32_f32_e32 v12, -1",
                "v_cvt_rpi_i32_f32_e32 v12, v13", "v_cvt_flr_i32_f32_e32 v12, v13"):
        expect_value_error(translate, with_constants([bad]), "Unsupported instruction")
    for inst, diagnostic in (("v_cvt_i32_f32_e32 v256, s44", "Invalid signed conversion register"),
                             ("v_cvt_i32_f32_e32 v12, v255", "Unbound pixel input v255")):
        expect_value_error(translate, with_constants([inst]), diagnostic)
    # Positive control: the same harness accepts the plain forms, so the rejections above come from the form.
    for good in ("v_cvt_u32_f32_e32 v12, s44", "v_cvt_i32_f32_e32 v12, s44", "v_cvt_u32_f32_e32 v12, 0x40400000"):
        translate(with_constants([good]))
    # Guards: the destination may not overlap an outstanding image_sample, and the lowering needs the entry mask.
    sample = next(i for i, line in enumerate(with_constants([])) if line.startswith("image_sample"))
    for op in ("u32", "i32"):
        overlap = with_constants([])
        overlap.insert(sample + 1, f"v_cvt_{op}_f32_e32 v12, 0x40400000")
        expect_value_error(translate, overlap, "Write overlaps an outstanding sample")
        ok = with_constants([])
        ok.insert(sample + 1, f"v_cvt_{op}_f32_e32 v20, 0x40400000")
        translate(ok)
        if op == "i32":
            quad = with_constants([])
            quad.insert(quad.index("s_wqm_b64 exec, exec") + 1, f"v_cvt_{op}_f32_e32 v20, 0x40400000")
            expect_value_error(translate, quad, "Invalid conversion")
        entry = with_constants([])
        entry.insert(entry.index("s_and_b64 exec, exec, s[50:51]") + 1, f"v_cvt_{op}_f32_e32 v20, 0x40400000")
        translate(entry)
    # v_cvt_f32_u32 and v_cvt_f32_i32 reject float modifiers, float literals, out-of-range immediates and bad registers.
    body = translate(with_constants([
        "v_cvt_f32_i32_e32 v12, -42",
        "v_cvt_f32_u32_e32 v13, 42",
        "v_cvt_f32_i32_e64 v14, 0x80000000",
        "v_cvt_f32_u32_e64 v15, 0x80000000 clamp",
    ])).split("float4 main", 1)[1]
    assert "float(asint(asfloat(4294967254u)))" in body and "float(asuint(asfloat(42u)))" in body, body
    assert "float(asint(asfloat(2147483648u)))" in body and "saturate(float(asuint(asfloat(2147483648u))))" in body, body
    plain_body = translate(with_constants(["v_cvt_f32_i32_e32 v12, s44", "v_cvt_f32_u32_e32 v13, s45"]))
    assert "float(asint(params[0].w))" in plain_body and "float(asuint(params[1].x))" in plain_body, plain_body
    for op in ("v_cvt_f32_u32_e32", "v_cvt_f32_i32_e32"):
        must_reject(translate, with_constants([f"{op} v12, -v18"]))
        must_reject(translate, with_constants([f"{op} v12, |v18|"]))
        must_reject(translate, with_constants([f"{op} v12, abs(v18)"]))
        must_reject(translate, with_constants([f"{op} v12, -s44"]))
        must_reject(translate, with_constants([f"{op} v12, |s44|"]))
        must_reject(translate, with_constants([f"{op} v12, abs(s44)"]))
        must_reject(translate, with_constants([f"{op} v12, 1.5"]))
        must_reject(translate, with_constants([f"{op} v12, 0x100000000"]))
        must_reject(translate, with_constants([f"{op} s40, v18"]))
    must_reject(translate, with_constants(["v_cvt_f32_u32_sdwa v12, s44 src0_sel:WORD_1"]))
    must_reject(translate, with_constants(["v_cvt_f32_i32_sdwa v12, s44 src0_sel:WORD_1"]))
    ctrl_prog = assembly(False)
    ctrl_wait = ctrl_prog.index("s_waitcnt vmcnt(0)")
    must_reject(translate, ctrl_prog[:ctrl_wait] + ["v_cvt_f32_u32_e32 v12, 0"] + ctrl_prog[ctrl_wait:])
    must_reject(translate, ctrl_prog[:ctrl_wait] + ["v_cvt_f32_i32_e32 v12, 0"] + ctrl_prog[ctrl_wait:])
    ctrl_pos = ctrl_prog.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, ctrl_prog[:ctrl_pos] + ["v_cvt_f32_u32_e32 v20, 0"] + ctrl_prog[ctrl_pos:])
    must_reject(translate, ctrl_prog[:ctrl_pos] + ["v_cvt_f32_i32_e32 v20, 0"] + ctrl_prog[ctrl_pos:])
    ctrl_base = with_constants([])
    ctrl_lgkm = ctrl_base.index("s_waitcnt lgkmcnt(0)")
    must_reject(translate, ctrl_base[:ctrl_lgkm] + ["v_cvt_f32_u32_e32 v12, s44"] + ctrl_base[ctrl_lgkm:])
    must_reject(translate, ctrl_base[:ctrl_lgkm] + ["v_cvt_f32_i32_e32 v12, s44"] + ctrl_base[ctrl_lgkm:])
    archive = directory / "input.bin"
    code = bytes.fromhex("000080bf 000081bf")  # s_nop 0 followed by s_endpgm
    check_container_selector(directory, code)
    record = envelope(code)
    archive.write_bytes(b"another_fixture\0" + record)
    assert native_program(archive, "fixture")[1] == code
    long_name = b"synthetic_pixel_with_a_printable_length"
    archive.write_bytes(envelope(code, long_name))
    assert native_program(archive, long_name.decode())[1] == code
    for invalid in (record[:-1], record + b"\0" + record, envelope(code + code)):
        archive.write_bytes(invalid)
        must_reject(native_program, archive, "fixture")

    partial = dual_program("partial")
    wait = partial.index("s_waitcnt vmcnt(1)")
    # Keep the early read of A but issue B first. Now A is still pending.
    reversed_samples = partial.copy()
    reversed_samples[wait - 2:wait] = reversed(reversed_samples[wait - 2:wait])
    must_reject(translate_dual, reversed_samples)
    for count in (2, 63):
        changed = partial.copy()
        changed[wait] = f"s_waitcnt vmcnt({count})"
        must_reject(translate_dual, changed)
    for early_use in ("v_mov_b32_e32 v24, v20", "v_mov_b32_e32 v20, v12",
                      "v_mul_legacy_f32_e32 v12, 1.0, v20",
                      "v_cvt_pkrtz_f16_f32_e32 v24, v12, v20"):
        changed = partial.copy()
        changed.insert(wait + 1, early_use)
        must_reject(translate_dual, changed)
    # Each of B's four destinations remains protected from a constant write.
    for reg in range(20, 24):
        changed = partial.copy()
        changed.insert(wait + 1, f"v_mov_b32_e32 v{reg}, 0")
        must_reject(translate_dual, changed)
    # Packing A is legal here even though B is outstanding.
    translate_dual(partial[:wait + 1] + ["v_cvt_pkrtz_f16_f32_e32 v24, v12, v13"] + partial[wait + 1:])
    # A larger legal wait after draining does not invalidate completed samples.
    end_wait = partial.index("s_waitcnt vmcnt(0)")
    drained = partial[:end_wait + 1] + ["s_waitcnt vmcnt(63)"] + partial[end_wait + 1:]
    translate_dual(drained)
    must_reject(translate_dual, [op.replace("vmcnt(63)", "vmcnt(64)") for op in drained])

    # Composite waitcnt forms with multiple counters, single counters, or raw 0.
    for legal_wait in (
        "s_waitcnt vmcnt(7) lgkmcnt(0)",
        "s_waitcnt vmcnt(7) lgkmcnt(63)",
        "s_waitcnt lgkmcnt(0) vmcnt(0)",
        "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
        "s_waitcnt 0",
        "s_waitcnt 0x0",
    ):
        translate_dual(partial[:end_wait + 1] + [legal_wait] + partial[end_wait + 1:])
    for illegal_wait in (
        "s_waitcnt lgkmcnt(64)",
        "s_waitcnt expcnt(8)",
        "s_waitcnt vmcnt(0) vmcnt(1)",
        "s_waitcnt lgkmcnt(0) lgkmcnt(1)",
        "s_waitcnt expcnt(0) expcnt(1)",
        "s_waitcnt invalid",
    ):
        must_reject(translate_dual, partial[:end_wait + 1] + [illegal_wait] + partial[end_wait + 1:])

    # Verify that composite waitcnt with lgkmcnt(0) satisfies constant read requirements,
    # while lgkmcnt(63) keeps pending scalar loads outstanding.
    cbuf_program = with_constants(["v_mov_b32_e32 v8, s44"])
    cbuf_wait = cbuf_program.index("s_waitcnt lgkmcnt(0)")
    cbuf_composite_ok = cbuf_program.copy()
    cbuf_composite_ok[cbuf_wait] = "s_waitcnt vmcnt(7) lgkmcnt(0)"
    translate(cbuf_composite_ok)
    cbuf_composite_pending = cbuf_program.copy()
    cbuf_composite_pending[cbuf_wait] = "s_waitcnt vmcnt(7) lgkmcnt(63)"
    must_reject(translate, cbuf_composite_pending)

    # LLVM 15 prints "exp mrt0 v16, ..."; newer LLVM prints "exp mrt0, v16, ...".
    for program in (assembly(False), assembly(True), curve(), shaping(), arithmetic(), three_way(), rounding()):
        modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
        assert modern != program
        assert translate(modern) == translate(program)
        must_reject(translate, [line.replace("exp mrt0 ", "exp mrt0,, ") for line in program])

    # Uncompressed binary32 exports need four distinct computed or sampled registers; an integer
    # export refuses the compressed form and nothing may follow the export.
    program = uncompressed_export(["v_mov_b32_e32 v12, s44", "v_mov_b32_e32 v13, s45",
                                   "v_mov_b32_e32 v14, s46", "v_mov_b32_e32 v15, s47"])
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert translate(modern) == translate(program)
    source = translate(program)
    assert "return float4(alu" in source and "export16(" not in source.split("float4 main", 1)[1]
    integral = lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True)
    assert "uint4 main" in integral and "return uint4(asuint(" in integral
    must_reject(lower, with_constants([]), 20, 24, [1, 2, 3, 4],
                constant_descriptor=64, integer_export=True)
    export = program.index("exp mrt0 v12, v13, v14, v15 done vm")
    must_reject(translate, program[:export] + ["exp mrt0 v12, v12, v13, v14 done vm"] + program[export+1:])
    must_reject(translate, program[:export] + ["exp mrt0 v12, v13, v14, v255 done vm"] + program[export+1:])
    sampled = assembly(False)
    sample_wait = sampled.index("s_waitcnt vmcnt(0)")
    must_reject(translate, sampled[:sample_wait] + ["exp mrt0 v12, v13, v14, v15 done vm", "s_endpgm"])
    must_reject(translate, program[:export+1] + ["v_mov_b32_e32 v20, 1.0"] + program[export+1:])

    # The literal of v_fmaak_f32 is its last operand and that of v_fmamk_f32 its second; a register there is refused.
    body = translate(with_constants(LITERAL_FMA_BODY)).split("float4 main", 1)[1]
    assert body.count("mad(") == 4 and "asfloat(1048576000u)" in body, body
    must_reject(translate, with_constants(["v_fmaak_f32 v12, s44, v13, v14"]))
    must_reject(translate, with_constants(["v_fmamk_f32 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_fmaak_f32 v12, s44, 0x3e800000"]))
    for old, new in (("v_sqrt_f32_e32 v18, v18", "v_sqrt_f32_e32 v18, v18, v18"),
                     ("v_add_f32_e32 v19, s44, v13", "v_add_f32_e32 v19, v13")):
        must_reject(translate, [line.replace(old, new) for line in shaping()])
    check_three_way_templates()
    source = translate(three_way())
    for expected in ("max(min(params[1].z, sample16.y), min(max(params[1].z, sample16.y), sample16.x))",
                     "max(max(sample16.x, sample16.z), sample16.y)",
                     "max(min(sample16.x, params[0].w), min(max(sample16.x, params[0].w), params[1].x))",
                     "min(min(abs(params[1].y), sample16.y), alu"):
        assert expected in source, expected
    for old, new in (("v_med3_f32 v12, v12, s44, s45", "v_med3_f32 v12, v12, s44"),
                     ("v_med3_f32 v15, s47", "v_med3_i32 v15, |s47|"),
                     ("v_max3_f32 v14, v12", "v_max3_u32 v14, |v12|"),
                     ("v_min3_f32 v13, |s46|", "v_min3_f16 v13, |s46|")):
        changed = [line.replace(old, new) for line in three_way()]
        assert changed != three_way()
        must_reject(translate, changed)

    program = truncate_values()
    source = translate(program)
    assert "trunc(params[1].x)" in source
    must_reject(translate, [line.replace("v13, s45", "v13, s45, v12") for line in program])
    wait = program.index("s_waitcnt vmcnt(0)")
    program.remove("v_trunc_f32_e32 v13, s45")
    program.insert(wait, "v_trunc_f32_e32 v13, s45")
    must_reject(translate, program)

    # v_cmp_class_f32 classifies the raw encoding; a mask with modifiers or a non-numeric operand is refused.
    body = translate(with_constants(CLASS_BODY)).split("float4 main", 1)[1]
    assert body.count("class32(") == 4 and "& 512u" in body and "asuint(alu" in body, body
    must_reject(translate, with_constants(["v_cmp_class_f32_e64 vcc_lo, s44, -s45"]))
    must_reject(translate, with_constants(["v_cmp_class_f32_e64 vcc_lo, s44"]))
    program = sopk_scalar()
    source = translate(program)
    assert "(asuint(params[0].w) * 3u)" in source and "(asuint(params[1].x) + 4294967294u)" in source

    scc_program = with_constants([
        "s_mov_b32 s52, 0", "s_mov_b32 s53, 0", "s_mov_b32 s54, 0", "s_mov_b32 s55, 0",
        "s_mov_b32 s50, s44", "s_addk_i32 s50, 0xfffe",
        "s_mov_b32 s70, 0", "s_mov_b32 s71, 0", "s_mov_b32 s74, 1", "s_mov_b32 s75, 0",
        "s_cmov_b64 s[70:71], s[74:75]",
    ])

    assert "0x80000000u" in translate(scc_program)
    cmov = scc_program.index("s_cmov_b64 s[70:71], s[74:75]")
    after_mul = scc_program[:cmov] + ["s_mulk_i32 s50, 0x3"] + scc_program[cmov:]
    assert "0x80000000u" in translate(after_mul)

    for old, new in (("s_mulk_i32 s50, 0x3", "s_mulk_i32 s50, 0x10000"),
                     ("s_addk_i32 s51, 0xfffe", "s_addk_i32 s51, 0x10001"),
                     ("s_mov_b32 s50, s44", "s_mov_b32 s50, s28"),
                     ("s_mulk_i32 s50, 0x3", "s_mulk_i32 v50, 0x3"),
                     ("s_addk_i32 s51, 0xfffe", "s_addk_i32 s51, v45")):
        changed = [line.replace(old, new) for line in program]
        assert changed != program
        must_reject(translate, changed)

    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    must_reject(translate, program[:wait] + ["s_mulk_i32 s44, 0x3"] + program[wait:])

    cmpk_program = sopk_cmpk_cmovk()
    cmpk_source = translate_int(cmpk_program)
    for expected in ("(asint(asuint(params[0].w)) < asint(4294934528u))",
                     "(asuint(params[1].x) < 32768u)",
                     "? 4294967294u : ",
                     "? 32767u : "):
        assert expected in cmpk_source, expected
    refs = [sopk_cmpk_cmovk_reference(words) for words in SOPK_CMPK_WORDS]
    for bit in range(len(SOPK_CMPK_OPS)):
        assert any(r[0] & (1 << bit) for r in refs) and any(not (r[0] & (1 << bit)) for r in refs)
        assert any(r[1] & (1 << bit) for r in refs) and any(not (r[1] & (1 << bit)) for r in refs)
    assert any(r[2] == 0xfffffffe for r in refs) and any(r[2] == 0x13579bdf for r in refs)
    base = assembly(False)
    wqm = base.index("s_wqm_b64 exec, exec")
    andexec = base.index("s_and_b64 exec, exec, s[50:51]")
    must_reject(translate, base[:andexec] + ["s_mov_b32 s50, 0", "s_cmovk_i32 s50, 1"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_mov_b32 s50, 0", "s_cmovk_i32 s50, 1"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_mov_b32 s50, 0", "s_cmovk_i32 s50, 1"] + base[wqm + 1:])
    must_reject(translate_int, [line.replace("s_mov_b32 s50, s46", "s_nop 0") for line in cmpk_program])
    cmpk_wait = cmpk_program.index("s_waitcnt lgkmcnt(0)", cmpk_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    for bad in ("s_cmpk_eq_i32 s44, 0x0", "s_cmovk_i32 s44, 0x1"):
        must_reject(translate_int, cmpk_program[:cmpk_wait] + ["s_cmp_eq_u32 1, 1", bad] + cmpk_program[cmpk_wait:])
    for old, new in (("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s44, 0x10000"),
                     ("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s44, -32769"),
                     ("s_cmpk_eq_u32 s45, 0x8000", "s_cmpk_eq_u32 s45, 0x10000"),
                     ("s_cmpk_eq_u32 s45, 0x8000", "s_cmpk_eq_u32 s45, -1"),
                     ("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s106, 0x0"),
                     ("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 v12, 0x0"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s50, 0x10000"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s50, -32769"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s106, 0x1"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 v12, 0x1")):
        changed = [line.replace(old, new) for line in cmpk_program]
        assert changed != cmpk_program, new
        must_reject(translate_int, changed)
    source = translate(rounding())
    for expected in ("= ceil(alu", "= round(alu", "= round(params[1].x);", "= ceil((-params[1].y));"):
        assert expected in source, expected
    # Neighbouring forms: other widths, output modifiers, a second source and the IEEE-flag reciprocal.
    for old, new in (("v_ceil_f32_e32 v12, v18", "v_ceil_f16_e32 v12, v18"),
                     ("v_ceil_f32_e32 v12, v18", "v_ceil_f64_e32 v[12:13], v[18:19]"),
                     ("v_rndne_f32_e32 v13, v19", "v_rndne_f16_e32 v13, v19"),
                     ("v_rndne_f32_e32 v13, v19", "v_rndne_f32_e32 v13, v19, v18"),
                     ("v_rndne_f32_e64 v14, s45", "v_rndne_f32_e64 v14, s45 mul:2"),
                     ("v_ceil_f32_e64 v15, -s46", "v_ceil_f32_e64 v15, -s46 div:2"),
                     ("v_ceil_f32_e64 v15, -s46", "v_rcp_iflag_f32_e64 v15, -s46, v12"),
                     ("v_ceil_f32_e64 v15, -s46", "v_ceil_f32_e64 v15, -s28")):
        changed = [line.replace(old, new) for line in rounding()]
        assert changed != rounding()
        must_reject(translate, changed)

    program = absolute_value()
    alternate = [line.replace("|v12|", "abs(v12)") for line in program]
    assert translate(program) == translate(alternate)
    # Each cube operation reads the three coordinates once and needs all three operands.
    body = translate(with_constants(CUBE_BODY)).split("float4 main", 1)[1]
    assert body.count("precise float3 cube") == 4 and body.count("* 2.0f)") == 1, body
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", "v_cubeid_f32 v12, s44, v18"]))
    # SDWA moves select and place fields; UNUSED_SEXT, byte selections for half widening and
    # non-canonical SDWA text are refused.
    body = translate(with_constants(SDWA_PRESERVE_BODY + SDWA_PAD_BODY)).split("float4 main", 1)[1]
    assert body.count("& 0xffff0000u) | ") == 2 and body.count("asint(") == 1 and body.count("widen16(") == 2, body
    must_reject(translate, with_constants(["v_mov_b32_sdwa v12, s44 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:WORD_1"]))
    must_reject(translate, with_constants(["v_cvt_f32_f16_sdwa v12, s44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1"]))
    must_reject(translate, with_constants(["v_mov_b32_sdwa v12, s44 src0_sel:WORD_1"]))
    # UNUSED_PRESERVE keeps bits only from a tracked numeric value; DWORD and padded writes may replace anything.
    for register in ("v0", "v1", "v8", "v9", "v20"):
        must_reject(translate, with_constants([f"v_mov_b32_sdwa {register}, s44 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1"]))
    body = translate(with_constants(["v_mov_b32_sdwa v12, -0.5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
                                     "v_mov_b32_sdwa v20, s44 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
                                     "v_mov_b32_sdwa v13, v20 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD"])).split("float4 main", 1)[1]
    assert "asuint((-0.5f))" in body, body
    # SDWA arithmetic takes whole-dword sources only and refuses output modifiers and a src1_sel on a unary op.
    body = translate(with_constants(SDWA_ARITH[3][0])).split("float4 main", 1)[1]
    assert body.count("& 0xffff0000u) | ") == 1 and body.count("& 0xff00ffffu) | ") == 1, body
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", sdwa_arith("add", "v12", ["s44", "v18 mul:2"])]))
    must_reject(translate, with_constants([sdwa_arith("add", "v12", ["s44", "v13"]).replace("src0_sel:DWORD", "src0_sel:WORD_1")]))
    must_reject(translate, with_constants([sdwa_arith("floor", "v12", ["s44"]) + " src1_sel:DWORD"]))
    # A partial UNUSED_PRESERVE result needs a tracked numeric destination, as for SDWA moves.
    for register in ("v0", "v8", "v20"):
        must_reject(translate, with_constants([sdwa_arith("floor", register, ["s44"], "BYTE_2", "UNUSED_PRESERVE")]))

    program = assembly(False)
    for old, new in (("s_mov_b32 s21, 0", "s_mov_b32 s21, 1"),
                     ("s_mov_b32 s40, 1", "s_mov_b32 s40, 8"),
                     ("attr0.y", "attr1.y"),
                     ("s_wqm_b64 exec, exec", "s_branch 0")):
        must_reject(translate, [line.replace(old, new) for line in program])
    must_reject(translate, program[:-1])
    for wait in ("s_waitcnt vmcnt(0)", "s_waitcnt lgkmcnt(0)"):
        must_reject(translate, [line for line in program if line != wait])
    for exp_inst in ("s_waitcnt_expcnt null, 0x0", "s_waitcnt_expcnt null, 0x3", "s_waitcnt_expcnt null, 0",
                     "s_waitcnt_expcnt s0, 0x0", "s_waitcnt_expcnt 0x3", "s_waitcnt_expcnt 0",
                     "s_waitcnt expcnt(0)", "s_waitcnt expcnt(7)"):
        assert "return float4(" in translate([exp_inst] + program)
        assert translate(program[:-1] + [exp_inst, program[-1]]) == translate(program)
    must_reject(translate, program + ["s_waitcnt_expcnt null, 0"])
    must_reject(translate, ["s_waitcnt_expcnt null, 0x8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt null, 8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt 0x8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt invalid"] + program)
    must_reject(translate, ["s_waitcnt_expcnt null, invalid"] + program)
    must_reject(translate, ["s_waitcnt expcnt(8)"] + program)

    wait = program.index("s_waitcnt vmcnt(0)")
    overwrite = "v_floor_f32_e32 v13, 1.0"
    must_reject(translate, program[:wait] + [overwrite] + program[wait:])
    free = translate(program[:wait] + ["v_floor_f32_e32 v24, 1.0"] + program[wait:])
    late = translate(program[:wait+1] + [overwrite] + program[wait+1:])
    assert "export16(sample15.y)" in free
    assert "export16(alu17)" in late

    program = arithmetic()
    source = translate(program)
    for expected in ("= sample16.x;", "= (params[2].x * alu20);", "= saturate((alu21 + params[2].y));",
                     "= params[2].z;", "= (alu23 - sample16.y);", "= saturate((alu25 - params[2].w));",
                     "= asfloat(1048576000u);", "= alu29;"):
        assert expected in source, expected
    # A move is a bit copy: LLVM prints integer inline constants as integers.
    for operand, expected in (("1", "asfloat(1u)"), ("-4", "asfloat(4294967292u)"),
                              ("-0.5", "(-0.5f)"), ("0.15915494", "0.15915494f"), ("s40", "asfloat(1u)")):
        changed = [line.replace("v20, 0x3e800000", f"v20, {operand}") for line in program]
        assert f"alu28 = {expected};" in translate(changed), operand
    for old, new in (("v18, v12", "v18, v0"), ("v18, v12", "v18, v8"), ("v18, v12", "v18, v30"),
                     ("v19, s46", "v19, s28"), ("v19, s46", "v19, -s46"), ("v19, s46", "v19, s46 clamp"),
                     ("v18, v12", "v18, v12, v13"), ("v14, -0.5, v14", "v14, v14"),
                     ("v19, v13, v19", "v19, v13, v19, v18"), ("s45 clamp", "s45 clamp mul:2")):
        changed = [line.replace(old, new) for line in program]
        assert changed != program
        must_reject(translate, changed)
    first = program.index("v_mov_b32_e32 v18, v12")
    early = program[:first-1] + [program[first], program[first-1]] + program[first+1:]
    assert early[first] == "s_waitcnt vmcnt(0)"
    must_reject(translate, early)  # copies the sample before the wait
    # Until vmcnt(0), the sample may still fill v12 to v15, so a move into one of them has no defined
    # order with it. The same move into a free register, or into v12 after the wait, is accepted.
    wait = program.index("s_waitcnt vmcnt(0)")
    for register in ("v12", "v13", "v14", "v15"):
        must_reject(translate, program[:wait] + [f"v_mov_b32_e32 {register}, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_ceil_f32_e32 v12, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_sin_f32_e32 v12, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_cos_f32_e64 v13, 1.0"] + program[wait:])
    trig = translate(with_constants(TRIG_BODY))
    assert trig.count("sin32(") == 3 and trig.count("cos32(") == 3, trig  # two calls plus each definition
    assert "saturate(cos32((-params" in trig and "sin32((-abs(" in trig and "cos32(params" in trig, trig
    free = translate(program[:wait] + ["v_mov_b32_e32 v21, 1.0"] + program[wait:])
    assert free.index("tex.Sample") < free.index("= 1.0f;") < free.index("= sample16.x;")
    late = translate(program[:wait+1] + ["v_mov_b32_e32 v12, 1.0"] + program[wait+1:])
    assert "= 1.0f;" in late and "= sample16.x;" not in late
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v17, v14, v15")
    must_reject(translate, program[:pos] + ["v_mov_b32_e32 v20, v16"] + program[pos:])
    pos = program.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, program[:pos] + ["v_mov_b32_e32 v20, 1.0"] + program[pos:])
    pos = program.index("v_interp_p1_f32_e32 v9, v0, attr0.y")
    partial = program[:pos] + ["v_interp_p1_f32_e32 v10, v0, attr0.x"] + program[pos:]
    translate(partial)  # an unpaired first phase is allowed, but its partial value is not numeric
    must_reject(translate, [line.replace("v18, v12", "v18, v10") for line in partial])
    pos = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0x20"))
    must_reject(translate, program[:pos] + ["v_mov_b32_e64 v19, s46"] + program[pos:])

    for old, new in (("0x40", "0x48"), ("0xc", "0xfe"), ("0xc", "0x100"),
                     ("s_waitcnt lgkmcnt(0)", "s_nop 0"), ("s[44:47], 0xc", "s[40:43], 0xc")):
        must_reject(translate, [line.replace(old, new) for line in curve()])
    source = translate(select())
    assert translate(select(True)) == source
    for expected in ("(sample16.x >= sample16.y || isnan(sample16.x) || isnan(sample16.y))",
                     "sample16.x == sample16.y",
                     "params[0].w < sample16.x", "sample16.y >= params[1].x",
                     "? 1.0f : asfloat(0u);", "? sample16.w : params[1].z;"):
        assert expected in source, expected
    for wave64 in (False, True):
        program = select(wave64)
        mask = "s[52:53]" if wave64 else "s52"
        absolute_compare = [line.replace("v13, s45", "v13, |v12|") for line in program]
        assert "abs(" in translate(absolute_compare)
        pos = program.index(f"v_cndmask_b32_e64 v13, v13, s46, {mask}")
        must_reject(translate, program[:pos] + ["s_mov_b32 s52, 0"] + program[pos:])
        for old, new in ((f"s46, {mask}", "s46, s[53:54]" if wave64 else "s46, s50"),
                         ("v_cmp_eq_f32_e32", "v_cmp_eq_f64_e32"),
                         ("v_cmp_ge_f32_e64", "v_cmpx_ge_f32_e64"),
                         (f"v_cmp_ge_f32_e64 {mask}", "v_cmp_ge_f32_e64 " + ("s[56:57]" if wave64 else "s56")),
                         ("v13, s45", "v13, |v12"),
                         ("v15, s47, v15", "v15, s50, v15")):
            changed = [line.replace(old, new) for line in program]
            assert changed != program
            must_reject(translate, changed)
        pos = program.index("s_waitcnt vmcnt(0)")
        must_reject(translate, program[:pos] + program[pos+1:])
    # v_fract_f32 goes through the clamped helper, never through a bare D3D frac.
    body = translate(shaping()).split("float4 main", 1)[1]
    assert "fract32(" in body and "frac(" not in body, body
    # v_mad_f32 and v_mac_f32 use the unfused, denormal-flushing helper, never the fma lowering.
    body = translate(with_constants(MAD_BODY)).split("float4 main", 1)[1]
    assert body.count("mad32(") == 4 and " mad(" not in body, body
    must_reject(translate, with_constants(["v_mad_f32 v12, s44, s45"]))
    must_reject(translate, with_constants(["v_mac_f32_e32 v14, s44, v15, v14"]))
    # v_madak_f32 and v_madmk_f32 use the same helper. The literal is the last operand of madak and the
    # second of madmk; a register there, a missing operand, and the e64, DPP and SDWA spellings are refused.
    body = translate(with_constants(MAD_LITERAL_BODY)).split("float4 main", 1)[1]
    assert body.count("mad32(") == 4 and " mad(" not in body and "asfloat(3414167552u)" in body, body
    must_reject(translate, with_constants(["v_madak_f32 v12, s44, v13, v14"]))
    must_reject(translate, with_constants(["v_madmk_f32 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_madak_f32 v12, s44, 0x3e800000"]))
    must_reject(translate, with_constants(["v_madak_f32_e64 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_madmk_f32_e64 v12, s44, 0x3e800000, v13"]))
    must_reject(translate, with_constants(["v_madak_f32_dpp v12, v13, v14, 0x3e800000 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf"]))
    must_reject(translate, with_constants(["v_madmk_f32_sdwa v12, v13, 0x3e800000, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"]))

    # 64-bit scalar ALU and BFE controls.
    for op in SCALAR_BFE64_OPS:
        bfe = scalar_bfe64(op)
        translate_int(bfe)
        signed = op == "s_bfe_i64"
        high = "4294967295u" if signed else "0u"
        literal = translate_int(scalar_bfe64(op, "0x80000000"))
        assert f"s_bfe64(uint2(2147483648u, {high})," in literal
        pos = bfe.index(f"{op} s[44:45], s[44:45], s46")
        for operands in ("s[45:46], s[44:45], s46", "exec, s[44:45], s46",
                         "s[44:45], exec, s46", "s[44:45], s[44:45], s[46:47]"):
            must_reject(translate_int, bfe[:pos] + [f"{op} {operands}"] + bfe[pos+1:])
        wait = bfe.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
        for operands in ("s[44:45], 0, 0", "s[50:51], s[44:45], 0", "s[50:51], 0, s46"):
            must_reject(translate_int, bfe[:wait] + [f"{op} {operands}"] + bfe[wait:])
    select64 = scalar_select64()
    translate_int(select64)
    selected = select64.index(S_CSELECT_B64)
    for bad in ("s_cselect_b64 s[45:46], s[44:45], s[46:47]",
                "s_cselect_b64 exec, s[44:45], s[46:47]",
                "s_cselect_b64 s[44:45], exec, s[46:47]"):
        must_reject(translate_int, select64[:selected] + [bad] + select64[selected+1:])
    wait = select64.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
    for bad in ("s_cselect_b64 s[44:45], 0, -1", "s_cselect_b64 s[48:49], s[44:45], 0"):
        must_reject(translate_int, select64[:wait] + ["s_cmp_eq_u32 0, 0", bad] + select64[wait:])
    program = scalar64()
    source = translate(program)
    for expected in ("uint2 s_lshl_b64(", "uint2 s_lshr_b64(", "uint2 s_ashr_i64(",
                     "uint2 s_bfm_b64(", "uint s_bfe_u32(", "uint s_bfe_i32(",
                     "reversebits(", "asfloat(salu"):
        assert expected in source, expected
    # Existing programs without shifts/bfm/bfe emit none of the scalar helper functions.
    assert "uint2 s_lshl_b64(" not in translate(assembly(False))
    # Verify 32-bit hex literal sign-extension for s_ashr_i64 vs zero-extension for b64 ops.
    ashr_pos = program.index("s_ashr_i64 s[72:73], s[52:53], s56")
    signed_lit = translate(program[:ashr_pos] + ["s_ashr_i64 s[72:73], 0x80000000, 4"] + program[ashr_pos+1:])
    assert "s_ashr_i64(uint2(2147483648u, 4294967295u), 4u)" in signed_lit
    unsigned_lit = translate(program[:ashr_pos] + ["s_and_b64 s[72:73], s[52:53], 0x80000000"] + program[ashr_pos+1:])
    assert "2147483648u" in unsigned_lit and "0u)" in unsigned_lit
    # Verify all 10 SCC observation bits flip between the two constant sets.
    refs = []
    for values in SCALAR64_SETS:
        constants = bytearray(256)
        struct.pack_into("<12I", constants, 16, *values)
        refs.append(scalar64_reference(program, struct.unpack("<64I", constants)))
    assert (refs[0][1] ^ refs[1][1]) == 0x3ff, (bin(refs[0][1]), bin(refs[1][1]))
    # Negative controls: exec save/restore, #257 ops, out-of-scope 64-bit ops, invalid operands and SCC rules.
    body_start = program.index("s_and_b64 s[72:73], s[52:53], s[54:55]")
    for bad in (
        "s_and_saveexec_b64 s[70:71], s[52:53]",
        "s_or_saveexec_b64 s[70:71], s[52:53]",
        "s_andn2_saveexec_b64 s[70:71], s[52:53]",
        "s_or_b64 exec, exec, s[52:53]",
        "s_and_b64 vcc, s[52:53], s[54:55]",
        "s_cmov_b64 exec, s[52:53]",
        "s_not_b64 vcc, s[52:53]",
        "s_and_b64 s[70:71], exec, s[52:53]",
        "s_xor_b64 s[70:71], vcc, s[52:53]",
        "s_cmov_b64 s[70:71], s[52:53]",  # before any SCC write
        "s_and_b64 s[53:54], s[52:53], s[54:55]",  # unaligned pair
        "s_and_b64 s[106:107], s[52:53], s[54:55]",  # out-of-range SGPR pair
        "s_bfe_u32 s106, s60, 0x80004",  # out-of-range SGPR
        "s_bfe_u32 s70, v12, 0x80004",  # VGPR source
        "s_bfe_u32 s70, m0, 0x80004",  # m0 source
        "s_and_b64 s[70:71], s[28:29], s[52:53]",  # texture descriptor source
        "s_and_b64 s[70:71], s[44:45], s[52:53]",  # CBV descriptor source
        "s_and_b64 s[70:71], s[50:51], s[52:53]",  # saved exec source
        "s_and_b64 s[70:71], s[20:21], s[52:53]",  # texture_ptr source
        "s_and_b64 s[70:71], s[90:91], s[52:53]",  # unset source
        "s_cselect_b64 s[70:71], s[52:53], s[54:55]",  # before any SCC write
        "s_bfm_b32 s70, v12, 8",  # VGPR source
        "s_bfm_b32 s106, 4, 8",  # out-of-range SGPR
        "s_bfm_b32 s[70:71], 4, 8",  # pair destination
        "s_bfm_b32 s70, 4",  # missing operand
        "s_bfm_b32 s70, s90, 8",  # unset source
        "s_cselect_b32 s70, s56, s57",
    ):
        must_reject(translate, program[:body_start] + [bad] + program[body_start:])
    for supported in (
        "s_add_u32 s70, s56, s57",
        "s_and_b32 s70, s56, s57",
        "s_not_b32 s70, s56",
        "s_bfm_b32 s70, 4, 8",
        "s_cmp_eq_i32 s56, s57",
    ):
        translate(program[:body_start] + [supported] + program[body_start:])
    # s_cmov_b64 rejects unset or non-numeric destination pairs even after SCC is set.
    after_and = body_start + 1
    for bad_dst in ("s[90:91]", "s[44:45]", "s[50:51]", "s[28:29]"):
        must_reject(translate, program[:after_and] + [f"s_cmov_b64 {bad_dst}, s[52:53]"] + program[after_and:])
    # Pending scalar load read and write overlap must be rejected.
    load_wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[60:63], s[44:47], 0x30"))
    for early in ("s_and_b64 s[70:71], s[52:53], s[54:55]", "s_not_b64 s[52:53], 0", "s_bfe_u32 s60, 1, 0x80004"):
        must_reject(translate, program[:load_wait] + [early] + program[load_wait:])
    # s_wqm_b64 and exec-restoring s_and_b64 clear tracked SCC.
    wqm_pos = program.index("s_wqm_b64 exec, exec")
    exec_restore_pos = program.index("s_and_b64 exec, exec, s[50:51]")
    prefix = ["s_mov_b32 s90, 0", "s_mov_b32 s91, 0", "s_not_b64 s[92:93], 0"]
    must_reject(translate, program[:wqm_pos] + prefix + [program[wqm_pos], "s_cmov_b64 s[90:91], s[92:93]"] + program[wqm_pos+1:])
    must_reject(translate, program[:exec_restore_pos] + prefix + [program[exec_restore_pos], "s_cmov_b64 s[90:91], s[92:93]"] + program[exec_restore_pos+1:])
    # A scalar result cannot supply a sampler word, texture address high word, or v_cndmask condition.
    sample_pos = program.index("image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D")
    must_reject(translate, program[:sample_pos] + ["s_bfe_u32 s40, 1, 0x10000"] + program[sample_pos:])
    sel = select(True)
    cnd_pos = sel.index("v_cndmask_b32_e64 v13, v13, s46, s[52:53]")
    must_reject(translate, sel[:cnd_pos] + ["s_not_b64 s[52:53], 0"] + sel[cnd_pos:])

    # 32-bit integer comparisons: signed vs unsigned casts, wave64 spelling, inline constants, and rejections.
    i32_source = translate(int_predicate_group("i32", "e32", 0))
    u32_source = translate(int_predicate_group("u32", "e64", 4))
    assert "asint(params[0].w) < asint(alu20)" in i32_source, i32_source
    assert "asuint(params[0].w) >= asuint(alu20)" in u32_source, u32_source
    assert translate(int_predicate_group("i32", "e32", 0, wave64=True)) == i32_source
    assert translate(int_predicate_group("u32", "e64", 4, wave64=True)) == u32_source
    for kind, token, expected in (
        ("i32", "-1", "asint(4294967295u)"),
        ("u32", "-1", "4294967295u"),
        ("i32", "0x80000000", "asint(2147483648u)"),
        ("u32", "0x80000000", "2147483648u"),
        ("i32", "1.0", "asint(1065353216u)"),
        ("u32", "1.0", "1065353216u"),
    ):
        prog = with_constants(["v_mov_b32_e32 v18, s45",
                               f"v_cmp_lt_{kind}_e32 vcc_lo, {token}, v18",
                               "v_cndmask_b32_e32 v12, 0, 1.0, vcc_lo"])
        assert f"bool cmp21 = {expected} < " in translate(prog), (kind, token)
    base_int = int_predicate_group("i32", "e64", 0)
    for bad in (
        "v_cmp_lt_i32_e64 s52, -s44, v18",
        "v_cmp_lt_i32_e64 s52, |s44|, v18",
        "v_cmp_lt_i32_e64 s52, abs(s44), v18",
        "v_cmp_lt_i32_e64 s52, s44, -v18",
        "v_cmp_lt_i32_e64 s52, s44, v18 clamp",
        "v_cmpx_lt_i32_e64 s52, s44, v18",
        "v_cmp_lt_i32_e64 s52, 0x100000000, v18",
    ):
        changed = [line.replace("v_cmp_lt_i32_e64 s52, s44, v18", bad) for line in base_int]
        assert changed != base_int
        must_reject(translate, changed)
    pos = base_int.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_u32_e32 vcc_lo, v12, v13"] + base_int[pos:])
    load = next(i for i, line in enumerate(base_int) if line.startswith("s_buffer_load_dwordx4"))
    pos = base_int.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_i32_e32 vcc_lo, s44, 0"] + base_int[pos:])
    pos = base_int.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_i32_e32 vcc_lo, 0, 0"] + base_int[pos:])
    # Every 32-bit integer comparison form is lowered here too, so tools/form_coverage.py counts all of them.
    for kind in ("i32", "u32"):
        for encoding in ("e32", "e64"):
            for first in (0, 4):
                translate(int_predicate_group(kind, encoding, first))

    # 16-bit integer comparisons: low-half extension, operands, wave64 spelling and rejected neighbours.
    i16_source = translate(int16_group("i16", "e32", 1))
    u16_source = translate(int16_group("u16", "e64", 1))
    assert "(asint(asuint(params[0].w) << 16) >> 16) != (asint(asuint(alu20) << 16) >> 16)" in i16_source, i16_source
    assert "(asint(4294967280u << 16) >> 16) < (asint(asuint(alu20) << 16) >> 16)" in i16_source, i16_source
    assert "(asint(32768u << 16) >> 16) > " in i16_source, i16_source
    assert "(asuint(alu20) & 0xffffu) >= (asuint(params[0].w) & 0xffffu)" in u16_source, u16_source
    assert "(asuint(alu20) & 0xffffu) < (64u & 0xffffu)" in u16_source, u16_source
    assert "(asuint(alu20) & 0xffffu) <= (32768u & 0xffffu)" in u16_source, u16_source
    for kind in ("i16", "u16"):
        for encoding in ("e32", "e64"):
            for group in (0, 1):
                assert (translate(int16_group(kind, encoding, group, wave64=True))
                        == translate(int16_group(kind, encoding, group)))
    for kind, signed in (("i16", True), ("u16", False)):
        for left, right in INT16_PAIRS:
            low = [w & 0xffff for w in (left, right)]
            if signed:
                low = [v - 0x10000 if v & 0x8000 else v for v in low]
            expected = [float(low[0] < low[1]), float(low[0] == low[1]),
                        float(low[0] <= low[1]), float(low[0] > low[1])]
            assert int16_reference(kind, "e32", 0, left, right) == expected
    left, right = INT16_PAIRS[0]
    assert int16_reference("i16", "e32", 0, left, right) != int16_reference("u16", "e32", 0, left, right)
    base16 = int16_group("i16", "e64", 0)
    target = "v_cmp_lt_i16_e64 s52, s44, v18"
    for bad in (
        "v_cmp_lt_i16_e64 s52, -s44, v18",
        "v_cmp_lt_i16_e64 s52, |s44|, v18",
        "v_cmp_lt_i16_e64 s52, s44, v18 clamp",
        "v_cmp_lt_i16_e64 s52, s44, v18 op_sel:[1,0,0]",
        "v_cmp_lt_i16_e64 s52, 1.0, v18",
        "v_cmp_lt_i16_e64 s52, 65, v18",
        "v_cmp_lt_i16_e64 s52, -17, v18",
        "v_cmp_lt_i16_e64 s52, 0x10000, v18",
        "v_cmp_f_i16_e64 s52, s44, v18",
        "v_cmp_t_u16_e64 s52, s44, v18",
        "v_cmpx_lt_i16_e64 s52, s44, v18",
        "v_cmpx_lt_i16_e32 s44, v18",
        "v_cmpx_lt_i32_e32 s44, v18",
        "v_cmp_lt_i64_e64 s52, s[44:45], v[18:19]",
        "v_cmp_lt_u64_e32 vcc_lo, s[44:45], v[18:19]",
        "v_cmp_class_f16_e32 vcc_lo, s44, v18",
        "v_cmp_lt_f16_e64 s52, s44, v18",
    ):
        changed = [line.replace(target, bad) for line in base16]
        assert changed != base16
        must_reject(translate, changed)
    pos = base16.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base16[:pos] + ["v_cmp_eq_u16_e32 vcc_lo, v12, v13"] + base16[pos:])
    load = next(i for i, line in enumerate(base16) if line.startswith("s_buffer_load_dwordx4"))
    pos = base16.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base16[:pos] + ["v_cmp_eq_i16_e32 vcc_lo, s44, 0"] + base16[pos:])
    pos = base16.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base16[:pos] + ["v_cmp_eq_i16_e32 vcc_lo, 0, 0"] + base16[pos:])

    # 32-bit integer min/max/min3/max3/med3 controls and rejections.
    pair_src = translate(int_minmax_pair("e32"))
    assert "asfloat(min(asint(params[0].w), asint(alu20)))" in pair_src, pair_src
    assert "asfloat(max(asuint(params[0].w), asuint(alu20)))" in pair_src, pair_src
    assert translate(int_minmax_pair("e64")) == pair_src
    tri_i32 = translate(int_minmax_triple("i32"))
    tri_u32 = translate(int_minmax_triple("u32"))
    assert "asfloat(min(min(asint(params[0].w), asint(alu20)), asint(alu21)))" in tri_i32, tri_i32
    assert tri_i32.count("int3 med") == 2 and tri_u32.count("uint3 med") == 2
    for bad_pair in (
        "v_min_i32_e64 v12, -s44, v18",
        "v_min_i32_e64 v12, |s44|, v18",
        "v_min_i32_e64 v12, s44, v18 clamp",
        "v_min_i32_e64 v12, s44",
        "v_min_i32_e64 v12, 0x100000000, v18",
    ):
        changed = [line.replace("v_min_i32_e64 v12, s44, v18", bad_pair) for line in int_minmax_pair("e64")]
        assert changed != int_minmax_pair("e64")
        must_reject(translate, changed)
    for bad_tri in (
        "v_med3_i32 v14, -s44, v18, v19",
        "v_med3_i32 v14, s44, |v18|, v19",
        "v_med3_i32 v14, s44, v18, v19 clamp",
        "v_med3_i32 v14, s44, v18",
    ):
        changed = [line.replace("v_med3_i32 v14, s44, v18, v19", bad_tri) for line in int_minmax_triple("i32")]
        assert changed != int_minmax_triple("i32")
        must_reject(translate, changed)
    base_pair = int_minmax_pair("e32")
    wait = base_pair.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_pair[:wait] + ["v_min_i32_e32 v12, 0, 1"] + base_pair[wait:])
    must_reject(translate, base_pair[:wait] + ["v_med3_u32 v13, 0, 1, 2"] + base_pair[wait:])
    load = next(i for i, line in enumerate(base_pair) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_pair.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_pair[:lgkm] + ["v_max_u32_e32 v20, s44, 0"] + base_pair[lgkm:])
    wqm = base_pair.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_pair[:wqm] + ["v_min_u32_e32 v20, 0, 1"] + base_pair[wqm:])

    # v_frexp_mant_f32 and v_ldexp_f32 controls and rejections.
    frexp_body = translate(frexp_mant_values()).split("float4 main", 1)[1]
    assert "frexp_mant32(params[0].w)" in frexp_body, frexp_body
    assert "frexp_mant32((-abs(params[1].x)))" in frexp_body, frexp_body
    assert "saturate(frexp_mant32(params[1].z))" in frexp_body, frexp_body
    ldexp_body = translate(ldexp_values()).split("float4 main", 1)[1]
    assert "ldexp32(alu20, asint(params[1].x))" in ldexp_body, ldexp_body
    assert "ldexp32((-abs(params[0].w)), asint(2u))" in ldexp_body, ldexp_body
    assert "ldexp32(params[1].y, asint(4294967294u))" in ldexp_body, ldexp_body
    assert "saturate(ldexp32(params[1].z, asint(3u)))" in ldexp_body, ldexp_body
    for bad in (
        "v_ldexp_f32 v12, v18, -s45",
        "v_ldexp_f32 v12, v18, |s45|",
        "v_ldexp_f32 v12, v18, abs(s45)",
        "v_ldexp_f32 v12, v18",
        "v_ldexp_f32 v12, v18, 0x100000000",
    ):
        changed = [line.replace("v_ldexp_f32 v12, v18, s45", bad) for line in ldexp_values()]
        assert changed != ldexp_values()
        must_reject(translate, changed)
    base_ld = ldexp_values()
    wait = base_ld.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_ld[:wait] + ["v_frexp_mant_f32_e32 v12, 1.0"] + base_ld[wait:])
    must_reject(translate, base_ld[:wait] + ["v_ldexp_f32 v13, 1.0, 2"] + base_ld[wait:])
    load = next(i for i, line in enumerate(base_ld) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_ld.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_ld[:lgkm] + ["v_ldexp_f32 v20, 1.0, s45"] + base_ld[lgkm:])
    wqm = base_ld.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_ld[:wqm] + ["v_ldexp_f32 v20, 1.0, 2"] + base_ld[wqm:])

    # v_mul_{i32_i24,u32_u24,hi_i32_i24,hi_u32_u24} controls and rejections.
    mul24_hlsl = translate(mul24_values())
    for helper_sig in (
        "uint mul24_u32(uint a, uint b)",
        "uint mul24_i32(uint a, uint b)",
        "uint mul24_hi_u32(uint a, uint b)",
        "uint mul24_hi_i32(uint a, uint b)",
    ):
        assert helper_sig in mul24_hlsl, helper_sig
    assert mul24_hlsl.index("uint mul24_hi_u32") < mul24_hlsl.index("uint mul24_hi_i32")
    # Using only v_mul_hi_i32_i24 must still pull in its mul24_hi_u32 dependency before it.
    hi_only = translate(with_constants(["v_mul_hi_i32_i24_e64 v12, s44, s45", "v_mov_b32_e32 v13, 0",
                                        "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0"]))
    assert hi_only.index("uint mul24_hi_u32") < hi_only.index("uint mul24_hi_i32")
    assert "mul24_u32(" not in hi_only and "mul24_i32(" not in hi_only
    base_hlsl = translate(assembly(False))
    assert "mul24_" not in base_hlsl
    mul24_main = mul24_hlsl.split("float4 main", 1)[1]
    for snippet in (
        "asfloat(mul24_i32(asuint(params[0].w), asuint(alu",
        "asfloat(mul24_i32(asuint(params[1].y), asuint(params[1].z)))",
        "asfloat(mul24_u32(asuint(params[1].x), asuint(alu",
        "asfloat(mul24_u32(asuint(params[1].z), asuint(params[0].w)))",
        "asfloat(mul24_hi_i32(asuint(params[0].w), asuint(alu",
        "asfloat(mul24_hi_i32(asuint(params[1].y), asuint(params[1].z)))",
        "asfloat(mul24_hi_u32(asuint(params[1].x), asuint(alu",
        "asfloat(mul24_hi_u32(asuint(params[1].z), asuint(params[0].w)))",
    ):
        assert snippet in mul24_main, snippet
    for words in MUL24_SETS:
        expected = mul24_reference(words)
        for ch in expected:
            assert half_rtz(ch) == ch, (words, expected)
    for mode in ("no_mask_24", "i24_as_u24", "u24_as_i24", "hi_i24_as_u24", "hi_i24_bad_lo0", "hi_u24_as_i24"):
        assert any(mul24_reference(w, mode=mode) != mul24_reference(w) for w in MUL24_SETS), mode
    base_m24 = mul24_values()
    for bad in (
        "v_mul_i32_i24_e64 v21, -s46, s47",
        "v_mul_i32_i24_e64 v21, |s46|, s47",
        "v_mul_i32_i24_e64 v21, abs(s46), s47",
        "v_mul_i32_i24_e64 v21, s46, s47 clamp",
        "v_mul_u32_u24_e64 v23, s47, s44 clamp",
        "v_mul_hi_i32_i24_e64 v25, s46, s47 clamp",
        "v_mul_hi_u32_u24_e64 v27, s47, s44 clamp",
        "v_mul_i32_i24_sdwa v21, s46, s47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
        "v_mul_i32_i24_e64 v21, s46",
        "v_mul_u32_u24_e32 v256, s45, v29",
        "v_mul_hi_u32_u24_e64 v27, s47, 0x100000000",
    ):
        changed = [line.replace("v_mul_i32_i24_e64 v21, s46, s47", bad) for line in base_m24]
        assert changed != base_m24, bad
        must_reject(translate, changed)
    wait = base_m24.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_m24[:wait] + ["v_mul_i32_i24_e32 v12, 1, v28"] + base_m24[wait:])
    must_reject(translate, base_m24[:wait] + ["v_mul_hi_u32_u24_e64 v13, 1, 2"] + base_m24[wait:])
    load = next(i for i, line in enumerate(base_m24) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_m24.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_m24[:lgkm] + ["v_mul_u32_u24_e64 v20, s44, 2"] + base_m24[lgkm:])
    wqm = base_m24.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_m24[:wqm] + ["v_mul_hi_i32_i24_e64 v20, 1, 2"] + base_m24[wqm:])


def check_program_extent(directory, library):
    # A conditional jumps over the early exit. The alternate block starts
    # with a two-dword instruction (s_mov_b32 with a literal operand).
    branch = bytes.fromhex("010084bf")
    end = bytes.fromhex("000081bf")
    mov = bytes.fromhex("ff0380be 01000000")
    pad = bytes.fromhex("00009fbf")
    code = branch + end + mov + end
    padding = pad * 5
    base = 288 - 40
    extent = 40 + len(code) + len(padding)
    archive = directory / "bounded.rpf"

    def extract(body, size=extent, limit=base + extent):
        archive.write_bytes(envelope(body, directory_end=(size, limit)))
        return native_program(archive, "fixture", llvm=library)[1]

    jump_over_nop = bytes.fromhex("010082bf 000080bf")
    assert extract(jump_over_nop + code + padding[8:]) == jump_over_nop + code
    must_reject(extract, bytes.fromhex("010082bf 810380be") + code + padding[8:])
    assert extract(code + padding + end) == code  # marker in metadata
    # A terminator in metadata cannot stand in for one missing inside the section.
    must_reject(extract, bytes(len(code)) + padding + end)
    assert extract(code + padding + pad * 4, limit=0xdeadbeefdeadbeef) == code
    must_reject(extract, code + padding, extent, base + extent + 4)
    must_reject(extract, code + padding[:-4])
    must_reject(extract, code + padding, extent, 0xdeadbeefdeadbeef)
    must_reject(extract, code + padding[:4] + bytes(16) + end)
    must_reject(extract, bytes.fromhex("020084bf") + code[4:] + padding)  # target inside literal
    must_reject(extract, bytes.fromhex("000080bf") + code[4:] + padding)  # disconnected block
    must_reject(extract, bytes.fromhex("040084bf") + code[4:] + padding)  # target in padding
    must_reject(extract, branch + bytes.fromhex("ff0380be") + end + bytes.fromhex("000080bf") + end + padding)


def check_binary(directory, assembler, objcopy, library):
    check_program_extent(directory, library)
    fixtures = [(assembly(False), translate), (assembly(True), translate), (curve(), translate), (shaping(), translate), (select(), translate), (absolute_value(), translate),
                (rounding(), translate), (arithmetic(), translate), (with_scalar_constant_reads(), translate), (with_scalar_pair_reads(), translate),
                (truncate_values(), translate), (affine_coordinates(), translate), (ceiling(), translate),
                (scalar64(), translate), (sopk_scalar(), translate), (sopk_cmpk_cmovk(), translate_int)]
    fixtures.append((rgb_sample(), translate))
    fixtures.append((with_constants(TRIG_BODY), translate))
    fixtures.append((with_constants(CLASS_BODY), translate))
    fixtures.append((sext_values(), translate))
    fixtures.append((sop1_scalar_ops(), translate_int))
    fixtures.append((bfm_scalar_ops(), translate_int))
    fixtures.append((scalar_select64(), translate_int))
    for op in SCALAR_BFE64_OPS:
        fixtures.extend((program, translate_int) for program in (
            scalar_bfe64(op), scalar_bfe64(op, "0x80000000"), scalar_bfe64(op, control="0x400000"),
        ))
    fixtures.extend((scalar_pack_values(halves), translate) for halves in PACK_HALVES)
    fixtures.extend((scalar_bit_count_values(width), translate) for width in (32, 64))
    fixtures.extend((scalar_carry_chain(subtract), translate_int) for subtract in (False, True))
    fixtures.extend((scalar_leading_bit_values(width), translate) for width in (32, 64))
    fixtures.append((scalar_mul_hi_values(), translate))
    fixtures.append((with_constants(RECIPROCAL_BODY), translate))
    fixtures.append((bitwise_program(), translate))
    fixtures.append((fract_values(), translate))
    fixtures.extend([(frexp_mant_values(), translate), (ldexp_values(), translate)])
    fixtures.append((mul24_values(), translate))
    fixtures.append((three_way(), translate))
    fixtures.extend((predicate_group(first), translate) for first in range(0, 16, 4))
    fixtures.extend((int_predicate_group(kind, encoding, first), translate)
                    for kind in ("i32", "u32") for encoding in ("e32", "e64") for first in (0, 4))
    fixtures.append((unsigned_less_than(), translate))
    fixtures.extend((int16_group(kind, encoding, group), translate)
                    for kind in ("i16", "u16") for encoding in ("e32", "e64") for group in (0, 1))

    fixtures.extend((int_minmax_pair(encoding), translate) for encoding in ("e32", "e64"))
    fixtures.extend((int_minmax_triple(kind), translate) for kind in ("i32", "u32"))
    fixtures.extend((predicate_group(first, sdwa=True), translate) for first in range(0, 16, 4))
    fixtures.extend((cvt_int_fixture(encoding), translate) for encoding in ("e32", "e64"))
    fixtures.append((cvt_int_literals(), translate))
    fixtures.append((with_constants(CUBE_BODY), translate))
    fixtures.append((with_constants(MAD_BODY), translate))
    fixtures.extend((dual_program(case), translate_dual) for case in DUAL_CASES)
    fixtures.append((with_constants(MOVK_BODY), translate))
    fixtures.append((with_constants(HALF_BODY), translate))
    fixtures.append((with_constants(VOP3P_BODY), translate))
    fixtures.append((with_constants(FMA_MIX_BODY), translate))
    fixtures.append((with_constants(NARROW_BODY), translate))
    fixtures.append((mode_program(), translate))
    fixtures.append((dual_program(swap_weights=True), translate_dual))
    fixtures.extend((with_constants(body), translate) for body, *_ in E64_ARITH)
    fixtures.extend((predicate_group_e64(first), translate) for first in range(0, 16, 4))
    fixtures.append((scalar(), translate))
    fixtures.append((lod_zero_sample(), translate))
    fixtures.append((with_constants(LITERAL_FMA_BODY), translate))
    fixtures.append((with_constants(MAD_LITERAL_BODY), translate))
    fixtures.append((unpack_bytes(), translate))
    fixtures.append((reciprocal_iflag(), translate))
    fixtures.append((offset_nibbles(), translate))
    fixtures.append((legacy_multiply_add(), translate))
    fixtures.append((scheduling_hints(), translate))
    fixtures.extend((buffer_color(rg), translate_buffer_color) for rg in (False, True))
    fixtures.extend((unsigned_conversion(quad), translate_buffer_color) for quad in (False, True))
    fixtures.extend((attribute_color(rg), translate_attribute) for rg in (False, True))
    fixtures.append((with_constants(DPP_INT32_BODY), translate))
    fixtures.append((with_constants(DPP_BODY), translate))
    fixtures.extend((with_constants(body), translate) for body, *_ in SDWA_ARITH)
    fixtures.append((with_constants(PK_INT16_BODY), translate))
    fixtures.extend((with_constants(body), translate) for body in (SDWA_PRESERVE_BODY, SDWA_PAD_BODY))
    exp_sync_program = ["s_waitcnt_expcnt null, 0"] + assembly(False)[:-1] + ["s_waitcnt_expcnt null, 0", "s_endpgm"]
    fixtures.append((exp_sync_program, translate))
    fixtures.append((with_constants(DOT_BODY), translate, "gfx1030"))
    fixtures.append((uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000)), translate))
    fixtures.append((uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True)), translate))
    fixtures.append((uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True, clamp=True)), translate))
    fixtures.append((uncompressed_export(cvt_float_body(0x3fc00000, 0xc0300000)), translate_int))
    fixtures.extend((uncompressed_export(nc_integer_group(group)), translate_int) for group in (1, 2, 3))
    for index, entry in enumerate(fixtures):
        instructions, lower_fixture = entry[:2]
        cpu = entry[2] if len(entry) > 2 else "gfx1013"
        source = directory / "fixture.s"
        obj = directory / "fixture.o"
        binary = directory / "fixture.bin"
        source.write_text("\n".join(instructions) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", f"-mcpu={cpu}",
                        "-filetype=obj", source, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
        archive = directory / "fixture.rpf"
        archive.write_bytes(envelope(binary.read_bytes()))
        _, native = native_program(archive, "fixture")
        assert native == binary.read_bytes()
        assert lower_fixture(disassemble(native, library, cpu=cpu.encode())) == lower_fixture(instructions)
        if index == 0:
            check_container_selector(directory, native, library)


def check_ceiling_draws(directory, compiler, renderer, backend):
    instructions = ceiling()
    source = directory / "ceiling.hlsl"
    shader = directory / "ceiling.cso"
    case = directory / "ceiling.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)
    for a, b, c, d in ((-1.75, 1.75, -1.75, -1.75), (-2.0, 2.0, 0.0, 0.0),
                       (-0.75, 0.75, -0.75, -0.75), (2.25, -2.75, 2.25, 2.25)):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        # Python's integer results lose the sign of zero. The readback compares float bits.
        expected = (math.copysign(math.ceil(a), a),
                    math.copysign(math.ceil(-abs(b)), -1), math.ceil(abs(c)),
                    min(math.ceil(abs(d)), 1))
        write_case(case, constants, expected)
        subprocess.run(draw, check=True)

    floor = [line.replace("v_ceil_f32_e32", "v_floor_f32_e32") for line in instructions]
    source.write_text(translate(floor), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing ceiling with floor escaped detection")


def check_affine_lod_draw(directory: Path, compiler: str, renderer: str, backend: str) -> None:
    program = affine_coordinates()
    coordinate = f"float2(alu{program.index('v_fma_f32 v8, 0.5, v8, 0')}, uv.y)"
    source = translate(program)
    assert source.count(coordinate) == 1
    # Replace values on uncovered lanes with the quad's top-left coordinates before computing LOD.
    poor_gradient = (f"(uv.x + uv.y < 0.96875f ? {coordinate} : "
                     "(((floor(uv * 4.0f) * 2.0f + 0.5f) / 8.0f) + float2(0.5f, 0.0f))"
                     " * float2(params[0].z * 0.5f, 1.0f))")
    hlsl = directory / "quad_lod.hlsl"
    shader = directory / "quad_lod.cso"
    draw = [renderer, backend, "--check-pixel-quad-lod", shader]
    hlsl.write_text(source, encoding="ascii")
    compile_shader(compiler, hlsl, shader)
    subprocess.run(draw, check=True)
    hlsl.write_text(source.replace(coordinate, poor_gradient), encoding="ascii")
    compile_shader(compiler, hlsl, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The draw accepted a missing helper-lane gradient")
# v_min_f32/v_max_f32 on the least binary32 subnormal keep it under -denorm preserve and flush it
# to zero under -denorm ftz; v_mul_f32 by 1.0 folds to a move, so that channel exports identical
# bits under both modes and cannot detect the mode by itself.
DENORM_BODY = [
    "v_min_f32_e32 v12, s44, s45",
    "v_max_f32_e32 v13, s44, s46",
    "v_mul_f32_e32 v14, s44, 1.0",
]
LEAST_SUBNORMAL = struct.unpack("<f", struct.pack("<I", 1))[0]


def check_denorm_mode_draw(directory, compiler, renderer, backend):
    """Draw one program under both -denorm modes with mode-matched references."""
    source = directory / "denorm_mode.hlsl"
    shader = directory / "denorm_mode.cso"
    case = directory / "denorm_mode.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, LEAST_SUBNORMAL, 1.0, -1.0, 0.0)
    source.write_text(translate(uncompressed_export(DENORM_BODY)), encoding="ascii")
    passthrough = 1.0  # the untouched sample alpha
    preserve = (LEAST_SUBNORMAL, LEAST_SUBNORMAL, LEAST_SUBNORMAL, passthrough)
    ftz = (flush_ftz(LEAST_SUBNORMAL), flush_ftz(LEAST_SUBNORMAL), LEAST_SUBNORMAL, passthrough)
    write_case(case, constants, preserve)
    compile_shader(compiler, source, shader)
    subprocess.run(draw, check=True)
    # The folding channel expects the same raw subnormal under ftz, so this run passing proves the
    # folded move exports identical bits in both modes; the min and max channels expect the flush.
    write_case(case, constants, ftz)
    compile_shader(compiler, source, shader, mode="ftz")
    subprocess.run(draw, check=True)
    # The min and max channels differ between the modes, so the preserve build must fail the ftz
    # reference still on disk.
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A preserve-mode export matched the ftz reference")


# Packed binary16 operations on subnormal halves. s44 holds the least (0x0001) and largest (0x03ff)
# binary16 subnormals, s45 the least normal 0x0400 and -0, s46 0.5 in both halves and s47 -0x0401 in
# both. The required row is the add: two normal inputs whose sum is the subnormal 0x8001, the case
# DXIL's fadd FTZ text requires to flush. max keeps a subnormal input and the multiply makes the
# subnormal 0x0200 from normal inputs (supplementary rows), and the fma rounds a subnormal and a normal
# result to nearest even. v14 exports 0x80000200, a binary32 subnormal as a word, so the draw reads
# back through the integer target, where no float path can flush the packed bits.
F16_DENORM_BODY = [
    "v_pk_add_f16 v12, s45, s47",
    "v_pk_max_f16 v13, s44, s47",
    "v_pk_mul_f16 v14, s45, s46",
    "v_pk_fma_f16 v15, s44, s46, s44",
]
F16_DENORM_WORDS = (0x03ff0001, 0x80000400, 0x38003800, 0x84018401)


def f16_denorm_words(flush):
    """Reference words for F16_DENORM_BODY, optionally flushing binary16 subnormal inputs and results."""
    def half(word):
        if flush and word & 0x7c00 == 0:
            word &= 0x8000
        return struct.unpack("<e", struct.pack("<H", word))[0]

    def pack(value):
        word, = struct.unpack("<H", struct.pack("<e", value))
        return word & 0x8000 if flush and word & 0x7c00 == 0 else word

    a, b, c, d = F16_DENORM_WORDS
    ops = (lambda x, y, z: x + y, lambda x, y, z: max(x, y), lambda x, y, z: x * y, lambda x, y, z: x * y + z)
    sources = ((b, d, 0), (a, d, 0), (b, c, 0), (a, c, a))
    words = []
    for op, (x, y, z) in zip(ops, sources):
        # Python doubles hold these products and sums exactly, so one rounding to binary16 matches the shader.
        lanes = [pack(op(half(x >> s & 0xffff), half(y >> s & 0xffff), half(z >> s & 0xffff))) for s in (0, 16)]
        words.append(lanes[0] | lanes[1] << 16)
    return tuple(words)


def check_f16_denorm_draw(directory, compiler, renderer, backend):
    """Draw packed binary16 subnormals under both -denorm modes; the f32 flag must not change them."""
    source = directory / "f16_denorm.hlsl"
    shader = directory / "f16_denorm.cso"
    case = directory / "f16_denorm.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *F16_DENORM_WORDS)
    source.write_text(lower(uncompressed_export(F16_DENORM_BODY), 20, 24, [1, 2, 3, 4],
                            constant_descriptor=64, integer_export=True), encoding="ascii")

    def as_floats(words):
        return tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)

    preserve = f16_denorm_words(flush=False)
    write_case(case, constants, as_floats(preserve), integer_rt=True)
    for mode in ("preserve", "ftz"):
        compile_shader(compiler, source, shader, mode=mode)
        subprocess.run(draw, check=True)
    # A reference that flushes binary16 subnormals differs from the preserve one in every channel,
    # so the ftz build still on disk must fail it.
    flushed = f16_denorm_words(flush=True)
    assert all(p != f for p, f in zip(preserve, flushed)), (preserve, flushed)
    write_case(case, constants, as_floats(flushed), integer_rt=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A binary16 export matched the flushed f16 reference")


def check_nan_class_draw(directory, compiler, renderer, backend):
    """Check v_add_f32 NaN results by class on masked channels; finite channels stay strict."""
    source = directory / "nan_class.hlsl"
    shader = directory / "nan_class.cso"
    case = directory / "nan_class.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    words = (0x7fc00000, 0x7f800001, 0xffc00003, 0x3f800000)
    values = [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *words)
    program = uncompressed_export([
        "v_add_f32_e32 v12, s44, s47",
        "v_add_f32_e32 v13, s45, s46",
        "v_add_f32_e32 v14, s45, s45",
    ])
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_case(case, constants, (*values[:3], 1.0), nan_mask=0b0111)
    subprocess.run(draw, check=True)
    # A masked channel whose shader output is finite must fail on every backend.
    finite = [line.replace("v12, s44, s47", "v12, s47, s47") for line in program]
    source.write_text(translate(finite), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A finite export satisfied a NaN-class channel")
    # A strict finite channel against a NaN export must also fail on every backend.
    nan_alpha = program[:-2] + ["v_add_f32_e32 v15, s45, s46"] + program[-2:]
    source.write_text(translate(nan_alpha), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A NaN export satisfied the strict finite channel")


def check_integer_readback_draw(directory, compiler, renderer, backend):
    """Round-trip v_mov_b32 bit copies bit-for-bit through the integer readback target."""
    source = directory / "integer_readback.hlsl"
    shader = directory / "integer_readback.cso"
    case = directory / "integer_readback.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    words = (0x7fc00001, 0xffc00003, 0x00000001, 0x80000000)
    constants = bytearray(256)
    struct.pack_into("<I", constants, 12, words[0])
    program = uncompressed_export([
        "v_mov_b32_e32 v12, s44",
        f"v_mov_b32_e32 v13, {words[1]:#x}",
        f"v_mov_b32_e32 v14, {words[2]:#x}",
        f"v_mov_b32_e32 v15, {words[3]:#x}",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    expected = tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)
    write_case(case, constants, expected, integer_rt=True)
    subprocess.run(draw, check=True)
    corrupted = list(expected)
    corrupted[2] = struct.unpack("<f", struct.pack("<I", words[2] ^ 3))[0]
    write_case(case, constants, corrupted, integer_rt=True)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A corrupted expected word escaped the integer readback")


def check_subrev_u32_draw(directory, compiler, renderer, backend):
    source = directory / "subrev_u32.hlsl"
    shader = directory / "subrev_u32.cso"
    program = uncompressed_export([
        "v_mov_b32_e32 v18, s45",
        "v_subrev_nc_u32_e32 v12, s44, v18",
        "v_sub_nc_u32_e32 v13, s44, v18",
        "v_mov_b32_e32 v19, s44",
        "v_subrev_nc_u32_e32 v14, v19, v18",
        "v_mov_b32_e32 v15, 0",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    # Check normal subtraction and unsigned wrap with exact RGBA32_UINT readback.
    for name, inputs, words, status in (
        ("forward-oracle", (1, 0), (1, 0xffffffff), "mismatch"),
        ("wrap", (1, 0), (0xffffffff, 1), "pass"),
        ("normal", (3, 7), (4, 0xfffffffc), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"subrev-u32-{name}.case"
        a, b = inputs
        two_vgprs = (b - a) % (1 << 32)
        case.write_bytes(constants + struct.pack("<4I", *words, two_vgprs, 0) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)


def binary32_normal_or_zero(word):
    """True for a normal binary32 or a signed zero: no NaN, infinity or subnormal pattern."""
    exponent = (word >> 23) & 0xff
    return 0 < exponent < 0xff or word & 0x7fffffff == 0


def check_integer_nc_draws(directory, compiler, renderer, backend):
    """Draw the 32-bit no-carry add/sub forms through the integer readback target and compare the words.

    Group 1 and 2 run on every row of NC_INT_ROWS and group 3 on the third row: 11 draws from 3 compiles.
    The expected words follow the sub and subrev operand order of the translator (see the docs); on a desktop
    GPU the draws establish that the translated shader computes exactly that, not what the console does.
    """
    source = directory / "integer_nc.hlsl"
    shader = directory / "integer_nc.cso"
    case = directory / "integer_nc.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]

    def words_as_floats(words):
        return tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)

    def expected_words(group, a, b):
        words = tuple(word & 0xffffffff for word in NC_INT_RESULTS[group](a, b))
        # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
        assert all(binary32_normal_or_zero(word) for word in (a, b, *words)), (group, hex(a), hex(b))
        return words

    for group, rows in ((1, NC_INT_ROWS), (2, NC_INT_ROWS), (3, NC_INT_ROWS[2:3])):
        source.write_text(translate_int(uncompressed_export(nc_integer_group(group))), encoding="ascii")
        compile_shader(compiler, source, shader)
        for a, b in rows:
            constants = bytearray(256)
            struct.pack_into("<II", constants, 12, a, b)
            words = expected_words(group, a, b)
            write_case(case, constants, words_as_floats(words), integer_rt=True)
            subprocess.run(draw, check=True)
        if group == 1:
            # Negative control: the third row has A - B != B - A, so swapped sub and subrev words must mismatch.
            a, b = NC_INT_ROWS[2]
            constants = bytearray(256)
            struct.pack_into("<II", constants, 12, a, b)
            words = expected_words(1, a, b)
            assert words[1] != words[2]
            swapped = (words[0], words[2], words[1], words[3])
            write_case(case, constants, words_as_floats(swapped), integer_rt=True)
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("Swapped sub and subrev words escaped the integer readback")
    return 0


def check_lshl_add_u32_draw(directory, compiler, renderer, backend):
    source = directory / "lshl_add_u32.hlsl"
    shader = directory / "lshl_add_u32.cso"
    program = uncompressed_export([
        "s_lshl1_add_u32 s60, s44, s45",
        "s_lshl2_add_u32 s61, s44, s45",
        "s_lshl3_add_u32 s62, s44, s45",
        "s_lshl4_add_u32 s63, s44, s45",
        "v_mov_b32_e32 v12, s60",
        "v_mov_b32_e32 v13, s61",
        "v_mov_b32_e32 v14, s62",
        "v_mov_b32_e32 v15, s63",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    # Check normal shifted addition, shift-overflow wrap, and add-overflow wrap.
    for name, inputs, words, status in (
        ("oracle-mismatch", (3, 5), (11, 17, 29, 999), "mismatch"),
        ("normal", (3, 5), (11, 17, 29, 53), "pass"),
        ("shift-wrap", (0x80000001, 10), (12, 14, 18, 26), "pass"),
        ("add-wrap", (1, 0xffffffff), (1, 3, 7, 15), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"lshl-add-u32-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)

    # Check SCC carry-out across selective shift-overflow and add-overflow conditions.
    scc_source = directory / "lshl_add_scc.hlsl"
    scc_shader = directory / "lshl_add_scc.cso"
    scc_program = uncompressed_export([
        "s_lshl1_add_u32 s60, s44, s45",
        "s_cselect_b32 s71, 1, 0",
        "s_lshl2_add_u32 s61, s44, s45",
        "s_cselect_b32 s72, 1, 0",
        "s_lshl3_add_u32 s62, s44, s45",
        "s_cselect_b32 s73, 1, 0",
        "s_lshl4_add_u32 s63, s44, s45",
        "s_cselect_b32 s74, 1, 0",
        "v_mov_b32_e32 v12, s71",
        "v_mov_b32_e32 v13, s72",
        "v_mov_b32_e32 v14, s73",
        "v_mov_b32_e32 v15, s74",
    ])
    scc_source.write_text(lower(scc_program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                                integer_export=True), encoding="ascii")
    compile_shader(compiler, scc_source, scc_shader)
    scc_cases = []
    for name, inputs, words, status in (
        ("oracle-mismatch", (1, 1), (1, 0, 0, 0), "mismatch"),
        ("none", (1, 1), (0, 0, 0, 0), "pass"),
        ("carry-4", (0x10000000, 0), (0, 0, 0, 1), "pass"),
        ("carry-3-4", (0x20000000, 0), (0, 0, 1, 1), "pass"),
        ("carry-2-3-4", (0x40000000, 0), (0, 1, 1, 1), "pass"),
        ("carry-all-shift", (0x80000000, 0), (1, 1, 1, 1), "pass"),
        ("carry-all-add", (1, 0xfffffffe), (1, 1, 1, 1), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"lshl-add-scc-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        scc_cases.append((case, status))
    run_pixel_batch(renderer, backend, scc_shader, scc_cases)


# v_cvt_f32_u32_e32 and v_cvt_f32_i32_e32 convert one 32-bit word to binary32, reading it unsigned
# or two's-complement. The inputs are shader literals moved through v_mov_b32 bit copies, not
# constant-buffer words (#360): the pairs cover zero and one, the last exactly representable integer
# 2**24, the ties 2**24+1 and 2**25+2, the non-tie neighbors 2**25+1 and 2**25+3 of a midpoint, the
# ties 2**24+3 and 2**25-1 that round up (3 mod 4, so clearing their low bit changes the result), and
# the extreme words 0xFFFFFFFF, 0x80000000 and 0x7FFFFFFF.
CVT_INT_WORDS = (
    (0, 1),
    (2**24, 2**24 + 1),
    (2**25 + 1, 2**25 + 3),
    (2**25 + 2, 0x7FFFFFFF),
    (2**24 + 3, 2**25 - 1),
    (0xFFFFFFFF, 0x80000000),
)


def cvt_int_body(w0, w1, *, swap=False, e64=False, clamp=False):
    """Convert each literal word twice: unsigned into the lower channel of its pair, signed into the
    upper. Swapping the conversions changes every channel whose word has the sign bit set. The e64
    encoding additionally carries the clamp modifier; the default e32 encoding carries neither, so
    asking for clamp without e64 builds the text the translator must refuse."""
    unsigned, signed = ("i32", "u32") if swap else ("u32", "i32")
    encoding = "_e64" if e64 else "_e32"
    suffix = " clamp" if clamp else ""
    body = []
    for dst, word in ((12, w0), (13, w0), (14, w1), (15, w1)):
        op = unsigned if dst % 2 == 0 else signed
        body += [f"v_mov_b32_e32 v{dst}, {word:#x}", f"v_cvt_f32_{op}{encoding} v{dst}, v{dst}{suffix}"]
    return body


def check_cvt_int_f32_draw(directory, compiler, renderer, backend):
    source = directory / "cvt_int_f32.hlsl"
    shader = directory / "cvt_int_f32.cso"
    case = directory / "cvt_int_f32.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytes(256)
    # The tie words sit exactly between two representable values and must round to the even one;
    # a half-up or half-away rule would export a different word, so the rows discriminate rounding.
    assert (u32f(2**24 + 1), u32f(2**25 + 2)) == (2.0**24, 2.0**25)
    for w0, w1 in CVT_INT_WORDS:
        program = uncompressed_export(cvt_int_body(w0, w1))
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        write_case(case, constants, (u32f(w0), i32f(w0), u32f(w1), i32f(w1)))
        subprocess.run(draw, check=True)
    # The round-up ties of [2**24, 2**25) must not be read with their low bit cleared: that lands on the
    # representable value 2 below the nearest-even result.
    w0, w1 = 2**24 + 3, 2**25 - 1
    assert (w0, w1) in CVT_INT_WORDS
    assert (u32f(w0), u32f(w1)) == (2.0**24 + 4, 2.0**25)
    assert (u32f(w0 & ~1), u32f(w1 & ~1)) == (2.0**24 + 2, 2.0**25 - 2)
    source.write_text(translate(uncompressed_export(cvt_int_body(w0, w1))), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_case(case, constants, (u32f(w0 & ~1), i32f(w0), u32f(w1 & ~1), i32f(w1)))
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding a midpoint u32 word from its low-bit-cleared neighbor escaped detection")
    # The last pair's words all have the sign bit set, so swapping the two conversions must be caught.
    w0, w1 = CVT_INT_WORDS[-1]
    assert u32f(w0) != i32f(w0) and u32f(w1) != i32f(w1)
    source.write_text(translate(uncompressed_export(cvt_int_body(w0, w1, swap=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping the unsigned and signed conversions escaped detection")


def check_cvt_int_f32_e64_draw(directory, compiler, renderer, backend):
    source = directory / "cvt_int_f32_e64.hlsl"
    shader = directory / "cvt_int_f32_e64.cso"
    case = directory / "cvt_int_f32_e64.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytes(256)
    # The plain e64 encoding performs the same conversion as e32, so the same word pairs and
    # nearest-even references prove the two encodings agree, ties included.
    for w0, w1 in CVT_INT_WORDS:
        program = uncompressed_export(cvt_int_body(w0, w1, e64=True))
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        write_case(case, constants, (u32f(w0), i32f(w0), u32f(w1), i32f(w1)))
        subprocess.run(draw, check=True)
    # SGPR source row: constant-buffer words with the sign bit set, read directly by the unclamped e64
    # conversions. The words are the normal binary32 patterns of -2.0 and -1.0, so the load carries no
    # NaN or denormal. Reading the unsigned conversion as signed must be caught.
    w0, w1 = 0xC0000000, 0xBF800000
    scalar_constants = bytearray(256)
    struct.pack_into("<2I", scalar_constants, 12, w0, w1)
    scalar_constants = bytes(scalar_constants)
    assert (u32f(w0), u32f(w1)) == (3221225472.0, 3212836864.0)
    assert (i32f(w0), i32f(w1)) == (-1073741824.0, -1082130432.0)
    program = uncompressed_export(["v_cvt_f32_u32_e64 v12, s44", "v_cvt_f32_i32_e64 v13, s44",
                                   "v_cvt_f32_u32_e64 v14, s45", "v_cvt_f32_i32_e64 v15, s45"])
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_case(case, scalar_constants, (u32f(w0), i32f(w0), u32f(w1), i32f(w1)))
    subprocess.run(draw, check=True)
    write_case(case, scalar_constants, (i32f(w0), i32f(w0), i32f(w1), i32f(w1)))
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading an SGPR word as signed in v_cvt_f32_u32_e64 escaped detection")
    # Clamp rows: the modifier maps the conversion through min(1.0, max(0.0, .)), computed from
    # the same u32f/i32f references. The words cover the interior 0 and 1, values past 1 (2,
    # 0xFFFFFFFF and 0x7FFFFFFF, which rounds to 2147483648.0 before clamping) and the negative
    # signed words 0xFFFFFFFF and 0x80000000, which clamp to 0.
    for w0, w1 in ((0, 1), (2, 0xFFFFFFFF), (0x80000000, 0x7FFFFFFF)):
        program = uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True))
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        expected = tuple(min(1.0, max(0.0, ref(word)))
                         for word, ref in ((w0, u32f), (w0, i32f), (w1, u32f), (w1, i32f)))
        write_case(case, constants, expected)
        subprocess.run(draw, check=True)
    # Dropping clamp must fail the clamped reference still on disk: without it u32 of 2 exports
    # 2.0 where the reference holds 1.0.
    write_case(case, constants, (1.0, 1.0, 1.0, 0.0))
    source.write_text(translate(uncompressed_export(cvt_int_body(2, 0xFFFFFFFF, e64=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Dropping clamp from the e64 conversion escaped detection")
    # Clamped VGPR row whose positive words have a zero low half: the signed conversion clamps them to
    # 1.0, and any reading that drops bits 16 to 30 gives 0 and clamps to 0.0.
    w0, w1 = 0x00010000, 0x7FFF0000
    program = uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True))
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    expected = tuple(min(1.0, max(0.0, ref(word)))
                     for word, ref in ((w0, u32f), (w0, i32f), (w1, u32f), (w1, i32f)))
    assert expected == (1.0, 1.0, 1.0, 1.0)
    write_case(case, constants, expected)
    subprocess.run(draw, check=True)
    masked = tuple(min(1.0, max(0.0, ref(word)))
                   for word, ref in ((w0, u32f), (w0 & 0x8000FFFF, i32f), (w1, u32f), (w1 & 0x8000FFFF, i32f)))
    assert masked == (1.0, 0.0, 1.0, 0.0)
    write_case(case, constants, masked)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Dropping the high bits of a clamped signed conversion escaped detection")
    # The last pair's words all have the sign bit set, so under clamp swapping the two conversions
    # exchanges 1.0 and 0.0 on every channel: u32 clamps to 1 where i32 of the same word clamps to 0.
    w0, w1 = CVT_INT_WORDS[-1]
    program = uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True))
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_case(case, constants, tuple(min(1.0, max(0.0, ref(word)))
                                      for word, ref in ((w0, u32f), (w0, i32f), (w1, u32f), (w1, i32f))))
    subprocess.run(draw, check=True)
    source.write_text(translate(uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True, swap=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping the clamped unsigned and signed conversions escaped detection")


def cvt_u32_ref(word):
    """Reference for v_cvt_u32_f32_e32: the result word for the binary32 bit pattern word."""
    x = struct.unpack("<f", struct.pack("<I", word))[0]
    if math.isnan(x) or x <= 0.0:
        return 0
    if x >= 2.0**32:
        return 0xFFFFFFFF
    return math.trunc(x)


def cvt_i32_ref(word):
    """Reference for v_cvt_i32_f32_e32: the two's-complement result word for the binary32 pattern word."""
    x = struct.unpack("<f", struct.pack("<I", word))[0]
    if math.isnan(x):
        return 0
    if x <= -2.0**31:
        return 0x80000000
    if x >= 2.0**31:
        return 0x7FFFFFFF
    return math.trunc(x) & 0xFFFFFFFF


# (word, u32 result, i32 result), consecutive pairs form the draw rows.
CVT_FLOAT_WORDS = (
    (0x00000000, 0x00000000, 0x00000000),
    (0x80000000, 0x00000000, 0x00000000),
    (0x3fc00000, 0x00000001, 0x00000001),
    (0xc0300000, 0x00000000, 0xfffffffe),
    (0xbf000000, 0x00000000, 0x00000000),
    (0xbf800000, 0x00000000, 0xffffffff),
    (0x7fc00000, 0x00000000, 0x00000000),
    (0xffc00001, 0x00000000, 0x00000000),
    (0x7f800000, 0xffffffff, 0x7fffffff),
    (0xff800000, 0x00000000, 0x80000000),
    (0x4f800000, 0xffffffff, 0x7fffffff),
    (0x4f7fffff, 0xffffff00, 0x7fffffff),
    (0x4f000000, 0x80000000, 0x7fffffff),
    (0x4effffff, 0x7fffff80, 0x7fffff80),
    (0xcf000000, 0x00000000, 0x80000000),
    (0xcf000001, 0x00000000, 0x80000000),
    (0x00000001, 0x00000000, 0x00000000),
    (0x3f7fffff, 0x00000000, 0x00000000),
)


def cvt_float_body(w0, w1, *, swap=False):
    """Convert each literal binary32 word twice: to u32 in the even registers, to i32 in the odd ones.
    Swapping the conversions changes the channels whose u32 and i32 results differ."""
    even, odd = ("i32", "u32") if swap else ("u32", "i32")
    body = []
    for dst, word in ((12, w0), (13, w0), (14, w1), (15, w1)):
        op = even if dst % 2 == 0 else odd
        body += [f"v_mov_b32_e32 v{dst}, {word:#x}", f"v_cvt_{op}_f32_e32 v{dst}, v{dst}"]
    return body


def as_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def check_cvt_float_int_draw(directory, compiler, renderer, backend):
    assert F32_DENORM_MODE == "preserve"
    source = directory / "cvt_float_int.hlsl"
    shader = directory / "cvt_float_int.cso"
    case = directory / "cvt_float_int.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytes(256)

    def write_words(words):
        # NaN payload guard: the float round trip must keep every expected word.
        for word in words:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(case, constants, tuple(as_float(word) for word in words), integer_rt=True)

    for word, u32, i32 in CVT_FLOAT_WORDS:
        assert (cvt_u32_ref(word), cvt_i32_ref(word)) == (u32, i32), hex(word)
    for (w0, u0, i0), (w1, u1, i1) in zip(CVT_FLOAT_WORDS[0::2], CVT_FLOAT_WORDS[1::2]):
        source.write_text(translate_int(uncompressed_export(cvt_float_body(w0, w1))), encoding="ascii")
        compile_shader(compiler, source, shader)
        write_words((u0, i0, u1, i1))
        subprocess.run(draw, check=True)

    # SGPR source: normal binary32 constants read through the buffer load.
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, 3.75, -3.75, 1e10, -1e10)
    constants = bytes(constants)
    program = uncompressed_export(["v_cvt_u32_f32_e32 v12, s44", "v_cvt_i32_f32_e32 v13, s45",
                                   "v_cvt_u32_f32_e32 v14, s46", "v_cvt_i32_f32_e32 v15, s47"])
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_words((3, 0xfffffffd, 0xffffffff, 0x80000000))
    subprocess.run(draw, check=True)

    # Literal source: hex binary32 operands outside the inline-constant set, converted directly. The
    # unsigned values exceed 1.0 and the signed ones are negative below -1.0, so clamping the unsigned
    # operand or taking the magnitude of the signed one changes every channel.
    constants = bytes(256)
    literals = (0x40600000, 0xc0300000, 0x4f7fffff, 0xc0500000)
    program = uncompressed_export([f"v_cvt_u32_f32_e32 v12, {literals[0]:#010x}",
                                   f"v_cvt_i32_f32_e32 v13, {literals[1]:#010x}",
                                   f"v_cvt_u32_f32_e32 v14, {literals[2]:#010x}",
                                   f"v_cvt_i32_f32_e32 v15, {literals[3]:#010x}"])
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    expected = (cvt_u32_ref(literals[0]), cvt_i32_ref(literals[1]),
                cvt_u32_ref(literals[2]), cvt_i32_ref(literals[3]))
    assert expected == (3, 0xfffffffe, 0xffffff00, 0xfffffffd)
    write_words(expected)
    subprocess.run(draw, check=True)
    one = 0x3f800000
    mutated = (cvt_u32_ref(one), cvt_i32_ref(literals[1] & 0x7fffffff),
               cvt_u32_ref(one), cvt_i32_ref(literals[3] & 0x7fffffff))
    assert mutated == (1, 2, 1, 3)
    write_words(mutated)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Clamping or dropping the sign of a literal float-to-word operand escaped detection")

    # Negative control: swapped conversions against the unswapped expectation. Channels v14 and v15
    # hold -1.0, which converts to 0 unsigned and 0xffffffff signed.
    constants = bytes(256)
    w0, w1 = 0xbf000000, 0xbf800000
    assert cvt_u32_ref(w1) != cvt_i32_ref(w1)
    source.write_text(translate_int(uncompressed_export(cvt_float_body(w0, w1, swap=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_words((0, 0, 0, 0xffffffff))
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping the float-to-word conversions escaped detection")


def check_int_minmax_draws(directory, compiler, renderer, backend):
    source = directory / "int_minmax.hlsl"
    shader = directory / "int_minmax.cso"
    case = directory / "int_minmax.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]

    def order_key(value, signed):
        return struct.unpack("<i" if signed else "<I", struct.pack("<f", value))[0]

    def pick_pair(a, b):
        return (
            min(a, b, key=lambda v: order_key(v, True)),
            max(a, b, key=lambda v: order_key(v, True)),
            min(a, b, key=lambda v: order_key(v, False)),
            max(a, b, key=lambda v: order_key(v, False)),
        )

    def pick_triple(a, b, c, signed):
        ordered = sorted((a, b, c), key=lambda v: order_key(v, signed))
        tied = sorted((c, b, b), key=lambda v: order_key(v, signed))[1]
        return ordered[0], ordered[2], ordered[1], tied

    # Exact binary16 floats whose 32-bit patterns reverse order across signed, unsigned, and float.
    for encoding in ("e32", "e64"):
        source.write_text(translate(int_minmax_pair(encoding)), encoding="ascii")
        compile_shader(compiler, source, shader)
        for a, b in ((-2.0, 1.0), (-2.0, -1.0)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, a, b, 0.0, 0.0)
            case.write_bytes(constants + struct.pack("<4f", *pick_pair(a, b)) * 64)
            subprocess.run(draw, check=True)

    swapped = [line.replace("v_min_i32_e32", "v_min_u32_e32") for line in int_minmax_pair("e32")]
    source.write_text(translate(swapped), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -2.0, 1.0, 0.0, 0.0)
    case.write_bytes(constants + struct.pack("<4f", *pick_pair(-2.0, 1.0)) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_min_i32 with v_min_u32 escaped detection")

    for kind in ("i32", "u32"):
        source.write_text(translate(int_minmax_triple(kind)), encoding="ascii")
        compile_shader(compiler, source, shader)
        for a, b, c in ((-2.0, -1.0, 1.0), (0.5, -0.5, 4.0)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, a, b, c, 0.0)
            case.write_bytes(constants + struct.pack("<4f", *pick_triple(a, b, c, kind == "i32")) * 64)
            subprocess.run(draw, check=True)

    wrong_med = [line.replace("v_med3_i32", "v_med3_u32") for line in int_minmax_triple("i32")]
    source.write_text(translate(wrong_med), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -2.0, -1.0, 1.0, 0.0)
    case.write_bytes(constants + struct.pack("<4f", *pick_triple(-2.0, -1.0, 1.0, True)) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_med3_i32 with v_med3_u32 escaped detection")


def check_cvt_int_draws(directory, compiler, renderer, backend):
    source = directory / "cvt_int.hlsl"
    shader = directory / "cvt_int.cso"
    case = directory / "cvt_int.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]

    # Two distinct constant sets: Set 1 tests negative numbers and values > 1.0;
    # Set 2 tests boundary zero, one, and an exact integer (1000).
    sets = [
        ((-20, 15, -50, 200), [-20.0, 15.0, -50.0, 200.0], [-20.0, 1.0, 0.0, 200.0]),
        ((0, 1, 1000, 0), [0.0, 1.0, 1000.0, 0.0], [0.0, 1.0, 1.0, 0.0]),
    ]
    for _, e32_vals, e64_vals in sets:
        for x in (*e32_vals, *e64_vals):
            assert half_rtz(x) == x, x

    # e32: unclamped conversions of SGPRs and VGPR.
    source.write_text(translate(cvt_int_fixture("e32")), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for (c0, c1, c2, c3), expected, _ in sets:
        constants = bytearray(256)
        struct.pack_into("<4i", constants, 12, c0, c1, c2, c3)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    # Negative control: swapping signed v_cvt_f32_i32 with unsigned v_cvt_f32_u32 must fail.
    swapped = [line.replace("v_cvt_f32_i32", "v_cvt_f32_u32") for line in cvt_int_fixture("e32")]
    source.write_text(translate(swapped), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    constants = bytearray(256)
    struct.pack_into("<4i", constants, 12, -20, 15, -50, 200)
    case.write_bytes(constants + struct.pack("<4f", *sets[0][1]) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_cvt_f32_i32 with v_cvt_f32_u32 escaped detection")

    # e64: includes clamp modifier (15.0 clamped -> 1.0; -50.0 clamped -> 0.0).
    source.write_text(translate(cvt_int_fixture("e64")), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    for (c0, c1, c2, c3), _, expected in sets:
        constants = bytearray(256)
        struct.pack_into("<4i", constants, 12, c0, c1, c2, c3)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    # Negative control: dropping clamp on e64 must fail.
    unclamped = [line.replace(" clamp", "") for line in cvt_int_fixture("e64")]
    source.write_text(translate(unclamped), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    constants = bytearray(256)
    struct.pack_into("<4i", constants, 12, -20, 15, -50, 200)
    case.write_bytes(constants + struct.pack("<4f", *sets[0][2]) * 64)
    wrong_clamp = subprocess.run(draw, capture_output=True, text=True)
    if wrong_clamp.returncode != 1 or "pixel_mismatch" not in wrong_clamp.stderr:
        raise AssertionError("Dropping clamp on v_cvt_f32_u32_e64/v_cvt_f32_i32_e64 escaped detection")

    # Direct literal operands with and without clamp.
    literal_expected = [-7.0, 25.0, 0.0, 1.0]
    source.write_text(translate(cvt_int_literals()), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    case.write_bytes(constants + struct.pack("<4f", *literal_expected) * 64)
    subprocess.run(draw, check=True)

    # Negative control: swapping signed literal to unsigned must report pixel mismatch.
    mutated_lit = [line.replace("v_cvt_f32_i32_e32 v12, -7", "v_cvt_f32_u32_e32 v12, -7")
                   for line in cvt_int_literals()]
    source.write_text(translate(mutated_lit), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong_lit = subprocess.run(draw, capture_output=True, text=True)
    if wrong_lit.returncode != 1 or "pixel_mismatch" not in wrong_lit.stderr:
        raise AssertionError("Literal signed-to-unsigned conversion mismatch escaped detection")

    # Negative control: dropping clamp on negative literal must report pixel mismatch.
    mutated_clamp = [line.replace("v_cvt_f32_i32_e64 v14, -100 clamp", "v_cvt_f32_i32_e64 v14, -100")
                     for line in cvt_int_literals()]
    source.write_text(translate(mutated_clamp), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong_lit_clamp = subprocess.run(draw, capture_output=True, text=True)
    if wrong_lit_clamp.returncode != 1 or "pixel_mismatch" not in wrong_lit_clamp.stderr:
        raise AssertionError("Literal clamp removal escaped detection")


def check_frexp_ldexp_draws(directory, compiler, renderer, backend):
    source = directory / "frexp_ldexp.hlsl"
    shader = directory / "frexp_ldexp.cso"
    case = directory / "frexp_ldexp.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]

    source.write_text(translate(frexp_mant_values()), encoding="ascii")
    compile_shader(compiler, source, shader)
    for a, b, c, d in ((3.0, 6.0, -0.25, -3.0), (-0.0, 0.0, 0.75, 12.0)):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        expected = (
            math.frexp(a)[0],
            math.frexp(-abs(b))[0],
            math.frexp(abs(c))[0],
            min(1.0, max(0.0, math.frexp(d)[0])),
        )
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    wrong_frexp = [line.replace("v_frexp_mant_f32_e32", "v_mov_b32_e32") for line in frexp_mant_values()]
    source.write_text(translate(wrong_frexp), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, 3.0, 6.0, -0.25, -3.0)
    case.write_bytes(constants + struct.pack("<4f", 0.75, -0.75, 0.5, 0.0) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_frexp_mant_f32 with v_mov_b32 escaped detection")

    source.write_text(translate(ldexp_values()), encoding="ascii")
    compile_shader(compiler, source, shader)
    for a, exp_i, c, d in ((0.75, 3, 6.0, 0.25), (-0.0, 10, -2.0, -0.5), (1.5, -2, 0.5, 0.0625)):
        constants = bytearray(256)
        struct.pack_into("<fiff", constants, 12, a, exp_i, c, d)
        expected = (
            math.ldexp(a, exp_i),
            math.ldexp(-abs(a), 2),
            math.ldexp(c, -2),
            min(1.0, max(0.0, math.ldexp(d, 3))),
        )
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    wrong_ldexp = [line.replace("v_ldexp_f32 v14, s46, -2", "v_ldexp_f32 v14, s46, 2") for line in ldexp_values()]
    source.write_text(translate(wrong_ldexp), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<fiff", constants, 12, 0.75, 3, 6.0, 0.25)
    case.write_bytes(constants + struct.pack("<4f", 6.0, -3.0, 1.5, 1.0) * 64)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Inverting the v_ldexp_f32 exponent sign escaped detection")


MUL24_BODY = [
    "v_mov_b32_e32 v28, s45",
    "v_mov_b32_e32 v29, s46",
    "v_mov_b32_e32 v30, s47",
    "v_mov_b32_e32 v31, s44",
    "v_mul_i32_i24_e32 v20, s44, v28",
    "v_mul_i32_i24_e64 v21, s46, s47",
    "v_mul_u32_u24_e32 v22, s45, v29",
    "v_mul_u32_u24_e64 v23, s47, s44",
    "v_mul_hi_i32_i24_e32 v24, s44, v28",
    "v_mul_hi_i32_i24_e64 v25, s46, s47",
    "v_mul_hi_u32_u24_e32 v26, s45, v29",
    "v_mul_hi_u32_u24_e64 v27, s47, s44",
    "v_mov_b32_e32 v16, 0x2468ace1",
    *[
        line
        for reg in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27")
        for line in (
            "v_lshlrev_b32_e32 v17, 5, v16",
            "v_lshrrev_b32_e32 v18, 27, v16",
            "v_or_b32_e32 v16, v17, v18",
            f"v_xor_b32_e32 v16, {reg}, v16",
            "v_add_nc_u32_e32 v16, 0x9e3779b9, v16",
        )
    ],
    "v_cvt_f32_ubyte0_e32 v12, v16",
    "v_cvt_f32_ubyte1_e32 v13, v16",
    "v_cvt_f32_ubyte2_e32 v14, v16",
    "v_cvt_f32_ubyte3_e32 v15, v16",
]

MUL24_SETS = (
    (0xab800000, 0xcd000200, 0xef7fffff, 0x12800000),
    (0xff800000, 0xee800000, 0xddffffff, 0xccffffff),
    (0x7f000000, 0x80ffffff, 0x99000001, 0xaa000000),
    (0x34ffffff, 0x56000001, 0x78123456, 0x9a654321),
    (0xdeadbeef, 0xcafebabe, 0x00abcdef, 0xff112233),
)


def mul24_values():
    return with_constants(MUL24_BODY)


def mul24_reference(words, *, mode="exact"):
    """Independent 48-bit Python integer reference for 24-bit integer multiplies."""
    mask32 = 0xffffffff

    def u24(x):
        return (x & mask32) if mode == "no_mask_24" else (x & 0xffffff)

    def s24(x):
        if mode == "no_mask_24":
            x &= mask32
            return x - 0x100000000 if (x & 0x80000000) else x
        v = x & 0xffffff
        return v - 0x1000000 if (v & 0x800000) else v

    def mul_i24(a, b):
        if mode == "i24_as_u24":
            return (u24(a) * u24(b)) & mask32
        return (s24(a) * s24(b)) & mask32

    def mul_u24(a, b):
        if mode == "u24_as_i24":
            return (s24(a) * s24(b)) & mask32
        return (u24(a) * u24(b)) & mask32

    def mul_hi_i24(a, b):
        if mode == "hi_i24_as_u24":
            return ((u24(a) * u24(b)) >> 32) & mask32
        prod = s24(a) * s24(b)
        if mode == "hi_i24_bad_lo0" and prod < 0 and (prod & mask32) == 0:
            return ((prod >> 32) - 1) & mask32
        return (prod >> 32) & mask32

    def mul_hi_u24(a, b):
        if mode == "hi_u24_as_i24":
            return ((s24(a) * s24(b)) >> 32) & mask32
        return ((u24(a) * u24(b)) >> 32) & mask32

    s44, s45, s46, s47 = words
    results = (
        mul_i24(s44, s45), mul_i24(s46, s47),
        mul_u24(s45, s46), mul_u24(s47, s44),
        mul_hi_i24(s44, s45), mul_hi_i24(s46, s47),
        mul_hi_u24(s45, s46), mul_hi_u24(s47, s44),
    )
    acc = 0x2468ace1
    for r in results:
        acc = (((acc << 5) | (acc >> 27)) & mask32) ^ (r & mask32)
        acc = (acc + 0x9e3779b9) & mask32
    return [float((acc >> (8 * k)) & 0xff) for k in range(4)]


def check_mul24_draws(directory, compiler, renderer, backend):
    source = directory / "mul24.hlsl"
    shader = directory / "mul24.cso"
    source.write_text(translate(mul24_values()), encoding="ascii")
    compile_shader(compiler, source, shader)

    cases = []

    def add_case(words, expected, status):
        path = directory / f"mul24-{len(cases)}.case"
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((path, status))

    for words in MUL24_SETS:
        add_case(words, mul24_reference(words), "pass")
    for mode in ("no_mask_24", "i24_as_u24", "u24_as_i24", "hi_i24_as_u24", "hi_i24_bad_lo0", "hi_u24_as_i24"):
        words = next(w for w in MUL24_SETS if mul24_reference(w, mode=mode) != mul24_reference(w))
        add_case(words, mul24_reference(words, mode=mode), "mismatch")
    # End batch with a passing reference after negative controls.
    add_case(MUL24_SETS[0], mul24_reference(MUL24_SETS[0]), "pass")

    run_pixel_batch(renderer, backend, shader, cases)

    # Mutated shader control: replacing v_mul_hi_i32_i24_e32 with v_mul_hi_u32_u24_e32 must mismatch.
    wrong_prog = [
        line.replace("v_mul_hi_i32_i24_e32 v24, s44, v28", "v_mul_hi_u32_u24_e32 v24, s44, v28")
        for line in mul24_values()
    ]
    source.write_text(translate(wrong_prog), encoding="ascii")
    compile_shader(compiler, source, shader)
    mut_case = directory / "mul24-mut.case"
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *MUL24_SETS[0])
    mut_case.write_bytes(constants + struct.pack("<4f", *mul24_reference(MUL24_SETS[0])) * 64)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, mut_case], capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_mul_hi_i32_i24_e32 with v_mul_hi_u32_u24_e32 escaped detection")


def check_three_way_draw(directory, compiler, renderer, backend):
    source = directory / "three_way.hlsl"
    shader = directory / "three_way.cso"
    source.write_text(translate(three_way()), encoding="ascii")
    compile_shader(compiler, source, shader)
    for constants in ((.25, .75, .375, .5), (.625, .125, -.3125, .5)):
        case = three_way_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        result = subprocess.run(draw)
        if result.returncode:
            return result.returncode
    without_abs = [line.replace("|s46|", "s46") for line in three_way()]
    source.write_text(translate(without_abs), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The renderer failed to detect the missing min3 absolute value")
    source.write_text(translate([line.replace("v_med3_f32 v12", "v_min3_f32 v12") for line in three_way()]),
                      encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A minimum replaced the median without a mismatch")
    return 0


def check_buffer_color_draw(directory, compiler, renderer, backend):
    source, shader, case = (directory / name for name in ("buffer.hlsl", "buffer.cso", "buffer.case"))
    for rg in (False, True):
        source.write_text(translate_buffer_color(buffer_color(rg)), encoding="ascii")
        compile_shader(compiler, source, shader)
        draw = [renderer, backend, "--check-pixel-case-rg" if rg else "--check-pixel-case", shader, case]
        for red, green in ((0.25, 0.75), (-0.5, 0.125)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 16, green, 2, 3, red)
            blue, alpha = (-1, -1) if rg else (green, red)
            expected = struct.pack("<4f", red, green, blue, alpha) * 64
            case.write_bytes(constants + expected)
            result = subprocess.run(draw)
            if result.returncode:
                return result.returncode
        struct.pack_into("<f", constants, 28, green)
        case.write_bytes(constants + expected)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError("Wrong buffer color escaped detection")
def check_unpack_byte_draws(directory, compiler, renderer, backend):
    instructions = unpack_bytes()
    source = directory / "ubyte.hlsl"
    shader = directory / "ubyte.cso"
    case = directory / "ubyte.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)
    # Every word is a normal binary32 value, so its bits survive the constant buffer, and
    # all four bytes of each word differ, so reading the wrong byte changes the result.
    for words in ((0x40ff8001, 0x3e7f02c3, 0x41a5005a, 0x00102030 | 0x3f000000),
                  (0x4280ff7f, 0x3f11fe22, 0x407f8013, 0x00000000 | 0x40c08070)):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        values = [float((word >> 8 * n) & 0xff) for n, word in enumerate(words)]
        values[3] = min(values[3], 1.0)
        case.write_bytes(constants + struct.pack("<4f", *values) * 64)
        subprocess.run(draw, check=True)

    wrong = [line.replace("ubyte1", "ubyte0") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    compile_shader(compiler, source, shader)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Reading the wrong byte escaped detection")
def check_reciprocal_iflag_draws(directory, compiler, renderer, backend):
    instructions = reciprocal_iflag()
    source = directory / "rcp_iflag.hlsl"
    shader = directory / "rcp_iflag.cso"
    case = directory / "rcp_iflag.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)
    inf = math.inf
    # Powers of two have exact reciprocals, so no rounding difference can hide a fault.
    # A zero operand gives an infinity of its sign, and clamp turns +inf into 1.
    for (a, b, c, d), expected in (((2.0, 4.0, -0.125, 0.5), (0.5, -0.25, 8.0, 1.0)),
                                   ((-0.25, 0.5, 16.0, 4.0), (-4.0, -2.0, 0.0625, 0.25)),
                                   ((0.0, 0.0, -0.0, 0.0), (inf, -inf, inf, 1.0))):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        subprocess.run(draw, check=True)

    wrong = [line.replace("v_rcp_iflag_f32", "v_sqrt_f32") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    compile_shader(compiler, source, shader)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Replacing the reciprocal with a square root escaped detection")
def check_offset_nibble_draws(directory, compiler, renderer, backend):
    instructions = offset_nibbles()
    source = directory / "cvt_off.hlsl"
    shader = directory / "cvt_off.cso"
    case = directory / "cvt_off.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)

    def offset(nibble):
        return (nibble - 16 if nibble & 8 else nibble) / 16

    # Each word is a normal binary32 value whose upper bits are set, so a result that used
    # more than bits 3:0 would change. All 16 nibbles appear across the four draws, and
    # the clamped lane sees both signs.
    high = 0x3f5a6b70
    nibbles = list(range(16))
    for start in range(0, 16, 4):
        group = nibbles[start:start + 4]
        words = [high | n for n in group]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        values = [offset(n) for n in group]
        values[3] = min(1.0, max(0.0, values[3]))
        # Python's integer results lose the sign of zero; nibble 0 gives +0.0 either way.
        case.write_bytes(constants + struct.pack("<4f", *values) * 64)
        subprocess.run(draw, check=True)

    # Reading the nibble as unsigned changes every result with bit 3 set.
    unsigned = translate(instructions).replace("asint(", "(")
    source.write_text(unsigned, encoding="ascii")
    compile_shader(compiler, source, shader)
    words = [high | 0x8, high | 0x9, high | 0xf, high | 0xc]
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *words)
    case.write_bytes(constants + struct.pack("<4f", -0.5, -0.4375, -0.0625, 0.0) * 64)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Reading the nibble without its sign escaped detection")
def check_trig_draws(directory, compiler, renderer, backend):
    source = directory / "trig.hlsl"
    shader = directory / "trig.cso"
    case = directory / "trig.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    instructions = with_constants(TRIG_BODY)
    hlsl = translate(instructions)

    def write_case(inputs):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *inputs)
        case.write_bytes(constants + struct.pack("<4f", *trig_reference(inputs)) * 64)

    source.write_text(hlsl, encoding="ascii")
    compile_shader(compiler, source, shader)
    for inputs in TRIG_CASES:
        write_case(inputs)
        subprocess.run(draw, check=True)

    def mismatches(text):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader)
        detected = []
        for inputs in TRIG_CASES:
            write_case(inputs)
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode not in (0, 1) or (wrong.returncode == 1) != ("pixel_mismatch" in wrong.stderr):
                raise AssertionError(f"Unexpected renderer result {wrong.returncode}: {wrong.stderr}")
            detected.append(wrong.returncode == 1)
        return detected

    # Both controls change at least one quantized channel of every case by many steps.
    swapped = [line.replace("v_sin_f32", "v_tmp").replace("v_cos_f32", "v_sin_f32").replace("v_tmp", "v_cos_f32")
               for line in instructions]
    assert all(trig_reference(inputs, swap=True) != trig_reference(inputs) for inputs in TRIG_CASES)
    reduced_argument = "(turns(value) * 6.28318548f)"
    assert hlsl.count(reduced_argument) == 2, hlsl
    for name, text in (("calls swapped between sine and cosine", translate(swapped)),
                       ("radians instead of turns", hlsl.replace(reduced_argument, "(value)"))):
        if not all(mismatches(text)):
            raise AssertionError(f"Sine and cosine with {name} escaped detection")
    # Without the reduction, a binary32 x * 2 pi of the large inputs moves each case by 20 or more steps.
    # Whether the adapter shows it depends on the precision of its own sin and cos argument handling, so
    # the count is reported rather than required; the reduced draws above must match on every adapter.
    unreduced = mismatches(hlsl.replace(reduced_argument, "(value * 6.28318548f)"))
    print(f"trig_unreduced_mismatches={sum(unreduced)}/{len(unreduced)}")


def check_legacy_multiply_add_draws(directory, compiler, renderer, backend):
    instructions = legacy_multiply_add()
    source = directory / "mad_legacy.hlsl"
    shader = directory / "mad_legacy.cso"
    case = directory / "mad_legacy.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]

    edge_program = uncompressed_export([
        "v_mov_b32_e32 v19, s47",
        "v_mad_legacy_f32 v12, s44, s45, 1.0",
        "v_mad_legacy_f32 v13, s46, 1.0, 1.0",
        "v_madak_f32 v14, s44, v19, 0x3f400000",
        "v_mad_legacy_f32 v15, 1.0, |v19|, 0.0",
    ])
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -0.0, math.inf, math.nan, -1.0)
    write_case(case, constants, (1.0, math.nan, 0.75, 1.0), nan_mask=0b0010)
    source.write_text(translate(edge_program), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    subprocess.run(draw, check=True)

    source.write_text(translate(instructions), encoding="ascii")
    subprocess.run(compile_shader, check=True)

    def legacy(x, y, z):
        product = 0.0 if x == 0 or y == 0 else x * y
        rounded_product, = struct.unpack("<f", struct.pack("<f", product))
        return rounded_product + z

    inf, nan = math.inf, math.nan
    # The last set detects unintended fusion: (1 + 2^-23) * (1 - 2^-23)
    # rounds to 1 as a float before adding -1, instead of returning -2^-46.
    delta = 2.0**-23
    sets = ((0.0, inf, 1.5, -2.0), (2.0, 3.0, 0.5, 0.25), (0.0, nan, -0.75, 4.0),
            (-1.5, 0.0, 2.0, 0.5), (1 + delta, 1 - delta, -1.0, 0.0))

    def write_legacy_case(values, *, fused_first_channel=False):
        a, b, c, d = values
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        expected = (legacy(a, b, c), legacy(b, a, -d), legacy(abs(c), d, a),
                    min(1.0, max(0.0, legacy(a, d, c))))
        if fused_first_channel:
            expected = (a * b + c, *expected[1:])
        assert not any(math.isnan(v) for v in expected), expected
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
    for values in sets:
        write_legacy_case(values)
        subprocess.run(draw, check=True)

    write_legacy_case(sets[-1], fused_first_channel=True)
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("The renderer accepted the fused reference despite intermediate rounding")

    # An ordinary multiply-add gives NaN for 0 * inf on the first set.
    wrong = [line.replace("v_mad_legacy_f32", "v_fma_f32") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    write_legacy_case(sets[0])
    result = subprocess.run(draw, capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Replacing the legacy multiply-add lost the zero rule without detection")


def check_rgb_sample_draw(directory: Path, compiler: str, renderer: str, backend: str) -> None:
    hlsl = directory / "rgb-sample.hlsl"
    shader = directory / "rgb-sample.cso"
    case = directory / "rgb-sample.case"
    hlsl.write_text(translate(rgb_sample()), encoding="ascii")
    compile_shader(compiler, hlsl, shader)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for alpha in (0.5, 1.0):
        colors = (struct.pack("<4f", half_rtz(red / 255), half_rtz(green / 255), half_rtz(128 / 255), alpha)
                  for green in RAMP for red in RAMP)
        case.write_bytes(bytes(256) + b"".join(colors))
        if alpha == 0.5:
            subprocess.run(draw, check=True)
        else:
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch x=0 y=0" not in wrong.stderr:
                raise AssertionError("RGB sample with an overwritten alpha register escaped detection")


def check_render(directory, compiler, renderer, backend, *, core_only=False):
    for swap in (False, True):
        source = directory / "fixture.hlsl"
        shader = directory / "fixture.cso"
        source.write_text(translate(assembly(swap)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        draw = [renderer, backend, "--check-pixel", shader]
        result = subprocess.run([*draw, "bgra" if swap else "rgba"])
        if result.returncode:
            return result.returncode
        if swap:
            wrong = subprocess.run([*draw, "rgba"], capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("The renderer failed to detect the changed channel order")
    check_dpp_int32_draw(directory, compiler, renderer, backend)
    check_dpp_draw(directory, compiler, renderer, backend)
    check_vop3p_draw(directory, compiler, renderer, backend)
    check_bitwise_draw(directory, compiler, renderer, backend)
    check_fma_mix_draw(directory, compiler, renderer, backend)
    check_dot_draw(directory, compiler, renderer, backend)
    check_pk_int16_draw(directory, compiler, renderer, backend)
    check_narrow_draw(directory, compiler, renderer, backend)
    check_mode_register_draw(directory, compiler, renderer, backend)
    check_movk_draw(directory, compiler, renderer, backend)
    check_half_draw(directory, compiler, renderer, backend)
    for exp_shader_inst in ("s_waitcnt_expcnt null, 0x3", "s_waitcnt expcnt(0)"):
        exp_program = [exp_shader_inst] + assembly(False)[:-1] + [exp_shader_inst, "s_endpgm"]
        source.write_text(translate(exp_program), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        subprocess.run([renderer, backend, "--check-pixel", shader, "rgba"], check=True)
    check_rgb_sample_draw(directory, compiler, renderer, backend)
    check_trig_draws(directory, compiler, renderer, backend)
    program = arithmetic()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for constants in ((4.0, -.0625, .125, -.015625), (2.0, -.03125, .0625, -.0078125)):
        draw = [renderer, backend, "--check-pixel-case", shader, arithmetic_case(directory, *constants)]
        subprocess.run(draw, check=True)
    swapped = [line.replace("v_subrev_f32", "v_sub_f32") for line in program]
    source.write_text(translate(swapped), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped subtraction operands escaped detection")

    program = with_scalar_constant_reads()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    case = directory / "scalar-constants.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for red, green, blue in ((.25, -.5, .75), (-.125, .625, .875)):
        constants = bytearray(256)
        for offset, value in ((0, red), (8, green), (252, blue)):
            struct.pack_into("<f", constants, offset, value)
        write_case(case, constants, (red, green, blue, 1))
        subprocess.run(draw, check=True)
    wrong_word = [line.replace("s62, s[44:47], 0x8", "s62, s[44:47], 0x4") for line in program]
    source.write_text(translate(wrong_word), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading the neighboring constant word escaped detection")

    check_affine_lod_draw(directory, compiler, renderer, backend)
    program = with_scalar_pair_reads()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for constants in ((2.0, -.0625, 1.0, .5, .25, .5), (.5, -.03125, .75, 2.0, .125, .25)):
        case = scalar_pair_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    # Words 4, 6 and 63 are the second word of each pair; each must reach its own channel.
    reference = case.read_bytes()
    for offset in (16, 24, 252):
        corrupted = bytearray(reference)
        struct.pack_into("<f", corrupted, offset, 1.0)
        case.write_bytes(corrupted)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Changing the constant at byte {offset} escaped detection")
    case.write_bytes(reference)
    shifted = [line.replace("s[62:63], s[44:47], 0x14", "s[62:63], s[44:47], 0x10") for line in program]
    source.write_text(translate(shifted), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading the pair one word early escaped detection")

    # The NVIDIA driver (32.0.16.2124, RTX 5060 Ti) refuses PSO creation for the curve program
    # under -denorm preserve (E_OUTOFMEMORY). Its rows contain no binary32 subnormals, so the
    # mode cannot change the exported bits; record this fixture's mode as any, not the assumption,
    # and assert that mode independence on every expected word instead of arguing it.
    for program_fn, case_fn, mode in ((curve, curve_case, "any"),
                                      (shaping, shaping_case, F32_DENORM_MODE)):
        source.write_text(translate(program_fn()), encoding="ascii")
        compile_shader(compiler, source, shader, mode=mode)
        for gain in (.75, 1.125):
            case = case_fn(directory, gain)
            if mode != F32_DENORM_MODE:
                words = struct.unpack("<256f", case.read_bytes()[256:1280])
                assert all(flush_ftz(word) == word for word in words), program_fn.__name__
            draw = [renderer, backend, "--check-pixel-case", shader, case]
            subprocess.run(draw, check=True)
        corrupted = bytearray(case.read_bytes())
        struct.pack_into("<f", corrupted, 16, 0.0)
        case.write_bytes(corrupted)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError("Stale reference accepted after changing the bound constants")

    if core_only:
        return 0
    check_predicate_and_rounding_draws(directory, compiler, renderer, backend)
    return check_blend_and_scalar_draws(directory, compiler, renderer, backend)


def check_sopk_cmpk_cmovk_draw(directory, compiler, renderer, backend):
    source = directory / "sopk_cmpk_cmovk.hlsl"
    shader = directory / "sopk_cmpk_cmovk.cso"
    case = directory / "sopk_cmpk_cmovk.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    program = sopk_cmpk_cmovk()

    def write_words(words, expected):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        for word in expected:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(case, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)

    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for words in SOPK_CMPK_WORDS:
        write_words(words, sopk_cmpk_cmovk_reference(words))
        subprocess.run(draw, check=True)

    words = SOPK_CMPK_WORDS[0]
    write_words(words, sopk_cmpk_cmovk_reference(words))
    for old, new, label in (
        ("s_cmpk_lt_i32 s44, 0x8000", "s_cmpk_lt_u32 s44, 0x8000", "signed sopk comparison"),
        ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s50, 0x7fff", "sopk conditional move"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutating {label} escaped detection")


def check_predicate_and_rounding_draws(directory, compiler, renderer, backend):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    program = truncate_values()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for constants in ((1.75, -1.75, -2.75, 2.75), (-1.75, 2.75, 3.75, -1.75)):
        case = truncate_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    rounded_down = [line.replace("v_trunc_f32", "v_floor_f32") for line in program]
    source.write_text(translate(rounded_down), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding toward negative infinity escaped detection")

    program = sopk_scalar()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for red, green in ((0.5, 0.25), (0.125, 1.0)):
        case = sopk_case(directory, red, green)
        subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)

    case = sopk_case(directory, 0.5, 0.25)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    changed = [line.replace("s_mulk_i32 s50, 0x3", "s_mulk_i32 s50, 0x5") for line in program]
    source.write_text(translate(changed), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)

    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Changing the sopk multiply escaped detection")

    check_sopk_cmpk_cmovk_draw(directory, compiler, renderer, backend)

    source.write_text(translate(rounding()), encoding="ascii")
    compile_shader(compiler, source, shader)
    # Ties go to even (2.5 to 2, -3.5 to -4), unlike rounding half away from zero or half up.
    for constants in ((4.0, 2.5, 1.25), (8.0, -3.5, -0.75)):
        draw = [renderer, backend, "--check-pixel-case", shader, rounding_case(directory, *constants)]
        subprocess.run(draw, check=True)
    for old, new, failure in (("v_rndne_f32_e32", "v_floor_f32_e32", "Flooring instead of rounding"),
                              ("v_ceil_f32_e32", "v_floor_f32_e32", "Flooring instead of ceiling")):
        source.write_text(translate([line.replace(old, new) for line in rounding()]), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"{failure} escaped detection")
    # Rounding half up gives -3 for the -3.5 tie, so the same draw must catch it.
    source.write_text(translate(rounding()).replace("round(", "floor(0.5f + "), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding ties up instead of to even escaped detection")

    program = absolute_value()
    case = absolute_case(directory)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    subprocess.run(draw, check=True)

    without_abs = [line.replace("|v12|", "v12") for line in program]
    source.write_text(translate(without_abs), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The renderer failed to detect the missing absolute value")
    check_ceiling_draws(directory, compiler, renderer, backend)
    check_unpack_byte_draws(directory, compiler, renderer, backend)
    check_reciprocal_iflag_draws(directory, compiler, renderer, backend)
    check_offset_nibble_draws(directory, compiler, renderer, backend)
    check_legacy_multiply_add_draws(directory, compiler, renderer, backend)
    source.write_text(translate(select()), encoding="ascii")
    compile_shader(compiler, source, shader)
    for constants in ((.5, .75, .375, .25), (.125, .5, 1.0, 0.0)):
        case = select_case(directory, *constants)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    corrupted = bytearray(case.read_bytes())
    struct.pack_into("<f", corrupted, 12, .5)
    case.write_bytes(corrupted)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Stale reference accepted after moving the comparison threshold")
    # Binary32 values on either side of a half-ULP midpoint, in both signs.
    values = (1 + 3 / 4096, 1 + 1 / 4096, -1 - 3 / 4096, -1 - 1 / 4096)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *values)
    program = with_constants([f"v_sub_f32_e64 v{12 + i}, s{44 + i}, 0" for i in range(4)])
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    case = directory / "signed-export.case"
    write_case(case, constants, tuple(map(half_rtz, values)))
    subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    check_class_draw(directory, compiler, renderer, backend)
    check_predicate_draws(directory, compiler, renderer, backend)
    check_int_predicate_draws(directory, compiler, renderer, backend)
    check_unsigned_less_than_draws(directory, compiler, renderer, backend)
    check_sext_draw(directory, compiler, renderer, backend)
    check_scalar_pack_draw(directory, compiler, renderer, backend)
    check_scalar_bit_count_draw(directory, compiler, renderer, backend)
    check_scalar_carry_chain_draw(directory, compiler, renderer, backend)
    check_scalar_leading_bit_draw(directory, compiler, renderer, backend)
    check_scalar_mul_hi_draw(directory, compiler, renderer, backend)
    check_int16_predicate_draws(directory, compiler, renderer, backend)
    check_sdwa_predicate_draws(directory, compiler, renderer, backend)
    check_literal_fma_draw(directory, compiler, renderer, backend)
    return 0


def check_blend_and_scalar_draws(directory, compiler, renderer, backend):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    capture = directory / "blend.rgba"
    draw = [renderer, backend, "--dual-pixel", shader, capture]
    for case in DUAL_CASES:
        source.write_text(translate_dual(dual_program(case)), encoding="ascii")
        compile_shader(compiler, source, shader)
        subprocess.run(draw, check=True)
    wrong = subprocess.run([renderer, backend, "--dual-pixel-swapped", shader, capture], capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped texture bindings escaped detection")
    source.write_text(translate_dual(dual_program(swap_weights=True)), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped blend weights escaped detection")
    check_cube_draw(directory, compiler, renderer, backend)
    check_sdwa_draw(directory, compiler, renderer, backend)
    check_sdwa_arith_draws(directory, compiler, renderer, backend)
    check_fract_draw(directory, compiler, renderer, backend)
    check_reciprocal_draw(directory, compiler, renderer, backend)
    check_int_minmax_draws(directory, compiler, renderer, backend)
    check_cvt_int_draws(directory, compiler, renderer, backend)

    check_frexp_ldexp_draws(directory, compiler, renderer, backend)
    check_mul24_draws(directory, compiler, renderer, backend)
    check_scalar_select64_draw(directory, compiler, renderer, backend)
    check_scalar_bfe64_draw(directory, compiler, renderer, backend)

    program = scalar64()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for values in SCALAR64_SETS:
        case = scalar64_case(directory, values)
        draw = [renderer, backend, "--check-pixel-case", shader, case]
        subprocess.run(draw, check=True)
    # Keep the original two sets for the SCC toggle checks and add a draw-only case.
    _, _, *middle, _, alt = SCALAR64_SETS[0]
    a = 0x11234567 << 32 | 0xffffffff
    b = middle[1] << 32 | middle[0]
    # s_not_b64 of a | b sets only bit 63, so its SCC depends on the high word alone.
    assert ~(a | b) & (2**64 - 1) == 1 << 63
    case = scalar64_case(directory, (0xffffffff, 0x11234567, *middle, 0x00400018, alt))
    subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    # Mutations verified against Set 1 (where negative signed values and SCC=1 paths are active).
    set1_case = scalar64_case(directory, SCALAR64_SETS[0])
    draw = [renderer, backend, "--check-pixel-case", shader, set1_case]
    for old, new, label in (
        ("s_ashr_i64 s[72:73], s[52:53], s56", "s_lshr_b64 s[72:73], s[52:53], s56", "signed right shift"),
        ("s_bfe_i32 s72, s60, s61", "s_bfe_u32 s72, s60, s61", "signed bitfield extract"),
        ("s_andn2_b64 s[72:73], s[52:53], s[54:55]", "s_and_b64 s[72:73], s[52:53], s[54:55]", "andn2 inversion"),
        ("s_cmov_b64 s[78:79], -1", "s_cmov_b64 s[78:79], 0", "conditional move"),
        ("s_nand_b64 s[72:73], s[52:53], s[54:55]", "s_and_b64 s[72:73], s[52:53], s[54:55]", "nand inversion"),
        ("s_nor_b64 s[72:73], s[52:53], s[54:55]", "s_or_b64 s[72:73], s[52:53], s[54:55]", "nor inversion"),
        ("s_xnor_b64 s[72:73], s[52:53], s[54:55]", "s_xor_b64 s[72:73], s[52:53], s[54:55]", "xnor inversion"),
        ("s_nor_b64 s[72:73], s[82:83], -1", "s_nor_b64 s[72:73], s[82:83], 0", "nor SCC probe"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")
    check_mad_draw(directory, compiler, renderer, backend)
    check_mad_literal_draw(directory, compiler, renderer, backend)
    source.write_text(translate(scalar()), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    for words in SCALAR_WORDS:
        case = scalar_case(directory, words)
        subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    for old, new in SCALAR_MUTATIONS:
        case = scalar_case(directory, scalar_detecting_words(old, new))
        source.write_text(translate([line.replace(old, new) for line in scalar()]), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, case], capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Replacing {old} with {new} escaped detection")
    check_sop1_scalar_ops_draw(directory, compiler, renderer, backend)
    return 0


def check_sop1_scalar_ops_draw(directory, compiler, renderer, backend):
    source = directory / "sop1_scalar_ops.hlsl"
    shader = directory / "sop1_scalar_ops.cso"
    case = directory / "sop1_scalar_ops.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    program = sop1_scalar_ops()

    def write_words(words, expected):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        for word in expected:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(case, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)

    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    for words in SOP1_WORDS:
        write_words(words, sop1_scalar_ops_reference(words))
        subprocess.run(draw, check=True)

    words = SOP1_WORDS[1]
    write_words(words, sop1_scalar_ops_reference(words))
    for old, new, label in (
        ("s_abs_i32 s50, s44", "s_not_b32 s50, s44", "signed scalar abs"),
        ("s_bitset0_b32 s52, s47", "s_bitset1_b32 s52, s47", "scalar bitset clear"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutating {label} escaped detection")

    # Both high bits start set; s45 equals s46, so the taken s_cmov_b32 leaves the value unchanged.
    words = (2, 0xc1400009, 0xc1400009, 31)
    set_bit = 4
    probe = [line.replace("s_bitset1_b32 s52, 31", f"s_bitset1_b32 s52, {set_bit}") for line in program]
    assert probe != program
    source.write_text(translate_int(probe), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_words(words, sop1_scalar_ops_reference(words, set_bit))
    subprocess.run(draw, check=True)


def check_bfm_draw(directory, compiler, renderer, backend):
    """s_bfm_b32 and s_bfm_b64 against the Python reference, then controls that must mismatch."""
    source = directory / "bfm.hlsl"
    shader = directory / "bfm.cso"
    case = directory / "bfm.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    program = bfm_scalar_ops()
    lowered = translate_int(program)

    def write_words(words):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        expected = bfm_scalar_ops_reference(words)
        write_case(case, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)

    source.write_text(lowered, encoding="ascii")
    compile_shader(compiler, source, shader)
    for words in BFM_WORDS:
        write_words(words)
        subprocess.run(draw, check=True)
    # Controls on the generated HLSL: a narrower count mask and swapped count/offset. Words 1 and 4 distinguish both.
    for old, new, label in (("(1u << (asuint(params[0].w) & 31u))", "(1u << (asuint(params[0].w) & 15u))", "b32 count mask"),
                            ("((1u << (asuint(params[0].w) & 31u)) - 1u) << (asuint(params[1].x) & 31u)",
                             "((1u << (asuint(params[1].x) & 31u)) - 1u) << (asuint(params[0].w) & 31u)", "b32 operand order")):
        mutated = lowered.replace(old, new)
        assert mutated != lowered, label
        source.write_text(mutated, encoding="ascii")
        compile_shader(compiler, source, shader)
        mismatches = 0
        for words in BFM_WORDS:
            write_words(words)
            wrong = subprocess.run(draw, capture_output=True, text=True)
            mismatches += wrong.returncode == 1 and "pixel_mismatch" in wrong.stderr
        if not mismatches:
            raise AssertionError(f"Mutating {label} escaped detection")


def check_e64_render(directory, compiler, renderer, backend):
    """The e64 draw cases on their own, so they run as a separate CTest entry per backend. A renderer that reports an
    unavailable adapter (exit 77) makes this a skip, as for the other pixel draws."""
    try:
        check_e64_draws(directory, compiler, renderer, backend)
    except subprocess.CalledProcessError as error:
        if error.returncode == 77:
            return 77
        raise
    check_denorm_mode_draw(directory, compiler, renderer, backend)
    check_f16_denorm_draw(directory, compiler, renderer, backend)
    check_nan_class_draw(directory, compiler, renderer, backend)
    check_integer_readback_draw(directory, compiler, renderer, backend)
    check_subrev_u32_draw(directory, compiler, renderer, backend)
    check_lshl_add_u32_draw(directory, compiler, renderer, backend)
    check_cvt_int_f32_draw(directory, compiler, renderer, backend)
    check_cvt_int_f32_e64_draw(directory, compiler, renderer, backend)
    check_cvt_float_int_draw(directory, compiler, renderer, backend)
    return 0


def main():
    if len(sys.argv) == 1:
        with tempfile.TemporaryDirectory() as tmp:
            check_controls(Path(tmp))
        return 0
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        if mode in ("draw-blend-scalar", "draw-predicate-rounding", "draw-integer-nc", "draw-bfm"):
            try:
                draw = {"draw-blend-scalar": check_blend_and_scalar_draws,
                        "draw-predicate-rounding": check_predicate_and_rounding_draws,
                        "draw-integer-nc": check_integer_nc_draws,
                        "draw-bfm": check_bfm_draw}[mode]
                return draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-e64":
            # The controls already run in the binary and draw modes; this mode only draws.
            return check_e64_render(directory, *args)
        check_controls(directory)
        if mode == "binary":
            check_binary(directory, *args)
        elif mode in ("draw", "draw-core"):
            return check_render(directory, *args, core_only=mode == "draw-core")
        elif mode == "three-way":
            return check_three_way_draw(directory, *args)
        elif mode == "buffer-color":
            return check_buffer_color_draw(directory, *args)
        else:
            raise ValueError("Expected binary, draw, draw-e64, three-way or buffer-color mode")


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:  # also on skip or failure; tools/form_coverage.py reads the test's status from JUnit
        write_record(os.environ.get("FORM_COVERAGE_RECORD"))
