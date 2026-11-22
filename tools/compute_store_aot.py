"""Translate bounded raw-UAV gfx1013 compute programs under explicit bindings."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from pixel_aot import container_code, disassemble
from resource_inventory import parts


CONSTANT_WORDS = 16
AXES = "xyz"


def checked_group(group):
    """Return (x, y, z) within the D3D12 cs_6_0 numthreads limits."""
    x, y, z = (group, 1, 1) if isinstance(group, int) else tuple(group)
    if not (all(isinstance(n, int) for n in (x, y, z)) and 1 <= x <= 1024 and 1 <= y <= 1024
            and 1 <= z <= 64 and x * y * z <= 1024):
        raise ValueError("Unsupported thread group")
    return x, y, z


def group_size(psv):
    size = struct.unpack_from("<I", psv)[0] if len(psv) >= 4 else 0
    if not 48 <= size <= len(psv) - 4 or psv[28] != 5:
        raise ValueError("Expected compute PSV runtime info with thread dimensions")
    return checked_group(struct.unpack_from("<3I", psv, 40))


def is_word(value):
    return isinstance(value, int) or (isinstance(value, tuple) and value[0] in ("constant_word", "thread"))


def offset_terms(tag):
    """Return the ((axis, shift), ...) terms of a dispatch byte offset, or None."""
    return tag[1] if isinstance(tag, tuple) and tag[0] == "offset" else None


def lower(instructions, group, *, table_reg, descriptor_offset, group_reg=None, stride=None,
          constant_descriptor=None, group_reg_y=None, group_reg_z=None):
    """v0, v1 and v2 are local X, Y and Z; group_reg, group_reg_y and group_reg_z are the group IDs.

    gfx10 compute waves receive each local ID in its own VGPR. Y and Z are bound only when that
    group dimension is larger than one. Descriptors name u0 and optionally b0.
    """
    dims = checked_group(group)
    if not 0 <= table_reg <= 105 or not 0 <= descriptor_offset < 1 << 20 or descriptor_offset % 4:
        raise ValueError("Invalid table or thread group binding")
    descriptor_bindings = {descriptor_offset: "uav"}
    if constant_descriptor is not None:
        if (not 0 <= constant_descriptor < 1 << 20 or constant_descriptor % 4
                or abs(constant_descriptor - descriptor_offset) < 16):
            raise ValueError("Invalid or overlapping constant descriptor binding")
        descriptor_bindings[constant_descriptor] = "cbv"
    group_regs = {axis: reg for axis, reg in enumerate((group_reg, group_reg_y, group_reg_z)) if reg is not None}
    if (any(not 0 <= reg <= 105 or reg == table_reg for reg in group_regs.values())
            or len(set(group_regs.values())) != len(group_regs)):
        raise ValueError("Invalid group register")
    if stride is not None and (not group_regs or not 4 <= stride <= 0x3fff or stride % 4):
        raise ValueError("Indexed stores require a group register and aligned stride")
    scalar = {table_reg: "table"}
    for axis, reg in group_regs.items():
        scalar[reg] = ("group", axis)
    vector = {axis: ("local", axis) for axis in range(3) if axis == 0 or dims[axis] > 1}
    pending = set()
    stores = []
    written_spans = []
    layout = None
    if not instructions or instructions[-1] != "s_endpgm":
        raise ValueError("Missing final program exit")

    for line in instructions[:-1]:
        if re.fullmatch(r"s_inst_prefetch (?:0x[0-9a-f]+|\d+)|s_nop \d+", line):
            continue
        if m := re.fullmatch(r"s_clause (0x[0-9a-f]+|\d+)", line):
            if int(m[1], 0) > 0x3f:
                raise ValueError("Invalid s_clause operand")
            continue
        if line in ("s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(0) vmcnt(0)", "s_waitcnt vmcnt(0) lgkmcnt(0)"):
            pending.clear()
        elif line == "s_waitcnt vmcnt(0)":
            continue
        elif m := re.fullmatch(r"s_mov_b32 s(\d+), (s\d+|-?\d+|0x[0-9a-f]+)", line):
            dst, src = int(m[1]), m[2]
            if dst in pending or dst > 105:
                raise ValueError("Invalid scalar destination or pending load")
            if src.startswith("s"):
                n = int(src[1:])
                if n > 105 or n in pending or n not in scalar:
                    raise ValueError("Unknown or pending scalar source")
                scalar[dst] = scalar[n]
            else:
                value = int(src, 0)
                if not -(1 << 31) <= value <= 0xffffffff:
                    raise ValueError("Invalid scalar immediate")
                scalar[dst] = value & 0xffffffff
        elif m := re.fullmatch(r"s_movk_i32 s(\d+), (-?\d+|0x[0-9a-f]+)", line):
            dst = int(m[1])
            value = int(m[2], 0)
            if dst in pending or dst > 105 or not -0x8000 <= value <= 0xffff:
                raise ValueError("Invalid scalar destination or movk immediate")
            signed = value - 0x10000 if value >= 0x8000 else value
            scalar[dst] = signed & 0xffffffff
        elif m := re.fullmatch(r"s_lshl_b32 s(\d+), s(\d+), (\d+)", line):
            dst, group, shift = map(int, m.groups())
            tag = scalar.get(group)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group" else None
            # Only the first dispatch thread of a group: group * size, for a power-of-two size.
            if (dst in pending or dst > 105 or group in pending or axis is None
                    or dims[axis] & (dims[axis] - 1) or shift != dims[axis].bit_length() - 1):
                raise ValueError("Unverified group base")
            scalar[dst] = ("group_base", axis)
        elif m := re.fullmatch(r"s_load_dwordx4 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line):
            dest, last, base, high = map(int, m.groups()[:4])
            offset = 0 if m[5] == "null" else int(m[5], 16)
            regs = range(dest, last + 1)
            if (dest % 4 or last != dest + 3 or last > 105 or base % 2 or high != base + 1
                    or high > 105 or pending.intersection((*regs, base, high))
                    or (scalar.get(base), scalar.get(high)) != ("table", 0) or offset not in descriptor_bindings):
                raise ValueError("Unresolved compute descriptor load")
            for component, reg in enumerate(regs):
                scalar[reg] = (descriptor_bindings[offset], component)
            pending.update(regs)
        elif m := re.fullmatch(r"(?:s_buffer_load_dwordx([24]) s\[(\d+):(\d+)\]|s_buffer_load_dword s(\d+)), "
                               r"s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line):
            count = int(m[1] or 1)
            dest, last = (int(m[4]), int(m[4])) if m[4] else (int(m[2]), int(m[3]))
            base, high = int(m[5]), int(m[6])
            offset = 0 if m[7] == "null" else int(m[7], 16)
            regs = range(dest, last + 1)
            if (dest % count or last != dest + count - 1 or last > 105
                    or base % 4 or high != base + 3 or high > 105
                    or offset % 4 or offset // 4 + count > CONSTANT_WORDS
                    or pending.intersection((*regs, *range(base, high + 1)))
                    or any(scalar.get(base + i) != ("cbv", i) for i in range(4))):
                raise ValueError("Unresolved or out-of-range compute constant read")
            for word, reg in enumerate(regs, offset // 4):
                scalar[reg] = ("constant_word", word)
            pending.update(regs)
        elif m := re.fullmatch(r"v_mov_b32(?:_e32|_e64)? v(\d+), (s\d+|v\d+|-?\d+|0x[0-9a-f]+)", line):
            dst, src = int(m[1]), m[2]
            if dst > 255:
                raise ValueError("Invalid vector destination")
            if src.startswith("s"):
                n = int(src[1:])
                val = scalar.get(n)
                if n > 105 or n in pending or not is_word(val):
                    raise ValueError("Unresolved scalar move source")
                vector[dst] = val
            elif src.startswith("v"):
                n = int(src[1:])
                val = vector.get(n)
                if n > 255 or not is_word(val):
                    raise ValueError("Unresolved vector move source")
                vector[dst] = val
            else:
                value = int(src, 0)
                if not -(1 << 31) <= value <= 0xffffffff:
                    raise ValueError("Invalid literal move")
                vector[dst] = value & 0xffffffff
        elif m := re.fullmatch(r"v_lshl_add_u32 v(\d+), s(\d+), (\d+), v(\d+)", line):
            dst, group, shift, local = map(int, m.groups())
            tag = scalar.get(group)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group" else None
            # The dispatch-thread ID is group * size + local only for a power-of-two size.
            if (dst > 255 or group in pending or axis is None or dims[axis] & (dims[axis] - 1)
                    or shift != dims[axis].bit_length() - 1 or vector.get(local) != ("local", axis)):
                raise ValueError("Unverified dispatch index")
            vector[dst] = ("thread", axis)
        elif m := re.fullmatch(r"v_add_lshl_u32 v(\d+), s(\d+), v(\d+), (\d+)", line):
            dst, base, local, shift = map(int, m.groups())
            tag = scalar.get(base)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group_base" else None
            # (group * size + local) << shift: a dispatch byte offset on one axis.
            if (dst > 255 or base in pending or axis is None or shift > 31
                    or vector.get(local) != ("local", axis)):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("offset", ((axis, shift),))
        elif m := re.fullmatch(r"v_lshl_add_u32 v(\d+), v(\d+), (\d+), v(\d+)", line):
            dst, src, shift, addend = map(int, m.groups())
            tag, terms = vector.get(src), offset_terms(vector.get(addend))
            # (thread << shift) + offset: one more axis in the byte offset. The store checks
            # that the axes nest without overlap.
            if (dst > 255 or shift > 31 or terms is None or not (isinstance(tag, tuple) and tag[0] == "thread")
                    or any(tag[1] == axis for axis, _ in terms)):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("offset", (*terms, (tag[1], shift)))
        elif m := re.fullmatch(r"v_lshlrev_b32(?:_e32|_e64)? v(\d+), (\d+), v(\d+)", line):
            dst, shift, src = map(int, m.groups())
            tag = vector.get(src)
            # The element and row checks at the store bound the shift.
            if dst > 255 or shift > 31 or not (isinstance(tag, tuple) and tag[0] == "thread"):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("offset", ((tag[1], shift),))
        elif m := re.fullmatch(r"buffer_store_dword(?:x([234]))? (v\d+|v\[\d+:\d+\]), (off|v\d+|v\[\d+:\d+\]), "
                               r"s\[(\d+):(\d+)\], 0( idxen)?( offen)?(?: offset:(\d+))?", line):
            count = int(m[1] or 1)
            first, *last = map(int, re.findall(r"\d+", m[2]))
            base, high = int(m[4]), int(m[5])
            offset = int(m[8] or 0)
            if (last != ([] if count == 1 else [first + count - 1])
                    or high != base + 3 or base % 4 or offset % 4 or offset > 4095
                    or pending.intersection(range(base, high + 1))
                    or any(scalar.get(base + i) != ("uav", i) for i in range(4))):
                raise ValueError("Unverified raw UAV store")
            address = f"{offset}u"
            addr_regs = list(map(int, re.findall(r"\d+", m[3])))
            tags = [vector.get(reg) for reg in addr_regs]
            if (m[3] == "off" and not m[6] and not m[7] and dims == (1, 1, 1) and stride is None):
                store_layout = ("fixed",)
            elif (stride is not None and m[6] and not m[7] and len(addr_regs) == 1
                    and isinstance(tags[0], tuple) and tags[0][0] == "thread" and offset + 4 * count <= stride):
                # One index axis: every other group dimension must be one, so no two
                # invocations of a group share an address.
                index = tags[0][1]
                store_layout = ("index", index)
                address = f"tid.{AXES[index]} * {stride}u + {offset}u"
            elif (stride is not None and m[6] and m[7] and len(addr_regs) == 2 and addr_regs[1] == addr_regs[0] + 1
                    and isinstance(tags[0], tuple) and tags[0][0] == "thread"
                    and offset_terms(tags[1]) is not None
                    and tags[0][1] not in [axis for axis, _ in offset_terms(tags[1])]):
                # idxen offen: address = index * stride + byte offset + instruction offset. The
                # byte offset holds one or two axes; ordered by shift, each axis of one group must
                # fit below the next shift, and the last one in the stride.
                index, terms = tags[0][1], sorted(offset_terms(tags[1]), key=lambda term: term[1])
                limits = [1 << shift for _, shift in terms[1:]] + [stride]
                if (offset + 4 * count > 1 << terms[0][1]
                        or any(dims[axis] << shift > limit for (axis, shift), limit in zip(terms, limits))):
                    raise ValueError("Store leaves its element or row")
                store_layout = ("index_offset", index, *terms)
                columns = " + ".join(f"(tid.{AXES[axis]} << {shift}u)" for axis, shift in reversed(terms))
                address = f"tid.{AXES[index]} * {stride}u + {columns} + {offset}u"
            else:
                raise ValueError("Unsupported store address")
            covered = ({store_layout[1], *(axis for axis, _ in store_layout[2:])}
                       if store_layout[0] != "fixed" else set())
            if any(dims[axis] > 1 for axis in range(3) if axis not in covered):
                raise ValueError("Store address does not separate every invocation of the group")
            if layout not in (None, store_layout):
                raise ValueError("Stores use different address layouts")
            layout = store_layout
            span_end = offset + 4 * count
            if any(lo < span_end and offset < hi for lo, hi in written_spans):
                raise ValueError("Overlapping raw UAV stores")
            written_spans.append((offset, span_end))
            values = [vector.get(reg) for reg in range(first, first + count)]
            if not all(is_word(value) for value in values):
                raise ValueError("Store data is not a known word")
            words = ", ".join(
                f"0x{value:08x}u" if isinstance(value, int)
                else f"tid.{AXES[value[1]]}" if value[0] == "thread"
                else f"params[{value[1] // 4}].{'xyzw'[value[1] % 4]}"
                for value in values
            )
            arg = words if count == 1 else f"uint{count}({words})"
            stores.append(f"    outputData.Store{count if count > 1 else ''}({address}, {arg});")
        else:
            raise ValueError(f"Unsupported compute instruction: {line}")
    if not stores:
        raise ValueError("Expected at least one observable store")
    body = "\n".join(stores)
    cbuffer = (f"cbuffer Constants : register(b0) {{ uint4 params[{CONSTANT_WORDS // 4}]; }};\n"
               if constant_descriptor is not None else "")
    return (f"{cbuffer}RWByteAddressBuffer outputData : register(u0);\n[numthreads({dims[0]},{dims[1]},{dims[2]})]\n"
            f"void main(uint3 tid : SV_DispatchThreadID) {{\n{body}\n}}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path, help="Local extracted DXBC container")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm", required=True)
    parser.add_argument("--table-reg", type=int, required=True)
    parser.add_argument("--descriptor-offset", type=lambda n: int(n, 0), required=True)
    parser.add_argument("--constant-descriptor", type=lambda n: int(n, 0), help="Table byte offset bound to b0")
    parser.add_argument("--group-reg", type=int, help="SGPR holding the group X ID")
    parser.add_argument("--group-reg-y", type=int, help="SGPR holding the group Y ID")
    parser.add_argument("--group-reg-z", type=int, help="SGPR holding the group Z ID")
    parser.add_argument("--stride", type=int, help="Indexed raw UAV byte stride")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".hlsl" or args.output.resolve() == args.container.resolve():
        parser.error("Choose an HLSL destination distinct from the input")
    blob = args.container.read_bytes()
    chunks = parts(blob)
    group = group_size(chunks.get(b"PSV0", b""))
    code = container_code(chunks.get(b"DXIL", b""), stage=5)
    source = lower(disassemble(code, args.llvm), group, table_reg=args.table_reg,
                   descriptor_offset=args.descriptor_offset, group_reg=args.group_reg, stride=args.stride,
                   constant_descriptor=args.constant_descriptor, group_reg_y=args.group_reg_y,
                   group_reg_z=args.group_reg_z)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="ascii")
    print(json.dumps({"container_sha256": hashlib.sha256(blob).hexdigest(),
                      "native_sha256": hashlib.sha256(code).hexdigest(), "numthreads": list(group)}))


if __name__ == "__main__":
    main()
