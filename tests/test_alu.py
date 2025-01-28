# Copyright (c) 2025- Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def rotate_right(n: int, s: int) -> int:
    return ((n << (32 - s)) | (n >> s)) & 0xFFFFFFFF


def count_leading_zeros(n: int) -> int:
    bit = 0x80000000
    count = 0
    while bit != n & bit:
        count += 1
        bit >>= 1
    return count


ops: dict[str | None, Callable[..., Any]] = {
    # binary ops
    "fadd": lambda a, b: a + b,
    "faddnf": lambda a, b: a + b,
    "fsub": lambda a, b: a - b,
    "fmin": np.minimum,
    "fmax": np.maximum,
    "fmul": lambda a, b: a * b,
    "fcmp": lambda a, b: a - b,
    "vfpack": lambda a, b: np.stack([a, b]).T.ravel(),
    "vfmin": np.minimum,
    "vfmax": np.maximum,
    "vfmul": lambda a, b: a * b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "imin": np.minimum,
    "imax": np.maximum,
    "umin": np.minimum,
    "umax": np.maximum,
    "shl": lambda a, b: a << (b % 32),
    "shr": lambda a, b: a >> (b % 32),
    "asr": lambda a, b: a.astype(np.int32) >> (b % 32),
    "ror": lambda a, b: np.vectorize(rotate_right)(a, b % 32),
    "band": lambda a, b: a & b,
    "bor": lambda a, b: a | b,
    "bxor": lambda a, b: a ^ b,
    "vadd": lambda a, b: a + b,
    "vsub": lambda a, b: a - b,
    "quad_rotate": lambda a, b: np.hstack([np.roll(x, -int(y % 4)) for x, y in zip(a.reshape(4, 4), b[::4])]),
    "rotate": lambda a, b: np.roll(a, -int(b[0] % 16)),
    "shuffle": lambda a, b: a[b % 16],
    # unary ops
    "fmov": lambda x: x,
    "fround": np.round,
    "ftrunc": np.trunc,
    "ffloor": np.floor,
    "fceil": np.ceil,
    "fdx": lambda x: (x[1::2] - x[0::2]).repeat(2),
    "fdy": lambda x: (lambda a: (a[1::2] - a[0::2]).ravel())(x.reshape(-1, 2).repeat(2, axis=0).reshape(-1, 4)),
    "ftoin": lambda x: x.round().astype(np.int32),
    "ftoiz": lambda x: np.float32(x).astype(np.int32),
    "ftouz": np.vectorize(lambda x: np.float32(x).astype(np.uint32) if x > -1 else 0),
    "bnot": lambda x: ~x,
    "neg": lambda x: -x,
    "itof": lambda x: x.astype(np.float32),
    "clz": np.vectorize(count_leading_zeros),
    "utof": lambda x: x.astype(np.float32),
    "recip": lambda x: 1 / x,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "exp": lambda x: np.exp2(x),
    "log": lambda x: np.log2(x),
    "sin": lambda x: np.sin(np.pi * x),
    "rsqrt2": lambda x: 1 / np.sqrt(x),
    "ballot": lambda x: np.full(x.shape, ((x != 0) << np.arange(x.size)).sum()),
    "bcastf": lambda x: np.full(x.shape, x[0]),
    "alleq": lambda x: np.full(x.shape, int(all(x.view(np.uint32) == x[0].view(np.uint32)))),
    "allfeq": lambda x: np.full(x.shape, int(all(x == x[0]))),
    # pack/unpack flags
    "l": lambda x: x[0::2],
    "h": lambda x: x[1::2],
    None: lambda x: x,
    "none": lambda x: x,
    "abs": np.abs,
    "r32": lambda x: x.repeat(2),
    "rl2h": lambda x: x[0::2].repeat(2),
    "rh2l": lambda x: x[1::2].repeat(2),
    "swap": lambda x: x.reshape(-1, 2)[:, ::-1].ravel(),
}


@qpu
def qpu_binary_ops(
    asm: Assembly,
    bin_ops: list[str],
    dst_ops: list[str | None],
    src1_ops: list[str | None],
    src2_ops: list[str | None],
) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)
    add(rf2, rf2, rf10)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, rf13)
    nop()
    mov(tmua, rf1, sig=thrsw).add(rf1, rf1, rf13)
    nop(sig=ldtmu(rf11))
    nop()
    nop(sig=ldtmu(rf12))

    g = globals()
    for op, pack, unpack1, unpack2 in itertools.product(bin_ops, dst_ops, src1_ops, src2_ops):
        g[op](
            rf10.pack(pack) if pack is not None else rf10,
            rf11.unpack(unpack1) if unpack1 is not None else rf11,
            rf12.unpack(unpack2) if unpack2 is not None else rf12,
        )
        mov(tmud, rf10)
        mov(tmua, rf2)
        tmuwt().add(rf2, rf2, rf13)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def boilerplate_binary_ops(
    bin_ops: list[str],
    dst: tuple[type, list[str | None]],
    src1: tuple[type, list[str | None]],
    src2: tuple[type, list[str | None]],
    domain1: tuple[Any, Any] | None = None,
    domain2: tuple[Any, Any] | None = None,
) -> None:
    dst_dtype, dst_ops = dst
    src1_dtype, src1_ops = src1
    src2_dtype, src2_ops = src2

    with Driver() as drv:
        cases = list(itertools.product(bin_ops, dst_ops, src1_ops, src2_ops))

        code = drv.program(lambda asm: qpu_binary_ops(asm, bin_ops, dst_ops, src1_ops, src2_ops))
        x1: Array[Any] = drv.alloc((16 * 4 // np.dtype(src1_dtype).itemsize,), dtype=src1_dtype)
        x2: Array[Any] = drv.alloc((16 * 4 // np.dtype(src2_dtype).itemsize,), dtype=src2_dtype)
        y: Array[Any] = drv.alloc((len(cases), 16 * 4 // np.dtype(dst_dtype).itemsize), dtype=dst_dtype)
        unif: Array[np.uint32] = drv.alloc(3, dtype="uint32")

        if domain1 is None:
            if np.dtype(src1_dtype).name.startswith("float"):
                domain1 = (-(2**7), 2**7)
            else:
                info1 = np.iinfo(src1_dtype)
                domain1 = (info1.min, info1.max - int(not np.dtype(src1_dtype).name.startswith("float")))

        if domain2 is None:
            if np.dtype(src2_dtype).name.startswith("float"):
                domain2 = (-(2**7), 2**7)
            else:
                info2 = np.iinfo(src2_dtype)
                domain2 = (info2.min, info2.max - int(not np.dtype(src2_dtype).name.startswith("float")))

        if domain1[0] == domain1[1]:
            x1[:] = domain1[0]
        elif domain1[0] < domain1[1]:
            x1[:] = np.random.uniform(domain1[0], domain1[1], x1.shape).astype(src1_dtype)
        else:
            raise ValueError("Invalid domain")

        if domain2[0] == domain2[1]:
            x2[:] = domain2[0]
        elif domain2[0] < domain2[1]:
            x2[:] = np.random.uniform(domain2[0], domain2[1], x2.shape).astype(src2_dtype)
        else:
            raise ValueError("Invalid domain")

        # if np.dtype(dst_dtype).name.startswith("float"):
        #     x1[:] = np.random.uniform(-(2**7), 2**7, x1.shape).astype(src1_dtype)
        #     x2[:] = np.random.uniform(-(2**7), 2**7, x2.shape).astype(src2_dtype)
        # elif any(np.dtype(dst_dtype).name.startswith(t) for t in ["int", "uint"]):
        #     info1 = np.iinfo(src1_dtype)
        #     info2 = np.iinfo(src2_dtype)
        #     x1[:] = np.random.randint(info1.min, info1.max - 1, x1.shape, dtype=src1_dtype)
        #     x2[:] = np.random.randint(info2.min, info2.max - 1, x2.shape, dtype=src2_dtype)
        # else:
        #     assert False, "unreachable"
        y[:] = 0.0

        unif[0] = x1.addresses()[0]
        unif[1] = x2.addresses()[0]
        unif[2] = y.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        for ix, (bin_op, dst_op, src1_op, src2_op) in enumerate(cases):
            msg = f"{bin_op}({dst_op}, {src1_op}, {src2_op})"
            if np.dtype(dst_dtype).name.startswith("float"):
                assert np.allclose(ops[dst_op](y[ix]), ops[bin_op](ops[src1_op](x1), ops[src2_op](x2)), rtol=1e-2), msg
            elif np.dtype(dst_dtype).name.startswith("int") or np.dtype(dst_dtype).name.startswith("uint"):
                assert np.all(ops[dst_op](y[ix]) == ops[bin_op](ops[src1_op](x1), ops[src2_op](x2))), msg


def test_binary_ops() -> None:
    packs: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none", "abs"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ["fadd", "faddnf", "fsub", "fmin", "fmax"],
            dst,
            src1,
            src2,
        )
    packs: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        # (np.float16, ["l", "h"]), # TODO: Why does this fail?
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none", "abs"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ["fmul", "fcmp"],
            dst,
            src1,
            src2,
        )
    packs: list[tuple[type, list[str | None]]] = [
        (np.float16, [None, "none"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src1, src2 in itertools.product(packs, unpacks, unpacks):
        boilerplate_binary_ops(
            ["vfpack"],
            dst,
            src1,
            src2,
        )
    packs: list[tuple[type, list[str | None]]] = [
        (np.float16, [None, "none"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, ["r32"]),
        (np.float16, ["rl2h", "rh2l", "swap"]),
    ]
    for dst, src1, src2 in itertools.product(packs, unpacks, packs):
        boilerplate_binary_ops(
            ["vfmin", "vfmax", "vfmul"],
            dst,
            src1,
            src2,
        )

    boilerplate_binary_ops(
        ["add", "sub", "imin", "imax", "asr"],
        (np.int32, [None, "none"]),
        (np.int32, [None, "none"]),
        (np.int32, [None, "none"]),
    )
    boilerplate_binary_ops(
        ["add", "sub", "umin", "umax"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
    )
    boilerplate_binary_ops(
        ["vadd", "vsub"],
        (np.int16, [None, "none"]),
        (np.int16, [None, "none"]),
        (np.int16, [None, "none"]),
    )
    boilerplate_binary_ops(
        ["shl", "shr", "ror"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
    )
    boilerplate_binary_ops(
        ["band", "bor", "bxor"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
    )

    boilerplate_binary_ops(
        ["quad_rotate", "rotate", "shuffle"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        domain1=(0, 1024),
        domain2=(0, 15),
    )


@qpu
def qpu_unary_ops(asm: Assembly, bin_ops: list[str], dst_ops: list[str | None], src_ops: list[str | None]) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, rf13)
    nop()
    nop()
    nop(sig=ldtmu(rf11))

    g = globals()
    for op, pack, unpack in itertools.product(bin_ops, dst_ops, src_ops):
        g[op](
            rf10.pack(pack) if pack is not None else rf10,
            rf11.unpack(unpack) if unpack is not None else rf11,
        )
        mov(tmud, rf10)
        mov(tmua, rf1)
        tmuwt().add(rf1, rf1, rf13)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def boilerplate_unary_ops(
    uni_ops: list[str],
    dst: tuple[type, list[str | None]],
    src: tuple[type, list[str | None]],
    domain: tuple[Any, Any] = (-(2**15), 2**15),
    one_of: list[Any] | None = None,
) -> None:
    dst_dtype, dst_ops = dst
    src_dtype, src_ops = src

    with Driver() as drv:
        cases = list(itertools.product(uni_ops, dst_ops, src_ops))

        code = drv.program(lambda asm: qpu_unary_ops(asm, uni_ops, dst_ops, src_ops))
        x: Array[Any] = drv.alloc((16 * 4 // np.dtype(src_dtype).itemsize,), dtype=src_dtype)
        y: Array[Any] = drv.alloc((len(cases), 16 * 4 // np.dtype(dst_dtype).itemsize), dtype=dst_dtype)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        if one_of is None:
            if domain[0] == domain[1]:
                x[:] = domain[0]
            elif domain[0] < domain[1]:
                x[:] = np.random.uniform(domain[0], domain[1], x.shape).astype(src_dtype)
            else:
                raise ValueError("Invalid domain")
        else:
            x[:] = np.random.choice(one_of, x.shape).astype(src_dtype)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        for ix, (uni_op, dst_op, src_op) in enumerate(cases):
            msg = f"{uni_op}({dst_op}, {src_op})"
            if np.dtype(dst_dtype).name.startswith("float"):
                assert np.allclose(ops[dst_op](y[ix]), ops[uni_op](ops[src_op](x)), rtol=1e-2), msg
            elif np.dtype(dst_dtype).name.startswith("int") or np.dtype(dst_dtype).name.startswith("uint"):
                assert np.all(ops[dst_op](y[ix]) == ops[uni_op](ops[src_op](x))), msg


def test_unary_ops() -> None:
    packs: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none", "abs"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["fmov"],
            dst,
            src,
        )
    packs: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["fround", "ftrunc", "ffloor", "fceil", "fdx", "fdy"],
            dst,
            src,
        )

    packs: list[tuple[type, list[str | None]]] = [
        (np.int32, [None, "none"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["ftoin", "ftoiz"],
            dst,
            src,
        )
    packs: list[tuple[type, list[str | None]]] = [
        (np.uint32, [None, "none"]),
    ]
    unpacks: list[tuple[type, list[str | None]]] = [
        (np.float32, [None, "none"]),
        (np.float16, ["l", "h"]),
    ]
    for dst, src in itertools.product(packs, unpacks):
        boilerplate_unary_ops(
            ["ftouz"],
            dst,
            src,
        )
    # TODO: 'ftoc': what is the meaning of this instruction ?
    # packs = [('int32', ['none'])]
    # unpacks = [('float32', ['none']), ('float16', ['l', 'h'])]
    # for dst, src in itertools.product(packs, unpacks):
    #     boilerplate_unary_ops(
    #         ['ftoc'],
    #         dst, src,
    #     )

    boilerplate_unary_ops(
        ["bnot", "neg"],
        (np.int32, [None]),
        (np.int32, [None]),
    )
    boilerplate_unary_ops(
        ["itof"],
        (np.float32, [None]),
        (np.int32, [None]),
    )
    boilerplate_unary_ops(
        ["clz"],
        (np.uint32, [None]),
        (np.uint32, [None]),
    )
    boilerplate_unary_ops(
        ["utof"],
        (np.float32, [None]),
        (np.uint32, [None]),
    )

    boilerplate_unary_ops(
        ["recip"],
        (np.float32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(0.001, 1024),
    )
    boilerplate_unary_ops(
        ["recip"],
        (np.float32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(-1024, -0.001),
    )
    boilerplate_unary_ops(
        ["exp"],
        (np.float32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(-1.0, 1.0),
    )
    boilerplate_unary_ops(
        ["sin"],
        (np.float32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(-0.5, 0.5),
    )
    boilerplate_unary_ops(
        ["rsqrt", "log", "rsqrt2"],
        (np.float32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(0.001, 1024),
    )

    boilerplate_unary_ops(
        ["ballot"],
        (np.int32, [None, "none"]),
        (np.int32, [None, "none"]),
        domain=(-3, 3),
    )

    boilerplate_unary_ops(
        ["bcastf"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
    )

    boilerplate_unary_ops(
        ["alleq"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        domain=(123, 123),
    )
    boilerplate_unary_ops(
        ["alleq"],
        (np.uint32, [None, "none"]),
        (np.uint32, [None, "none"]),
        domain=(0, 10),
    )

    boilerplate_unary_ops(
        ["allfeq"],
        (np.uint32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(123, 123),
    )
    boilerplate_unary_ops(
        ["allfeq"],
        (np.uint32, [None, "none"]),
        (np.float32, [None, "none"]),
        domain=(0, 10),
    )
    boilerplate_unary_ops(
        ["allfeq"],
        (np.uint32, [None, "none"]),
        (np.float32, [None, "none"]),
        one_of=[np.float32("-0"), np.float32("0")],
    )
