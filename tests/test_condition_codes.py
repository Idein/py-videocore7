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
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


# `cond = 'push*'` sets the conditional flag A
@qpu
def qpu_cond_push_a(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf12))
    shl(rf10, rf10, 2)
    add(rf12, rf12, rf10)
    mov(rf11, 4)
    shl(rf11, rf11, 4)

    cond_pairs: list[tuple[ALUConditionLiteral, ALUConditionLiteral]] = [
        ("pushz", "ifa"),
        ("pushn", "ifna"),
        ("pushc", "ifa"),
    ]

    for cond_push, cond_if in cond_pairs:
        eidx(rf10)
        sub(rf10, rf10, 10, cond=cond_push)
        mov(rf10, 0)
        mov(rf10, 1, cond=cond_if)
        mov(tmud, rf10)
        mov(tmua, rf12)
        tmuwt().add(rf12, rf12, rf11)
        mov(rf10, 0)
        nop().mov(rf10, 1, cond=cond_if)
        mov(tmud, rf10)
        mov(tmua, rf12)
        tmuwt().add(rf12, rf12, rf11)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_cond_push_a() -> None:
    with Driver() as drv:
        code = drv.program(qpu_cond_push_a)
        data: Array[np.uint32] = drv.alloc((6, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 0

        unif[0] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        pushz_if_expected = np.zeros((16,), dtype=np.uint32)
        pushz_if_expected[10] = 1

        pushn_ifn_expected = np.zeros((16,), dtype=np.uint32)
        pushn_ifn_expected[10:] = 1

        pushc_if_expected = np.zeros((16,), dtype=np.uint32)
        pushc_if_expected[:10] = 1

        assert np.all(data[0] == pushz_if_expected)
        assert np.all(data[1] == pushz_if_expected)
        assert np.all(data[2] == pushn_ifn_expected)
        assert np.all(data[3] == pushn_ifn_expected)
        assert np.all(data[4] == pushc_if_expected)
        assert np.all(data[5] == pushc_if_expected)


# `cond = 'push*'` moves the old conditional flag A to B
@qpu
def qpu_cond_push_b(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf12))
    shl(rf10, rf10, 2)
    add(rf12, rf12, rf10)
    mov(rf11, 4)
    shl(rf11, rf11, 4)

    eidx(rf10)
    sub(null, rf10, 10, cond="pushz")
    mov(rf10, 0, cond="ifa")
    eidx(rf10).mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    eidx(rf10)
    sub(null, rf10, 5, cond="pushz")
    mov(rf10, 0, cond="ifa")
    eidx(rf10).mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)
    mov(rf10, 0, cond="ifb")
    eidx(rf10).mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    eidx(rf10)
    sub(null, rf10, 1, cond="pushz")
    mov(rf10, 0, cond="ifa")
    eidx(rf10).mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)
    mov(rf10, 0, cond="ifb")
    eidx(rf10).mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_cond_push_b() -> None:
    with Driver() as drv:
        code = drv.program(qpu_cond_push_b)
        data: Array[np.uint32] = drv.alloc((5, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 0

        unif[0] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        push0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15]
        push1 = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        push2 = [0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        expected = np.array(
            #  pushz
            [
                push0,  # ifa
                # pushz
                push1,  # ifa
                push0,  # ifb
                # pushz
                push2,  # ifa
                push1,
            ],  # ifb
            dtype=np.uint32,
        )

        assert np.all(data == expected)


# `cond = '{and,nor}*'` updates the conditional flag A and it don't affect to B
@qpu
def qpu_cond_update(asm: Assembly, cond_update_flags: list[ALUConditionLiteral]) -> None:
    eidx(rf10, sig=ldunifrf(rf12))
    shl(rf10, rf10, 2)
    add(rf12, rf12, rf10)
    mov(rf11, 4)
    shl(rf11, rf11, 4)

    for cond_update_flag in cond_update_flags:
        eidx(rf10)
        band(rf10, rf10, 1, cond="pushz")  # fla = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        eidx(rf10)
        sub(null, rf10, 5, cond=cond_update_flag)
        mov(rf10, 0)
        mov(rf10, 1, cond="ifa")
        mov(tmud, rf10)
        mov(tmua, rf12)
        tmuwt().add(rf12, rf12, rf11)

    for cond_update_flag in cond_update_flags:
        eidx(rf10)
        band(rf10, rf10, 1, cond="pushz")
        eidx(rf10)
        add(rf13, rf10, rf10).sub(rf10, rf10, 5, cond=cond_update_flag)
        mov(rf10, 0)
        mov(rf10, 1, cond="ifa")
        mov(tmud, rf10)
        mov(tmua, rf12)
        tmuwt().add(rf12, rf12, rf11)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_cond_update() -> None:
    cond_update_flags: list[ALUConditionLiteral] = [
        "andz",
        "andnz",
        "nornz",
        "norz",
        "andn",
        "andnn",
        "nornn",
        "norn",
        "andc",
        "andnc",
        "nornc",
        "norc",
    ]

    def cond_update_op[T: np.generic](
        cond_update_flag: str,
    ) -> Callable[[npt.NDArray[T], npt.NDArray[T]], npt.NDArray[np.bool]]:
        bin_op: Callable[[npt.NDArray[T], npt.NDArray[T]], npt.NDArray[np.bool]] = [
            lambda a, b: np.logical_not(np.logical_or(a, b)),  # xor
            lambda a, b: np.logical_and(a, b),  # and
        ][cond_update_flag[:3] == "and"]
        b_op: Callable[[npt.NDArray[T]], npt.NDArray[np.bool]] = [
            lambda b: b < 0,  # negative
            lambda b: b == 0,  # zero
        ][cond_update_flag[-1] == "z"]
        not_op: Callable[[npt.NDArray[np.bool]], npt.NDArray[np.bool]] = [
            lambda x: x,  # id
            lambda x: np.logical_not(x),  # not
        ][cond_update_flag[3:-1] == "n"]
        return lambda a, b: bin_op(a, not_op(b_op(b)))

    with Driver() as drv:
        code = drv.program(qpu_cond_update, cond_update_flags)
        data: Array[np.uint32] = drv.alloc((len(cond_update_flags), 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 0

        unif[0] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        a = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) > 0
        b = np.arange(16) - 5

        expected = np.vstack([cond_update_op(flag)(a, b) for flag in cond_update_flags])

        assert np.all(data == expected)


# dual `cond=''` instruction
@qpu
def qpu_cond_combination(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf12))
    shl(rf10, rf10, 2)
    add(rf12, rf12, rf10)
    mov(rf11, 4)
    shl(rf11, rf11, 4)

    # if / push
    eidx(rf10)
    sub(rf10, rf10, 10, cond="pushz")
    eidx(rf10)
    mov(rf15, 5)
    mov(rf10, rf15, cond="ifa").sub(rf13, rf10, rf15, cond="pushn")
    mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)
    eidx(rf10)
    mov(rf10, 0, cond="ifa")
    mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    # push / if
    eidx(rf10)
    sub(rf10, rf10, 10, cond="pushz")
    eidx(rf10)
    mov(rf15, 5)
    sub(null, rf10, rf15, cond="pushn").mov(rf10, rf15, cond="ifa")
    mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)
    eidx(rf10)
    mov(rf10, 0, cond="ifa")
    mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    # if / if
    eidx(rf10)
    sub(null, rf10, 10, cond="pushn")
    eidx(rf13)
    mov(rf15, 0)
    mov(rf10, rf15, cond="ifna").mov(rf13, rf15, cond="ifna")
    mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)
    mov(tmud, rf13)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    # update / if
    eidx(rf10)
    sub(null, rf10, 10, cond="pushn")
    eidx(rf13)
    mov(rf15, 5)
    sub(null, rf10, rf15, cond="andn").mov(rf13, rf15, cond="ifa")
    eidx(rf10)
    mov(rf10, 0, cond="ifa")
    mov(tmud, rf10)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)
    mov(tmud, rf13)
    mov(tmua, rf12)
    tmuwt().add(rf12, rf12, rf11)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_cond_combination() -> None:
    with Driver() as drv:
        code = drv.program(qpu_cond_combination)
        data: Array[np.uint32] = drv.alloc((8, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 0

        unif[0] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        expected = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 11, 12, 13, 14, 15],
                [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 11, 12, 13, 14, 15],
                [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 11, 12, 13, 14, 15],
            ],
            dtype=np.uint32,
        )

        assert np.all(data == expected)


# vflx instructions read a condition flag as int16
@qpu
def qpu_cond_vflx(asm: Assembly, ops: list[str]) -> None:
    eidx(rf10, sig=ldunifrf(rf12))
    shl(rf10, rf10, 2)
    add(rf12, rf12, rf10)
    mov(rf11, 4)
    shl(rf11, rf11, 4)

    # init fla/flb
    bxor(rf0, rf0, rf0).sub(rf1, rf1, rf1)
    eidx(rf10)
    band(null, rf10, 1 << 0, cond="pushz")  # a = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    band(
        null, rf10, 1 << 1, cond="pushz"
    )  # a = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0], b = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]

    # flapush
    g = globals()
    for op in ops:
        g[op](rf10)
        mov(tmud, rf10)
        mov(tmua, rf12)
        tmuwt().add(rf12, rf12, rf11)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_cond_vflx() -> None:
    def expected(op: str) -> npt.NDArray[np.int16]:
        result = [
            np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int16),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int16),
        ][op[-1] == "b"].repeat(2)
        if op[3:-1] == "n":
            result = 1 - result
        return result

    ops = [
        "vfla",
        "vflna",
        "vflb",
        "vflnb",
    ]

    with Driver() as drv:
        code = drv.program(qpu_cond_vflx, ops)
        data: Array[np.int16] = drv.alloc((len(ops), 32), dtype=np.int16)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 0

        unif[0] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        for ix, op in enumerate(ops):
            assert np.all(data[ix] == expected(op))


@qpu
def qpu_cond_flx(asm: Assembly, ops: list[str]) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))  # in
    nop(sig=ldunifrf(rf2))  # out
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

    # init fla/flb
    mov(null, rf12, cond="pushn")
    band(null, rf12, 1, cond="pushz")  # fla, flb = ~(rf12 & 1), rf12 < 0

    g = globals()
    for op in ops:
        g[op](tmud, rf11)
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


def test_cond_flx() -> None:
    ops = [
        "flapush",
        "flbpush",
        "flpop",
    ]

    with Driver() as drv:
        code = drv.program(qpu_cond_flx, ops)
        x1: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        x2: Array[np.int32] = drv.alloc((16,), dtype=np.int32)
        y: Array[np.uint32] = drv.alloc((len(ops), 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x1[:] = (np.random.randn(*x1.shape) * (2**24)).astype(np.uint32)
        x2[:] = np.random.randn(*x2.shape).astype(np.int32)
        y[:] = 0.0

        unif[0] = x1.addresses()[0]
        unif[1] = x2.addresses()[0]
        unif[2] = y.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        expected = np.vstack(
            [
                (x1 << 2) | (3 * (1 - x2 & 1)),
                (x1 << 2) | (3 * (x2 < 0)),
                x1 >> 2,
            ]
        )

        assert np.all(y == expected)
