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
import numpy as np

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


# branch (destination from relative imm)
@qpu
def qpu_branch_rel_imm(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf11))

    b(2 * 8, cond="always")
    nop()
    nop()
    nop()
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)  # jump comes here
    add(rf11, rf11, 1)

    mov(tmud, rf11)
    mov(tmua, rf1)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_branch_rel_imm() -> None:
    with Driver() as drv:
        code = drv.program(qpu_branch_rel_imm)
        x: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == x + 2)


# branch (destination from absolute imm)
@qpu
def qpu_branch_abs_imm(asm: Assembly, absimm: int) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf11))

    b(absimm, absolute=True, cond="always")
    nop()
    nop()
    nop()
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)  # jump comes here
    add(rf11, rf11, 1)

    mov(tmud, rf11)
    mov(tmua, rf1)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_branch_abs_imm() -> None:
    with Driver() as drv:

        @qpu
        def qpu_dummy(asm: Assembly) -> None:
            nop()

        dummy = drv.program(qpu_dummy)
        code = drv.program(qpu_branch_abs_imm, int(dummy.addresses()[0] + 16 * 8))
        x: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == x + 2)


# branch (destination from label)
@qpu
def qpu_branch_rel_label(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf11))

    b(R.foo, cond="always")
    nop()
    nop()
    nop()
    add(rf11, rf11, 1)
    L.foo
    add(rf11, rf11, 1)  # jump comes here
    L.bar
    add(rf11, rf11, 1)
    L.baz
    add(rf11, rf11, 1)

    mov(tmud, rf11)
    mov(tmua, rf1)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_branch_rel_label() -> None:
    with Driver() as drv:
        code = drv.program(qpu_branch_rel_label)
        x: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.arange(16)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == x + 3)


# branch (destination from regfile)
@qpu
def qpu_branch_abs_reg(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf2))

    mov(rf11, 0)
    b(rf2, cond="always")
    nop()
    nop()
    nop()
    L.label
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)
    add(rf11, rf11, 1)  # jump comes here

    mov(tmud, rf11)
    mov(tmua, rf1)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_branch_abs_reg() -> None:
    with Driver() as drv:
        code = drv.program(qpu_branch_abs_reg)
        x: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = code.addresses()[0] + 17 * 8
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == 1)


# branch (destination from link_reg)
@qpu
def qpu_branch_link_reg(asm: Assembly, set_subroutine_link: bool, use_link_reg_direct: bool) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf12))

    mov(rf2, 0)
    mov(rf3, 0)
    b(R.init_link, cond="always", set_link=True)
    nop()  # delay slot
    nop()  # delay slot
    nop()  # delay slot
    L.init_link

    # subroutine returns to here if set_subroutine_link is False.
    add(rf3, rf3, 1)

    # jump to subroutine once.
    mov(null, rf2, cond="pushz")
    b(R.subroutine, cond="alla", set_link=set_subroutine_link)
    mov(rf2, 1)  # delay slot
    nop()  # delay slot
    nop()  # delay slot

    # subroutine returns to here if set_subroutine_link is True.
    mov(rf11, 4)
    shl(rf11, rf11, 4)
    mov(tmud, rf3)  # rf3 will be 1 if set_subroutine_link, else 2.
    mov(tmua, rf1).add(rf1, rf1, rf11)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()

    L.subroutine

    mov(rf11, 4)
    shl(rf11, rf11, 4)
    mov(tmud, rf12)
    mov(tmua, rf1).add(rf1, rf1, rf11)
    tmuwt()

    if use_link_reg_direct:
        b(link, cond="always")
    else:
        lr(rf32)  # lr instruction reads link register
        b(rf32, cond="always")
    nop()  # delay slot
    nop()  # delay slot
    nop()  # delay slot


def test_branch_link_reg() -> None:
    for set_subroutine_link, expected in [(False, 2), (True, 1)]:
        for use_link_reg_direct in [False, True]:
            with Driver() as drv:
                code = drv.program(qpu_branch_link_reg, set_subroutine_link, use_link_reg_direct)
                x: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
                y: Array[np.uint32] = drv.alloc((2, 16), dtype=np.uint32)
                unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)

                x[:] = (np.random.randn(16) * 1024).astype("uint32")
                y[:] = 0.0

                unif[0] = x.addresses()[0]
                unif[1] = y.addresses()[0, 0]

                drv.execute(code, unif.addresses()[0])

                assert np.all(y[0] == x)
                assert np.all(y[1] == expected)


# uniform branch (destination from uniform relative value)
@qpu
def qpu_uniform_branch_rel(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)

    b(R.label, cond="always").unif_addr()
    nop()
    nop()
    nop()
    L.label
    nop(sig=ldunifrf(tmud))
    mov(tmua, rf0)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_uniform_branch_rel() -> None:
    with Driver() as drv:
        code = drv.program(qpu_uniform_branch_rel)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(5, dtype=np.uint32)

        y[:] = 0.0

        unif[0] = y.addresses()[0]
        unif[1] = 8  # relative address for uniform branch
        unif[2] = 5
        unif[3] = 6
        unif[4] = 7  # uniform branch point here

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == 7)


# uniform branch (destination from uniform absolute value)
@qpu
def qpu_uniform_branch_abs(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)

    b(R.label, cond="always").unif_addr(absolute=True)
    nop()
    nop()
    nop()
    L.label
    nop(sig=ldunifrf(tmud))
    mov(tmua, rf0)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_uniform_branch_abs() -> None:
    with Driver() as drv:
        code = drv.program(qpu_uniform_branch_abs)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(5, dtype=np.uint32)

        y[:] = 0.0

        unif[0] = y.addresses()[0]
        unif[1] = unif.addresses()[3]  # absolute address for uniform branch
        unif[2] = 5
        unif[3] = 6  # uniform branch point here
        unif[4] = 7

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == 6)


# uniform branch (destination from register)
@qpu
def qpu_uniform_branch_reg(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf2))

    b(R.label, cond="always").unif_addr(rf2)
    nop()
    nop()
    nop()
    L.label
    nop(sig=ldunifrf(rf3))
    mov(tmud, rf3)
    mov(tmua, rf1)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_uniform_branch_reg() -> None:
    with Driver() as drv:
        code = drv.program(qpu_uniform_branch_reg)
        x: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        y: Array[np.uint32] = drv.alloc((16,), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(6, dtype=np.uint32)

        x[1] = unif.addresses()[4]  # absolute address for uniform branch
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0]
        unif[2] = 3
        unif[3] = 4
        unif[4] = 5  # uniform branch point here
        unif[5] = 6

        drv.execute(code, unif.addresses()[0])

        assert np.all(y == 5)
