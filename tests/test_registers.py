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


@qpu
def qpu_regs_rep(asm: Assembly) -> None:
    nop(sig=ldunifrf(rf1))
    mov(rep, rf1)
    eidx(rf10)
    shl(rf10, rf10, 2)
    mov(tmud, rf0)
    add(tmua, rf1, rf10)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_regs_rep() -> None:
    with Driver() as drv:
        code = drv.program(qpu_regs_rep)
        dst: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        unif[0] = dst.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(dst == unif[0])


@qpu
def qpu_regs_quad(asm: Assembly) -> None:
    nop(sig=ldunifrf(rf1))
    eidx(quad)
    eidx(rf10)
    shl(rf10, rf10, 2)
    mov(tmud, rf0)
    add(tmua, rf1, rf10)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_regis_quad() -> None:
    with Driver() as drv:
        code = drv.program(qpu_regs_quad)
        dst: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        unif[0] = dst.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(dst == np.array([0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]))
