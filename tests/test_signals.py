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


# ldtmu
@qpu
def qpu_signal_ldtmu(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10)
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, rf13)  # start load X
    mov(rf10, 1.0)  # rf10 <- 1.0
    mov(rf11, 2.0)  # rf11 <- 2.0
    fadd(rf10, rf10, rf10).fmul(rf11, rf11, rf11, sig=ldtmu(rf31))  # rf10 <- 2 * rf10, rf11 <- rf11 ^ 2, rf31 <- X
    mov(tmud, rf31)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, rf13)
    mov(tmud, rf10)
    mov(tmua, rf1)
    tmuwt().add(rf1, rf1, rf13)
    mov(tmud, rf11)
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


def test_signal_ldtmu() -> None:
    with Driver() as drv:
        code = drv.program(qpu_signal_ldtmu)
        x: Array[np.float32] = drv.alloc((16,), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((3, 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        x[:] = np.random.randn(*x.shape).astype(np.float32)
        y[:] = 0.0

        unif[0] = x.addresses()[0]
        unif[1] = y.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(y[0] == x)
        assert np.all(y[1] == 2)
        assert np.all(y[2] == 4)
