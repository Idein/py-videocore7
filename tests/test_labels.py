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
def qpu_label_with_namespace(asm: Assembly) -> None:
    mov(rf10, 0)

    with namespace("ns1"):
        b(R.test, cond="always")
        nop()
        nop()
        nop()
        add(rf10, rf10, 10)
        L.test
        add(rf10, rf10, 1)

        with namespace("nested"):
            b(R.test, cond="always")
            nop()
            nop()
            nop()
            add(rf10, rf10, 10)
            L.test
            add(rf10, rf10, 1)

    with namespace("ns2"):
        b(R.test, cond="always")
        nop()
        nop()
        nop()
        add(rf10, rf10, 10)
        L.test
        add(rf10, rf10, 1)

    b(R.test, cond="always")
    nop()
    nop()
    nop()
    add(rf10, rf10, 10)
    L.test
    add(rf10, rf10, 1)

    with namespace("ns3"):
        b(R.test, cond="always")
        nop()
        nop()
        nop()
        add(rf10, rf10, 10)
        L.test
        add(rf10, rf10, 1)

    eidx(rf11, sig=ldunifrf(rf2))
    shl(rf11, rf11, 2)

    mov(tmud, rf10)
    add(tmua, rf2, rf11)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_label_with_namespace() -> None:
    with Driver() as drv:
        code = drv.program(qpu_label_with_namespace)
        data: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        data[:] = 1234

        unif[0] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(data == 5)
