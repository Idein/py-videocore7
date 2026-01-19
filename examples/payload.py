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
import sys

import numpy as np

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def next_power_of_two_u32(x: int) -> int:
    assert x > 0
    x &= 0xFFFFFFFF
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


def find_first_bit_u32(x: int) -> int:
    assert x > 0
    x &= 0xFFFFFFFF
    lsb = x & (-x & 0xFFFFFFFF)
    idx = sum(
        [
            [0, 1, 28, 2, 29, 14, 24, 3],
            [30, 22, 20, 15, 25, 17, 4, 8],
            [31, 27, 13, 23, 21, 19, 16, 7],
            [26, 12, 18, 6, 11, 5, 10, 9],
        ],
        [],
    )
    return idx[((lsb * 0x077CB531) & 0xFFFFFFFF) >> 27]


local_x = 8
local_y = 1
local_z = 1


@qpu
def gpu_code(asm: Assembly) -> None:
    # Compute Shader Payload
    reg_wg_x = rf3.unpack("ul")
    reg_wg_y = rf3.unpack("uh")
    reg_wg_z = rf2.unpack("ul")
    reg_iid = rf2.unpack("uh")

    reg_wg_x_size = rf4
    reg_wg_y_size = rf5
    reg_wg_z_size = rf6

    reg_base = rf9
    reg_stride = rf10

    reg_work_id = rf11
    reg_id = rf12

    nop(sig=ldunifrf(reg_wg_x_size))
    nop(sig=ldunifrf(reg_wg_y_size))
    nop(sig=ldunifrf(reg_wg_z_size))

    nop(sig=ldunifrf(reg_base))
    nop(sig=ldunifrf(reg_stride))

    # work_id = (wg_z * wg_y_size + wg_y) * wg_x_size + wg_x
    mov(reg_work_id, reg_wg_z)
    mov(rf0, reg_wg_y).umul24(reg_work_id, reg_work_id, reg_wg_y_size)
    add(reg_work_id, reg_work_id, rf0)
    mov(rf0, reg_wg_x).umul24(reg_work_id, reg_work_id, reg_wg_x_size)
    add(reg_work_id, reg_work_id, rf0)

    umul24(reg_id, reg_work_id, reg_wg_x_size)
    umul24(reg_id, reg_id, reg_wg_y_size)
    umul24(reg_id, reg_id, reg_wg_z_size)

    umul24(rf0, reg_work_id, reg_stride)
    add(reg_base, reg_base, rf0)

    eidx(rf0)
    shl(rf0, rf0, 2)
    add(reg_base, reg_base, rf0)

    mov(rf0, 1)
    shl(rf0, rf0, 6)

    mov(tmud, reg_wg_x)
    mov(tmua, reg_base).add(reg_base, reg_base, rf0)
    mov(tmud, reg_wg_y)
    mov(tmua, reg_base).add(reg_base, reg_base, rf0)
    mov(tmud, reg_wg_z)
    mov(tmua, reg_base).add(reg_base, reg_base, rf0)
    mov(rf1, reg_iid)
    shr(tmud, rf1, 17 - find_first_bit_u32(next_power_of_two_u32(max(local_x * local_y * local_z, 64))) - 1)
    mov(tmua, reg_base).add(reg_base, reg_base, rf0)

    tmuwt()

    L.skip

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def main() -> None:
    with Driver() as drv:
        code = drv.program(gpu_code)

        wg_x = 15
        wg_y = 14
        wg_z = 13

        payload: Array[np.uint32] = drv.alloc((wg_z, wg_y, wg_x, 4, 16), dtype=np.uint32)
        payload[:] = 0

        unif: Array[np.uint32] = drv.alloc((5,), dtype=np.uint32)
        unif[0] = wg_x
        unif[1] = wg_y
        unif[2] = wg_z
        unif[3] = payload.addresses().item(0)
        unif[4] = payload.strides[2]

        drv.execute(
            code,
            local_invocation=(local_x, local_y, local_z),
            uniforms=unif.addresses().item(0),
            workgroup=(wg_x, wg_y, wg_z),
            wgs_per_sg=24,
            thread=wg_x * wg_y * wg_z,
        )

        np.set_printoptions(threshold=sys.maxsize)
        print(payload)


if __name__ == "__main__":
    main()
