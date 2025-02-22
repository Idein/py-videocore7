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
from collections import deque
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np

# from videocore7 import pack_unpack
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


@qpu
def qpu_sgemm_rnn_naive(asm: Assembly) -> None:
    # rf0 - rf15: regs
    # rf16 - rf31: 16x16 accumulators
    # rf32 - rf63: (unused)

    reg_tile_i = rf1
    reg_tile_j = rf2

    reg_wg_id = rf2.unpack("ul")  # if workgroup is (1, 1, 16), wg_id = wg_z_id
    mov(rf2, reg_wg_id, sig=ldunif)  # load tile_r
    itof(rf4, rf2)  # rf4 = float(wg_id)
    itof(rf5, rf0)  # rf0 = float(tile_r)
    recip(rf3, rf5)  # rf3 = 1 / float(tile_r)
    fmul(rf1, rf4, rf3)  # rf1=  wg_id / tile_r
    ffloor(rf1, rf1)  # rf1 = floor(wg_id / tile_r)
    ftouz(reg_tile_i, rf1)  # rf1 = i (/ tile_p)
    umul24(rf3, rf1, rf0)  # rf3 = i * tile_r
    sub(reg_tile_j, rf2, rf3)  # rf2 = j (/ tile_r)

    # params
    # rf0: free
    # rf1: reg_tile_i
    # rf2: reg_tile_j
    # rf3: free
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11  # use after reg_c_stride is released
    reg_a_stride = rf9  # use for init, unused in loop
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10  # use for init, unused in loop
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]  # accumulators

    # a_base = a[i,:] = a_item0_addr + 16 * i * a_stride
    # b_base = b[:,j] = b_item0_addr + 16 * j * 4
    nop(sig=ldunifrf(reg_a_stride))  # load a_stride
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))  # load a_item0 addr
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))  # load b_stride
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))  # load b_item0 addr
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)  # eidx for next block

    # calc base addr of A and B
    # eidx(rf3)  # merged pred instruction
    umul24(rf4, rf3, reg_a_stride)
    del reg_a_stride  # reg_a_stride is released
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)  # keep 4*b_stride
    del reg_b_stride  # reg_b_stride is released
    add(reg_b_base, reg_b_base, rf3)

    # start loading A
    bnot(tmuc, 3)
    mov(tmua, reg_a_base)

    # start loading B
    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    # calc base addr of C
    nop(sig=ldunifrf(reg_c_stride))  # load c_stride
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i  # reg_tile_i is released
    del reg_tile_j  # reg_tile_j is released
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))  # c_item0_addr
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride  # reg_c_stride is released
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)  # load q
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)  # c_base[0] = c_item0_addr + 16 * i * c_stride + 16 * j * 4

    for i in range(8):
        r1 = reg_accum[i]
        r2 = reg_accum[i + 8]
        bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu((reg_a + reg_b)[i]))  # load reg_a and reg_b
    with loop as lk:
        # start loading next A
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        # start loading next B
        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b_order = deque(reg_b * 16)

        def rotate_broadcast_reg_b() -> None:
            r = rotate_broadcast_reg_b_order.popleft()
            rotate(r, r, 1).mov(rep, r)

        # 16x4x16 FMA
        rotate_broadcast_reg_b()
        sub(reg_i, reg_i, 1, cond="pushz").fmul(rf1, rf0, reg_a[0])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[0])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[1])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[1])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[2])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[2])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[3])
        for i in range(8):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[8], reg_accum[8], rf1, sig=ldtmu(reg_a[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[9], reg_accum[9], rf1, sig=ldtmu(reg_a[1])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[10], reg_accum[10], rf1, sig=ldtmu(reg_a[2])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).fmul(rf1, rf0, reg_a[3])  # reg_a[3] is using
        rotate_broadcast_reg_b()
        fadd(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).fmul(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        rotate_broadcast_reg_b()  # delay slot
        fadd(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).fmul(rf1, rf0, reg_a[3])  # delay slot
        fadd(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)  # delay slot

    # rf3-14 is released
    del reg_a
    del reg_b
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b_base

    # load C and store αAB+βC
    reg_tmuc_vec_4_cfg = rf12
    reg_alpha = rf13
    reg_beta = rf14
    bnot(reg_tmuc_vec_4_cfg, 3)  # pixel, regular, vec4
    nop(sig=ldunifrf(reg_alpha))
    mov(tmuc, reg_tmuc_vec_4_cfg, sig=ldunifrf(reg_beta)).fmul(reg_accum[0], reg_accum[0], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[1], reg_accum[1], reg_alpha)
    fmul(reg_accum[2], reg_accum[2], reg_alpha)
    fmul(reg_accum[3], reg_accum[3], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[0], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[1], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[2], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[3], rf2)
    mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[4], reg_accum[4], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[5], reg_accum[5], reg_alpha)
    fmul(reg_accum[6], reg_accum[6], reg_alpha)
    fmul(reg_accum[7], reg_accum[7], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[4], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[5], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[6], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[7], rf2)
    mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[8], reg_accum[8], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[9], reg_accum[9], reg_alpha)
    fmul(reg_accum[10], reg_accum[10], reg_alpha)
    fmul(reg_accum[11], reg_accum[11], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[8], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[9], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[10], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[11], rf2)
    mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[12], reg_accum[12], reg_alpha)
    mov(tmua, reg_c_base, sig=thrsw)
    fmul(reg_accum[13], reg_accum[13], reg_alpha)
    fmul(reg_accum[14], reg_accum[14], reg_alpha)
    fmul(reg_accum[15], reg_accum[15], reg_alpha, sig=ldtmu(rf1))
    mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[12], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[13], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
    fadd(tmud, reg_accum[14], rf2).fmul(rf2, rf1, reg_beta)
    fadd(tmud, reg_accum[15], rf2)
    mov(tmua, reg_c_base)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def sgemm_rnn_naive() -> None:
    p = 1024
    q = 1024
    r = 1024

    assert p % 16 == 0
    assert q % 4 == 0
    assert r % 16 == 0

    tile_p = p // 16
    tile_r = r // 16

    thread = tile_p * tile_r

    with Driver() as drv:
        code = drv.program(qpu_sgemm_rnn_naive)

        a: Array[np.float32] = drv.alloc((p, q), dtype=np.float32)
        b: Array[np.float32] = drv.alloc((q, r), dtype=np.float32)
        c: Array[np.float32] = drv.alloc((p, r), dtype=np.float32)

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        a_ref = np.random.randn(*a.shape).astype(a.dtype)
        b_ref = np.random.randn(*b.shape).astype(b.dtype)
        c_ref = np.random.randn(*c.shape).astype(c.dtype)
        expected = np.empty(c.shape, dtype=c.dtype)

        a[:] = a_ref
        b[:] = b_ref
        c[:] = c_ref

        start = getsec()
        expected[:] = alpha * a_ref.dot(b_ref) + beta * c_ref
        time_ref = getsec() - start

        unif: Array[np.uint32] = drv.alloc(10, dtype=np.uint32)
        unif[0] = tile_r
        unif[1] = a.strides[0]
        unif[2] = a.addresses().item(0)
        unif[3] = b.strides[0]
        unif[4] = b.addresses().item(0)
        unif[5] = c.strides[0]
        unif[6] = c.addresses().item(0)
        unif[7] = q
        unif[8] = np.float32(alpha).view(np.uint32).item()
        unif[9] = np.float32(beta).view(np.uint32).item()

        start = getsec()
        drv.execute(code, unif.addresses().item(0), workgroup=(1, 1, 16), wgs_per_sg=24, thread=thread)
        time_gpu = getsec() - start

        # np.set_printoptions(threshold=100000)
        # print(c)

        def gflops(sec: float) -> float:
            return (2 * p * q * r + 3 * p * r) / sec * 1e-9

        print(f"==== sgemm example ({p}x{q} times {q}x{r}) ====")
        print(f"numpy: {time_ref:.4f} sec, {gflops(time_ref):.4f} Gflop/s")
        print(f"QPU:   {time_gpu:.4f} sec, {gflops(time_gpu):.4f} Gflop/s")
        print(f"Minimum absolute error: {np.min(np.abs(c - expected))}")
        print(f"Maximum absolute error: {np.max(np.abs(c - expected))}")
        print(f"Minimum relative error: {np.min(np.abs((c - expected) / expected))}")
        print(f"Maximum relative error: {np.max(np.abs((c - expected) / expected))}")


def main() -> None:
    sgemm_rnn_naive()


if __name__ == "__main__":
    main()
