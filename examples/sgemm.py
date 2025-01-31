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
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np

# from videocore7 import pack_unpack
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


@qpu
def load_params(asm: Assembly, thread: int, regs: list[Register]) -> None:
    if thread == 1:
        bxor(rf0, rf0, rf0, sig=ldunifrf(rf6))
        # rf0, rf1 = (0, _)
    elif thread == 12:
        #  8 threads (1 threads / qpu)
        tidx(rf0, sig=ldunifrf(rf6))
        shr(rf0, rf0, 2)
        mov(rf1, 0b1111)
        # rf0, rf1 = (tidx >> 2, 0b1111)
    elif thread == 24:
        # 16 threads (2 threads / qpu)
        tidx(rf0, sig=ldunifrf(rf6))
        shr(rf0, rf0, 1).mov(rf1, 1)
        shl(rf1, rf1, 5)
        sub(rf1, rf1, 1)
        # rf0, rf1 = (tidx >> 1, 0b11111)
    else:
        assert thread in [1, 12, 24]

    band(rf3, rf0, rf1, sig=ldunifrf(rf7))  # rf3 = thread id
    mov(rf15, rf3)
    shl(rf0, rf7, 2)
    umul24(rf0, rf0, rf3)
    eidx(rf1).add(rf0, rf0, rf6)
    shl(rf1, rf1, 2)
    mov(rf3, 4)
    shl(rf3, rf3, 4).add(rf0, rf0, rf1)
    n = len(regs)
    mov(tmua, rf0, sig=thrsw).add(rf0, rf0, rf3)
    nop()
    nop()
    nop(sig=ldtmu(rf1))  # load unif_params[0:16] to rf1
    for i in range(n):
        if i % 16 == 0:
            bcastf(regs[i], rf1)
            rotate(rf1, rf1, 1)
        elif i % 16 == 15 and i != n - 1:
            mov(tmua, rf0, sig=thrsw).add(rf0, rf0, rf3)
            bcastf(regs[i], rf1)
            nop()
            nop(sig=ldtmu(rf1))
        else:
            bcastf(regs[i], rf1)
            rotate(rf1, rf1, 1)


@qpu
def qpu_sgemm_rnn_naive(asm: Assembly, thread: int) -> None:
    # rf0 - rf15: regs
    # rf16 - rf26: params (from uniform)
    # rf32 - rf37: values
    # rf48 - rf63: 16x16 accumulators

    # params
    reg_p = rf16
    reg_q = rf17
    reg_r = rf18
    reg_a_base = rf19
    reg_a_stride = rf20
    reg_b_base = rf21
    reg_b_stride = rf22
    reg_c_base = rf23
    reg_c_stride = rf24
    reg_alpha = rf25
    reg_beta = rf26

    # values
    reg_a_cur = rf32
    reg_b_cur = rf33
    reg_c_cur = rf34
    reg_i = rf35
    reg_j = rf36
    reg_k = rf37

    # accumulators
    reg_accum = [rf[48 + i] for i in range(16)]

    load_params(
        asm,
        thread,
        [
            reg_p,
            reg_q,
            reg_r,
            reg_a_base,
            reg_a_stride,
            reg_b_base,
            reg_b_stride,
            reg_c_base,
            reg_c_stride,
            reg_alpha,
            reg_beta,
        ],
    )

    add(rf1, reg_p, 15)
    shr(rf1, rf1, 4)
    shl(rf1, rf1, 4)
    add(rf2, reg_r, 15)
    shr(rf2, rf2, 4)
    shl(rf2, rf2, 6)
    umul24(rf4, rf1, reg_a_stride)
    add(reg_a_base, reg_a_base, rf4)
    add(reg_b_base, reg_b_base, rf2)
    umul24(rf4, rf1, reg_c_stride)
    add(reg_c_base, reg_c_base, rf4)
    add(reg_c_base, reg_c_base, rf2)

    for i in range(16):
        mov(reg_accum[i], 0.0)

    # i=(p+15)/16.
    add(rf1, reg_p, 15)
    shr(reg_i, rf1, 4)
    with loop as li:
        # j=(r+15)/16
        add(rf1, reg_r, 15)
        shr(reg_j, rf1, 4)
        with loop as lj:
            shl(rf1, reg_i, 4)
            umul24(rf4, rf1, reg_c_stride)
            shl(rf2, reg_j, 6)
            sub(reg_c_cur, reg_c_base, rf4)
            sub(reg_c_cur, reg_c_cur, rf2)
            umul24(rf4, rf1, reg_a_stride)
            sub(reg_a_cur, reg_a_base, rf4)
            sub(reg_b_cur, reg_b_base, rf2)

            mov(reg_k, reg_q)
            with loop as lk:
                eidx(rf1)
                umul24(rf2, rf1, reg_a_stride)
                add(rf2, rf2, reg_a_cur).add(reg_a_cur, reg_a_cur, 4)
                mov(tmua, rf2, sig=thrsw)
                shl(rf2, rf1, 2)
                add(rf2, rf2, reg_b_cur).add(reg_b_cur, reg_b_cur, reg_b_stride)
                mov(tmua, rf2, sig=thrsw)

                nop(sig=ldtmu(rf1))
                rotate(rf1, rf1, 1).mov(rep, rf1)
                nop(sig=ldtmu(rf5))
                nop().fmul(rf4, rf0, rf5)
                for i in range(1, 16):
                    rotate(rf1, rf1, 1).mov(rep, rf1)
                    fadd(reg_accum[i - 1], reg_accum[i - 1], rf4).fmul(rf4, rf0, rf5)
                fadd(reg_accum[15], reg_accum[15], rf4)

                sub(reg_k, reg_k, 1, cond="pushz")
                lk.b(cond="anyna")
                nop()  # delay slot
                nop()  # delay slot
                nop()  # delay slot

            eidx(rf1)
            shl(rf1, rf1, 2)
            add(rf2, reg_c_cur, rf1)
            mov(tmua, rf2, sig=thrsw).add(rf2, rf2, reg_c_stride)
            fmul(reg_accum[0], reg_accum[0], reg_alpha)
            for i in range(1, 16):
                mov(tmua, rf2, sig=thrsw).add(rf2, rf2, reg_c_stride)
                fmul(reg_accum[i], reg_accum[i], reg_alpha, sig=ldtmu(rf1))
                fmul(rf1, rf1, reg_beta)
                fadd(reg_accum[i - 1], reg_accum[i - 1], rf1)
            nop(sig=ldtmu(rf1))
            fmul(rf1, rf1, reg_beta)
            fadd(reg_accum[15], reg_accum[15], rf1)

            eidx(rf1)
            shl(rf1, rf1, 2)
            add(rf2, reg_c_cur, rf1)
            for i in range(16):
                mov(tmud, reg_accum[i])
                mov(tmua, rf2).add(rf2, rf2, reg_c_stride)
                mov(reg_accum[i], 0.0)
                tmuwt()

            sub(reg_j, reg_j, 1, cond="pushz")
            lj.b(cond="anyna")
            nop()  # delay slot
            nop()  # delay slot
            nop()  # delay slot

        sub(reg_i, reg_i, 1, cond="pushz")
        li.b(cond="anyna")
        nop()
        nop()
        nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def sgemm_rnn_naive() -> None:
    thread = 12

    p = 256 * 3
    q = 1024
    r = 256 * 4

    assert p % (16 * 3) == 0
    assert r % (16 * 4) == 0

    with Driver() as drv:
        code = drv.program(lambda asm: qpu_sgemm_rnn_naive(asm, thread))

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

        def block_3x4_params(i: int, j: int) -> list[int]:
            tile_p = p // 3
            tile_r = r // 4
            return [
                tile_p,
                q,
                tile_r,
                a.addresses()[tile_p * i, 0],
                a.strides[0],
                b.addresses()[0, tile_r * j],
                b.strides[0],
                c.addresses()[tile_p * i, tile_r * j],
                c.strides[0],
                np.float32(alpha).view(np.uint32).item(),
                np.float32(beta).view(np.uint32).item(),
            ]

        unif_params: Array[np.uint32] = drv.alloc((thread, len(block_3x4_params(0, 0))), dtype=np.uint32)
        for th in range(thread):
            unif_params[th] = block_3x4_params(th // 4, th % 4)

        unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)
        unif[0] = unif_params.addresses()[0, 0]
        unif[1] = unif_params.shape[1]

        start = getsec()
        drv.execute(code, unif.addresses()[0], thread=thread)
        time_gpu = getsec() - start

        # np.set_printoptions(threshold=100000)
        # print(c - expected)

        def gflops(sec: float) -> float:
            return (2 * p * q * r + 3 * p * r) / sec * 1e-9

        print(f"==== sgemm example ({p}x{q} times {q}x{r}) ====")
        print(f"numpy: {time_ref:.4} sec, {gflops(time_ref):.4} Gflop/s")
        print(f"QPU:   {time_gpu:.4} sec, {gflops(time_gpu):.4} Gflop/s")
        print(f"Minimum absolute error: {np.min(np.abs(c - expected))}")
        print(f"Maximum absolute error: {np.max(np.abs(c - expected))}")
        print(f"Minimum relative error: {np.min(np.abs((c - expected) / expected))}")
        print(f"Maximum relative error: {np.max(np.abs((c - expected) / expected))}")


def main() -> None:
    sgemm_rnn_naive()


if __name__ == "__main__":
    main()
