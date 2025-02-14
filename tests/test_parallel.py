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
import time
from typing import Any, Literal

import hypothesis
import hypothesis.stateful
import hypothesis.strategies
import numpy as np

from videocore7.assembler import *
from videocore7.assembler import qpu
from videocore7.driver import Array, Driver


@qpu
def cost(asm: Assembly) -> None:
    mov(rf10, 8)
    shl(rf10, rf10, 8)
    shl(rf10, rf10, 8)
    with loop as l:  # noqa: E741
        sub(rf10, rf10, 1, cond="pushn")
        l.b(cond="anyna")
        nop()
        nop()
        nop()


@qpu
def qpu_serial(asm: Assembly, thread: int) -> None:
    nop(sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    nop(sig=ldunifrf(rf3))

    eidx(rf10)
    shl(rf10, rf10, 2)
    add(rf2, rf2, rf10)
    add(rf3, rf3, rf10)
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    for i in range(thread):
        mov(tmua, rf2, sig=thrsw).add(rf2, rf2, rf13)
        nop()
        nop()
        nop(sig=ldtmu(rf10))
        mov(tmud, rf10)
        mov(tmua, rf3, sig=thrsw).add(rf3, rf3, rf13)
        tmuwt()

    cost(asm)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


# This code requires 12 or 24 thread execution.
# If # of thread == 12, thread id = (tidx & 0b111100) >> 2.
# If # of thread == 24, thread id = (tidx & 0b111110) >> 1.
@qpu
def qpu_parallel(asm: Assembly, thread: Literal[12, 24]) -> None:
    tidx(rf10, sig=ldunifrf(rf0))
    shr(rf10, rf10, 3 - thread // 12)
    mov(rf11, 1)
    shl(rf11, rf11, 3 + thread // 12)
    sub(rf11, rf11, 1)
    band(rf31, rf10, rf11)

    # rf31 = thread_id = (tidx & 0b111100) >> 2

    nop(sig=ldunifrf(rf1))  # rf1 = unif[0,1]
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf31)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)

    # rf10 = thread_id * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))

    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 =  unif[th,2:18]
    bcastf(rf2, rf10)  # rf2 = unif[th,2] (x address)
    rotate(rf10, rf10, 1)
    bcastf(rf3, rf10)  # rf3 = unif[th,3] (y address)

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf2, rf12, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf32))

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf32)
    add(tmua, rf3, rf12)
    tmuwt()

    cost(asm)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_parallel() -> None:
    n_of_threads: list[Literal[12, 24]] = [12, 24]
    for thread in n_of_threads:
        with Driver() as drv:
            serial_code = drv.program(qpu_serial, thread)
            parallel_code = drv.program(qpu_parallel, thread)
            x: Array[np.uint32] = drv.alloc((thread, 16), dtype=np.uint32)
            ys: Array[np.uint32] = drv.alloc((thread, 16), dtype=np.uint32)
            yp: Array[np.uint32] = drv.alloc((thread, 16), dtype=np.uint32)
            unif: Array[np.uint32] = drv.alloc((thread, 4), dtype=np.uint32)

            x[:] = np.arange(thread * 16).reshape(x.shape)

            unif[:, 0] = unif.addresses()[:, 0]
            unif[:, 1] = unif.shape[1]
            unif[:, 2] = x.addresses()[:, 0]

            ys[:] = 0
            unif[:, 3] = ys.addresses()[:, 0]

            start = time.time()
            drv.execute(serial_code, unif.addresses()[0, 0])
            end = time.time()
            serial_cost = end - start

            yp[:] = 0
            unif[:, 3] = yp.addresses()[:, 0]

            start = time.time()
            drv.execute(parallel_code, unif.addresses()[0, 0], wgs_per_sg=thread, thread=thread)
            end = time.time()
            parallel_cost = end - start

            assert np.all(x == ys)
            assert np.all(x == yp)
            assert parallel_cost < serial_cost * 2, f"{parallel_cost=}, {serial_cost=}"


# If remove `barrierid` in this code, `test_barrier` will fail.
@qpu
def qpu_barrier(asm: Assembly) -> None:
    tidx(rf10, sig=ldunifrf(rf0))  # rf0 = unif[0,0]
    shr(rf12, rf10, 2)
    band(rf11, rf10, 0b11)  # thread_id
    band(rf12, rf12, 0b1111)  # qpu_id
    shr(rf11, rf11, 1)
    shl(rf12, rf12, 1)
    add(rf31, rf11, rf12)  # rf31 = (qpu_id * 2) + (thread_id >> 1)

    nop(sig=ldunifrf(rf1))  # rf1 = unif[0,1]

    # rf31 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf31)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 =  unif[th,2:18]
    bcastf(rf2, rf10)  # rf2 = unif[th,2] (x address)
    rotate(rf10, rf10, 1)
    bcastf(rf3, rf10)  # rf3 = unif[th,3] (y address)

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf2, rf12, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))

    mov(rf11, rf31)
    shl(rf11, rf11, 8)
    L.loop
    sub(rf11, rf11, 1, cond="pushn")
    b(R.loop, cond="anyna")
    nop()
    nop()
    nop()

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf10)
    add(tmua, rf3, rf12)
    tmuwt()

    barrierid(syncb, sig=thrsw)

    add(rf32, rf31, 1)
    mov(rf31, 8)
    add(rf31, rf31, 8)
    add(rf31, rf31, 8)
    sub(null, rf32, rf31, cond="pushz")
    b(R.skip, cond="allna")
    nop()
    nop()
    nop()
    mov(rf32, 0)
    L.skip

    # rf32 = (rf31 + 1) mod 24

    # rf32 * unif[0,1] * sizeof(float) + (unif.addresses[0,0] + 2 * sizeof(float))
    shl(rf10, rf1, 2)
    umul24(rf10, rf10, rf32)
    add(rf11, rf0, 8)
    add(rf10, rf10, rf11)
    eidx(rf11)
    shl(rf11, rf11, 2)
    add(tmua, rf10, rf11, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 = unif[(th+1)%16,2:18]
    bcastf(rf4, rf10)  # rf4 = unif[(th+1)%16,2]
    rotate(rf10, rf10, 1)
    bcastf(rf5, rf10)  # rf5 = unif[(th+1)%16,3]

    eidx(rf12)
    shl(rf12, rf12, 2)
    add(tmua, rf5, rf12, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))

    eidx(rf12)
    shl(rf12, rf12, 2)
    mov(tmud, rf10)
    add(tmua, rf3, rf12)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_barrier() -> None:
    with Driver() as drv:
        thread = 24

        code = drv.program(qpu_barrier)
        x: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        y: Array[np.float32] = drv.alloc((thread, 16), dtype=np.float32)
        unif: Array[np.uint32] = drv.alloc((thread, 4), dtype=np.uint32)

        x[:] = np.random.randn(*x.shape)
        y[:] = -1

        unif[:, 0] = unif.addresses()[:, 0]
        unif[:, 1] = unif.shape[1]
        unif[:, 2] = x.addresses()[:, 0]
        unif[:, 3] = y.addresses()[:, 0]

        drv.execute(code, unif.addresses()[0, 0], wgs_per_sg=thread, thread=thread)

        assert np.all(y == np.concatenate([x[1:], x[:1]]))


@qpu
def qpu_parallel_full(asm: Assembly) -> None:
    tidx(rf1, sig=ldunifrf(rf2))
    shl(rf1, rf1, 6)
    add(rf2, rf2, rf1, sig=thrsw)
    eidx(rf1)
    shl(rf1, rf1, 2)
    add(rf2, rf2, rf1)

    tidx(tmud)
    mov(tmua, rf2)
    tmuwt()

    barrierid(syncb, sig=thrsw)
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


def test_parallel_full() -> None:
    """48 Threads."""
    with Driver() as drv:
        code = drv.program(qpu_parallel_full)
        dst: Array[np.uint32] = drv.alloc((48, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(1, dtype=np.uint32)

        dst[:] = 0xDEADBEEF

        unif[0] = dst.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0], workgroup=(4, 4, 3), thread=48, threading=True)

        assert np.all(dst == np.arange(48, dtype=np.uint32).reshape(48, 1))


@qpu
def qpu_parallel_with_workgroup(asm: Assembly, workgroup: tuple[int, int, int]) -> None:
    wg_x, wg_y, wg_z = workgroup
    wg_size = wg_x * wg_y * wg_z

    def next_pow2(x: int) -> int:
        return 1 << (x - 1).bit_length()

    def ffs(x: int) -> int:
        return (x & -x).bit_length()

    local_invocation_index_bits = ffs(next_pow2(max(wg_size, 64))) - 1

    reg_wg_x_id = rf20
    reg_wg_y_id = rf21
    reg_wg_z_id = rf22
    reg_wg_id = rf23
    reg_local_invocation_id = rf24
    reg_global_invocation_id = rf25
    reg_local_x_id = rf27
    reg_local_y_id = rf28
    reg_local_z_id = rf29

    # pick threaad info
    mov(reg_wg_x_id, rf3.unpack("ul"))
    mov(reg_wg_y_id, rf3.unpack("uh"))
    mov(reg_wg_z_id, rf2.unpack("ul"))
    mov(rf2, rf2.unpack("uh"))
    shr(reg_local_invocation_id, rf2, 16 - local_invocation_index_bits)

    # each axis local (= in workgroup) id
    mov(rf1, wg_x)
    umul24(rf1, rf1, wg_y)
    mov(rf2, reg_local_invocation_id, cond="pushn").mov(reg_local_z_id, -1)
    with loop as mod_xy:
        mod_xy.b(cond="anyna")
        add(reg_local_z_id, reg_local_z_id, 1, cond="ifna")
        sub(rf2, rf2, rf1, cond="ifna")
        mov(null, rf2, cond="pushn")
    add(rf2, rf2, rf1, cond="pushn").mov(rf1, wg_x)
    mov(reg_local_y_id, -1)
    with loop as mod_x:
        mod_x.b(cond="anyna")
        add(reg_local_y_id, reg_local_y_id, 1, cond="ifna")
        sub(rf2, rf2, rf1, cond="ifna")
        mov(null, rf2, cond="pushn")
    add(reg_local_x_id, rf2, rf1)

    # wg_id = wg_x_id + wg_x * (wg_y_id + wg_y * wg_z_id)
    mov(rf1, reg_wg_z_id)
    umul24(rf1, rf1, wg_y)
    add(rf1, rf1, reg_wg_y_id)
    umul24(rf1, rf1, wg_x)
    add(reg_wg_id, rf1, reg_wg_x_id)

    # global_invocation_id = wg_id * wg_size + local_invocation_id
    umul24(rf1, reg_wg_id, wg_x)
    umul24(rf1, rf1, wg_y)
    umul24(rf1, rf1, wg_z)
    add(reg_global_invocation_id, rf1, reg_local_invocation_id)

    shl(rf12, reg_global_invocation_id, 2)

    mov(tmud, reg_global_invocation_id, sig=ldunifrf(rf11))
    add(tmua, rf11, rf12)

    mov(tmud, reg_local_x_id, sig=ldunifrf(rf11))
    add(tmua, rf11, rf12)

    mov(tmud, reg_local_y_id, sig=ldunifrf(rf11))
    add(tmua, rf11, rf12)

    mov(tmud, reg_local_z_id, sig=ldunifrf(rf11))
    add(tmua, rf11, rf12)

    # mov(tmud, reg_wg_x_id, sig=ldunifrf(rf11))
    # add(tmua, rf11, rf12)

    # mov(tmud, reg_wg_y_id, sig=ldunifrf(rf11))
    # add(tmua, rf11, rf12)

    # mov(tmud, reg_wg_z_id, sig=ldunifrf(rf11))
    # add(tmua, rf11, rf12)

    tmuwt()

    barrierid(syncb, sig=thrsw)
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


def case_parallel_with_workgroup(n: int, workgroup: tuple[int, int, int], threading: bool) -> None:
    wg_x, wg_y, wg_z = workgroup
    wg_size = wg_x * wg_y * wg_z
    thread = n * wg_size // 16
    with Driver() as drv:
        code = drv.program(qpu_parallel_with_workgroup, workgroup)
        dst_wg_id: Array[np.uint32] = drv.alloc((n, *reversed(workgroup)), dtype=np.uint32)
        dst_local_x_id: Array[np.uint32] = drv.alloc((n, *reversed(workgroup)), dtype=np.uint32)
        dst_local_y_id: Array[np.uint32] = drv.alloc((n, *reversed(workgroup)), dtype=np.uint32)
        dst_local_z_id: Array[np.uint32] = drv.alloc((n, *reversed(workgroup)), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(4, dtype=np.uint32)

        dst_wg_id[:] = 0xDEADBEEF
        dst_local_x_id[:] = 0xDEADBEEF
        dst_local_y_id[:] = 0xDEADBEEF
        dst_local_z_id[:] = 0xDEADBEEF

        unif[0] = dst_wg_id.addresses().item(0)
        unif[1] = dst_local_x_id.addresses().item(0)
        unif[2] = dst_local_y_id.addresses().item(0)
        unif[3] = dst_local_z_id.addresses().item(0)

        slice_count = 3
        qpu_per_slice = 4
        qpu_count = slice_count * qpu_per_slice
        max_thread_per_qpu = 4 if threading else 2
        max_elements_per_sg = qpu_count * max_thread_per_qpu // 2 * 16
        max_wgs_per_sg = max_elements_per_sg // wg_size

        drv.execute(
            code,
            unif.addresses()[0],
            workgroup=workgroup,
            thread=thread,
            wgs_per_sg=max_wgs_per_sg,
            threading=threading,
        )

        # np.set_printoptions(threshold=10000000)
        # print(f"workgroup: {workgroup}")
        # print(f"thread: {thread}")
        # print("wg_id")
        # print(dst_wg_id)
        # print("local_x_id")
        # print(dst_local_x_id)
        # print("local_y_id")
        # print(dst_local_y_id)
        # print("local_z_id")
        # print(dst_local_z_id)

        assert np.all(dst_wg_id == np.arange(n * wg_size).reshape(n, *reversed(workgroup)))
        assert np.all(dst_local_x_id[:] == np.arange(wg_x).reshape(1, 1, wg_x))
        assert np.all(dst_local_y_id[:] == np.arange(wg_y).reshape(1, wg_y, 1))
        assert np.all(dst_local_z_id[:] == np.arange(wg_z).reshape(wg_z, 1, 1))


@hypothesis.strategies.composite
def hypothesis_strategies_workgroup(draw: Any) -> tuple[tuple[int, int, int], bool]:
    threading = draw(hypothesis.strategies.booleans())
    limit = 255 if threading else 192

    # wg_size = wg_x * wg_y * wg_z must be multiple of 16
    # wg_x < 16, wg_y < 16, wg_z < 16

    [a, b] = draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.sampled_from(range(4)),
            min_size=2,
            max_size=2,
            unique=True,
        )
    )

    pow = [min(a, b), abs(b - a), 4 - max(a, b)]
    rng = draw(hypothesis.strategies.randoms())
    rng.shuffle(pow)
    a, b, c = pow

    pow_of_two_x = 2**a
    pow_of_two_y = 2**b
    pow_of_two_z = 2**c

    lim = limit // 16

    x = draw(hypothesis.strategies.integers(min_value=1, max_value=min(15 // pow_of_two_x, lim)))
    lim //= x
    y = draw(hypothesis.strategies.integers(min_value=1, max_value=min(15 // pow_of_two_y, lim)))
    lim //= y
    z = draw(hypothesis.strategies.integers(min_value=1, max_value=min(15 // pow_of_two_z, lim)))

    wg = [x * pow_of_two_x, y * pow_of_two_y, z * pow_of_two_z]
    rng = draw(hypothesis.strategies.randoms())
    rng.shuffle(wg)
    wg_x, wg_y, wg_z = wg

    if (wg_x * wg_y * wg_z) > limit:
        print(f"wg_x: {wg_x}, wg_y: {wg_y}, wg_z: {wg_z}, limit: {limit}")
    hypothesis.assume((wg_x * wg_y * wg_z) <= limit)

    return (wg_x, wg_y, wg_z), threading


@hypothesis.given(
    n=hypothesis.strategies.integers(min_value=1, max_value=32),
    wg_and_threading=hypothesis_strategies_workgroup(),
)
def test_parallel_with_workgroup(n: int, wg_and_threading: tuple[tuple[int, int, int], bool]) -> None:
    case_parallel_with_workgroup(n, *wg_and_threading)
