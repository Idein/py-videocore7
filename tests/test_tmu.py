# Copyright (c) 2019-2020 Idein Inc.
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
import hypothesis
import numpy as np

from videocore7.assembler import *
from videocore7.assembler import Assembly, TMULookUpConfig, qpu
from videocore7.driver import Array, Driver


def test_tmu_lookup_config() -> None:
    assert int(TMULookUpConfig()) == 0xFF
    assert TMULookUpConfig.sequential_read_write_vec(1) == 0xFFFFFFFF
    assert TMULookUpConfig.sequential_read_write_vec(2) == 0xFFFFFFFA
    assert TMULookUpConfig.sequential_read_write_vec(3) == 0xFFFFFAFB
    assert TMULookUpConfig.sequential_read_write_vec(4) == 0xFFFAFBFC


@qpu
def qpu_tmu_single_write(asm: Assembly) -> None:
    nop(sig=ldunifrf(rf11))
    nop(sig=ldunifrf(rf12))

    # rf12 = addr + eidx * 4
    # rf0 = eidx
    eidx(rf10)
    shl(rf10, rf10, 2).mov(rf0, rf10)
    add(rf12, rf12, rf10)

    mov(rf10, 4)
    shl(rf10, rf10, 4)

    with loop as l:  # noqa: E741
        # rf0: Data to be written.
        # rf10: Overwritten.
        # rf12: Address to write data to.

        sub(rf11, rf11, 1, cond="pushz").mov(tmud, rf0)
        l.b(cond="anyna")
        # rf0 += 16
        sub(rf0, rf0, -16).mov(tmua, rf12)
        nop()
        # rf12 += 64
        tmuwt().add(rf12, rf12, rf10)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@hypothesis.given(
    n=hypothesis.strategies.integers(min_value=1, max_value=4096),
)
def test_tmu_single_write(n: int) -> None:
    with Driver(data_area_size=n * 16 * 4 + 2 * 4) as drv:
        code = drv.program(qpu_tmu_single_write)
        data: Array[np.uint32] = drv.alloc((n, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)

        data[:] = 0xDEADBEAF

        unif[0] = n
        unif[1] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(data == np.arange(n * 16).reshape(n, 16))


@qpu
def qpu_tmu_multiple_interleaved_transform_write(asm: Assembly, use_n_vec: int, interleave: int) -> None:
    reg_addr = rf0
    reg_stride = rf1
    reg_tmu_config = rf2

    nop(sig=ldunifrf(reg_addr))
    bxor(rf3, rf3, rf3, sig=ldunifrf(reg_stride))
    if use_n_vec > 1:
        if use_n_vec >= 2:
            bor(rf3, rf3, 5)
        if use_n_vec >= 3:
            shl(rf3, rf3, 8)
            bor(rf3, rf3, 4)
        if use_n_vec >= 4:
            shl(rf3, rf3, 8)
            bor(rf3, rf3, 3)
        bxor(reg_tmu_config, -1, rf3)
        # reg_tmu_config = TMULookUpConfig.sequential_read_write_vec(use_n_vec)
        mov(tmuc, reg_tmu_config)

    mov(rf11, 4)
    shl(rf11, rf11, 2)
    eidx(rf10)
    for _ in range(use_n_vec):
        mov(tmud, rf10)
        add(rf10, rf10, rf11)

    eidx(rf10)
    shl(rf10, rf10, 2)
    umul24(rf10, rf10, interleave + 1)
    umul24(rf10, rf10, reg_stride)
    add(tmua, rf10, reg_addr)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@hypothesis.given(
    use_n_vec=hypothesis.strategies.integers(min_value=1, max_value=4),
    interleave=hypothesis.strategies.integers(min_value=0, max_value=2),
    pad_u=hypothesis.strategies.integers(min_value=0, max_value=15),
    pad_l=hypothesis.strategies.integers(min_value=0, max_value=15),
    pad_r=hypothesis.strategies.integers(min_value=0, max_value=15),
)
def test_tmu_multiple_interleaved_transform_write(  # FIXME: This test make other tests hang.
    use_n_vec: int,
    interleave: int,
    pad_u: int,
    pad_l: int,
    pad_r: int,
) -> None:
    """Write with N-vec transpose and strided."""
    base = np.arange(use_n_vec * 16).reshape(use_n_vec, 16).astype(np.int32)
    # tmud <- [ 0, 1,...,15]
    # tmud <- [16,17,...,31] (if use_n_vec >= 2)
    # tmud <- [32,33,...,47] (if use_n_vec >= 3)
    # tmud <- [48,49,...,63] (if use_n_vec == 4)
    # base (on tmud, case: use_n_vec = 4):
    # [ [  0,  1,  2, ..., 15],
    #   [ 16, 17, 18, ..., 31],
    #   [ 32, 33, 34, ..., 47],
    #   [ 48, 49, 50, ..., 63] ]
    expected = -np.ones((pad_u + 16 + 15 * interleave, pad_l + use_n_vec + pad_r), dtype=np.int32)
    expected[pad_u :: interleave + 1, pad_l : pad_l + use_n_vec] = base.T
    # interleaved (case: interleave = 1):
    # [ [  0, -1,  1, -1,  2, -1, ..., 15],
    #   [ 16, -1, 17, -1, 18, -1, ..., 31],
    #   [ 32, -1, 33, -1, 34, -1, ..., 47],
    #   [ 48, -1, 49, -1, 50, -1, ..., 63] ]
    #
    # expected (case: pad_u = 2, pad_l = 3, pad_r = 4):
    # [ [ -1, -1, -1,      ...     , -1, -1, -1, -1],
    #   [ -1, -1, -1,      ...     , -1, -1, -1, -1],
    #         .                             .
    #         .                             .
    #   [ -1, -1, -1, interleaved.T, -1, -1, -1, -1] ]
    #
    with Driver() as drv:
        code = drv.program(lambda asm: qpu_tmu_multiple_interleaved_transform_write(asm, use_n_vec, interleave))
        data: Array[np.int32] = drv.alloc(expected.shape, dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)

        data[:] = -1

        unif[0] = data.addresses()[pad_u, pad_l]
        unif[1] = expected.shape[1]

        drv.execute(code, unif.addresses()[0])

        print("------------------------------------------------")
        print(use_n_vec, interleave, pad_u, pad_l, pad_r)
        print("[actual]")
        print(data)
        print("[expected]")
        print(expected)

        assert np.all(data == expected)


@qpu
def qpu_tmu_multiple_write_with_uniform_config(asm: Assembly, use_n_vec: int, interleave: int) -> None:
    reg_addr = rf0
    reg_stride = rf1

    nop(sig=ldunifrf(reg_addr))
    nop(sig=ldunifrf(reg_stride))

    mov(rf11, 4)
    shl(rf11, rf11, 2)
    eidx(rf10)
    for _ in range(use_n_vec):
        mov(tmud, rf10)
        add(rf10, rf10, rf11)

    eidx(rf10)
    shl(rf10, rf10, 2)
    umul24(rf10, rf10, interleave + 1)
    umul24(rf10, rf10, reg_stride)
    add(tmuau, rf10, reg_addr)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@hypothesis.given(
    use_n_vec=hypothesis.strategies.integers(min_value=1, max_value=4),
    interleave=hypothesis.strategies.integers(min_value=0, max_value=2),
    pad_u=hypothesis.strategies.integers(min_value=0, max_value=15),
    pad_l=hypothesis.strategies.integers(min_value=0, max_value=15),
    pad_r=hypothesis.strategies.integers(min_value=0, max_value=15),
)
def _test_tmu_multiple_write_with_uniform_config(  # FIXME: This test make other tests hang.
    use_n_vec: int,
    interleave: int,
    pad_u: int,
    pad_l: int,
    pad_r: int,
) -> None:
    """Write with N-vec transpose and strided."""
    base = np.arange(use_n_vec * 16).reshape(use_n_vec, 16).astype(np.int32)
    # tmud <- [ 0, 1,...,15]
    # tmud <- [16,17,...,31] (if use_n_vec >= 2)
    # tmud <- [32,33,...,47] (if use_n_vec >= 3)
    # tmud <- [48,49,...,63] (if use_n_vec == 4)
    # base (on tmud, case: use_n_vec = 4):
    # [ [  0,  1,  2, ..., 15],
    #   [ 16, 17, 18, ..., 31],
    #   [ 32, 33, 34, ..., 47],
    #   [ 48, 49, 50, ..., 63] ]
    expected = -np.ones((pad_u + 16 + 15 * interleave, pad_l + use_n_vec + pad_r), dtype=np.int32)
    expected[pad_u :: interleave + 1, pad_l : pad_l + use_n_vec] = base.T
    # interleaved (case: interleave = 1):
    # [ [  0, -1,  1, -1,  2, -1, ..., 15],
    #   [ 16, -1, 17, -1, 18, -1, ..., 31],
    #   [ 32, -1, 33, -1, 34, -1, ..., 47],
    #   [ 48, -1, 49, -1, 50, -1, ..., 63] ]
    #
    # expected (case: pad_u = 2, pad_l = 3, pad_r = 4):
    # [ [ -1, -1, -1,      ...     , -1, -1, -1, -1],
    #   [ -1, -1, -1,      ...     , -1, -1, -1, -1],
    #         .                             .
    #         .                             .
    #   [ -1, -1, -1, interleaved.T, -1, -1, -1, -1] ]
    #
    with Driver() as drv:
        code = drv.program(lambda asm: qpu_tmu_multiple_write_with_uniform_config(asm, use_n_vec, interleave))
        data: Array[np.int32] = drv.alloc(expected.shape, dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        data[:] = -1

        unif[0] = data.addresses()[pad_u, pad_l]
        unif[1] = expected.shape[1]
        unif[2] = TMULookUpConfig.sequential_read_write_vec(use_n_vec)

        drv.execute(code, unif.addresses()[0])

        assert np.all(data == expected)


@qpu
def qpu_tmu_single_read(asm: Assembly) -> None:
    # rf10: Number of vectors to read.
    # rf11: Pointer to the read vectors + eidx * 4.
    # rf12: Pointer to the write vectors + eidx * 4
    eidx(rf13, sig=ldunifrf(rf10))
    nop(sig=ldunifrf(rf11))
    shl(rf13, rf13, 2)
    add(rf11, rf11, rf13, sig=ldunifrf(rf12))
    add(rf12, rf12, rf13)
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    with loop as l:  # noqa: E741
        mov(tmua, rf11, sig=thrsw)
        nop()
        sub(rf10, rf10, 1, cond="pushz")
        nop(sig=ldtmu(rf0))
        l.b(cond="anyna")
        add(tmud, rf0, 1).add(rf11, rf11, rf13)  # rf11 += 64
        mov(tmua, rf12).add(rf12, rf12, rf13)  # rf12 += 64
        tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@hypothesis.given(
    n=hypothesis.strategies.integers(min_value=1, max_value=4096),
)
def test_tmu_single_read(n: int) -> None:
    with Driver() as drv:
        code = drv.program(qpu_tmu_single_read)
        data: Array[np.uint32] = drv.alloc((n, 16), dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        base = np.arange(n * 16).reshape(n, 16)

        data[:] = base
        unif[0] = n
        unif[1] = data.addresses()[0, 0]
        unif[2] = data.addresses()[0, 0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(data == base + 1)


@qpu
def qpu_tmu_multiple_read_with_uniform_config(asm: Assembly, use_n_vec: int, interleave: int) -> None:
    reg_src_addr = rf0
    reg_dst_addr = rf1
    reg_stride = rf2

    nop(sig=ldunifrf(reg_src_addr))
    nop(sig=ldunifrf(reg_dst_addr))
    nop(sig=ldunifrf(reg_stride))

    eidx(rf10)
    shl(rf10, rf10, 2)
    umul24(rf10, rf10, interleave + 1)
    umul24(rf10, rf10, reg_stride)
    add(tmuau if use_n_vec > 1 else tmua, reg_src_addr, rf10, sig=thrsw)
    nop()
    nop()
    for i in range(use_n_vec):
        nop(sig=ldtmu(rf[10 + i]))

    eidx(rf14)
    shl(rf14, rf14, 2)
    add(reg_dst_addr, reg_dst_addr, rf14)
    mov(rf14, 4)
    shl(rf14, rf14, 4)

    for i in range(use_n_vec):
        mov(tmud, rf[10 + i])
        mov(tmua, reg_dst_addr)
        add(reg_dst_addr, reg_dst_addr, rf14)
        tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@hypothesis.given(
    use_n_vec=hypothesis.strategies.integers(min_value=1, max_value=4),
    interleave=hypothesis.strategies.integers(min_value=0, max_value=2),
    pad_u=hypothesis.strategies.integers(min_value=0, max_value=15),
    pad_l=hypothesis.strategies.integers(min_value=0, max_value=15),
    pad_r=hypothesis.strategies.integers(min_value=0, max_value=15),
)
def _test_tmu_multiple_read_with_uniform_config(  # FIXME: This test make other tests hang.
    use_n_vec: int,
    interleave: int,
    pad_u: int,
    pad_l: int,
    pad_r: int,
) -> None:
    expected = np.arange(use_n_vec * 16).reshape(use_n_vec, 16).astype(np.int32)
    # expected (ase: use_n_vec = 4):
    # [ [  0,  1,  2, ..., 15],
    #   [ 16, 17, 18, ..., 31],
    #   [ 32, 33, 34, ..., 47],
    #   [ 48, 49, 50, ..., 63] ]
    """Read with N-vec transpose and strided."""
    source = -np.ones((pad_u + 16 + 15 * interleave, pad_l + use_n_vec + pad_r), dtype=np.int32)
    source[pad_u :: interleave + 1, pad_l : pad_l + use_n_vec] = expected.T
    # interleaved (case: interleave = 1):
    # [ [  0, -1,  1, -1,  2, -1, ..., 15],
    #   [ 16, -1, 17, -1, 18, -1, ..., 31],
    #   [ 32, -1, 33, -1, 34, -1, ..., 47],
    #   [ 48, -1, 49, -1, 50, -1, ..., 63] ]
    #
    # expected (case: pad_u = 2, pad_l = 3, pad_r = 4):
    # [ [ -1, -1, -1,      ...     , -1, -1, -1, -1],
    #   [ -1, -1, -1,      ...     , -1, -1, -1, -1],
    #         .                             .
    #         .                             .
    #   [ -1, -1, -1, interleaved.T, -1, -1, -1, -1] ]
    #
    with Driver() as drv:
        code = drv.program(lambda asm: qpu_tmu_multiple_read_with_uniform_config(asm, use_n_vec, interleave))
        src: Array[np.int32] = drv.alloc(source.shape, dtype=np.int32)
        dst: Array[np.int32] = drv.alloc((use_n_vec, 16), dtype=np.int32)
        unif: Array[np.uint32] = drv.alloc(4, dtype=np.uint32)

        src[:] = source
        dst[:] = -1

        unif[0] = src.addresses()[pad_u, pad_l]
        unif[1] = dst.addresses()[0, 0]
        unif[2] = source.shape[1]
        unif[3] = TMULookUpConfig.sequential_read_write_vec(use_n_vec)

        drv.execute(code, unif.addresses()[0])

        assert np.all(dst == expected)


# VC4 TMU cache & DMA break memory consistency.
# VC6 TMU cache keeps memory consistency.
# How about VC7 TMU ?
@qpu
def qpu_tmu_keeps_memory_consistency(asm: Assembly) -> None:
    nop(sig=ldunifrf(rf10))

    mov(tmua, rf10, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf11))

    add(tmud, rf11, 1)
    mov(tmua, rf10)
    tmuwt()

    mov(tmua, rf10, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf11))

    add(tmud, rf11, 1)
    mov(tmua, rf10)
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_keeps_memory_consistency() -> None:
    with Driver() as drv:
        code = drv.program(qpu_tmu_keeps_memory_consistency)
        data: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        data[:] = 1
        unif[0] = data.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(data[0] == 3)
        assert np.all(data[1:] == 1)


@qpu
def qpu_tmu_read_tmu_write_uniform_read(asm: Assembly) -> None:
    eidx(rf10, sig=ldunifrf(rf0))
    shl(rf10, rf10, 2)
    add(rf0, rf0, rf10, sig=ldunifrf(rf1))
    add(rf1, rf1, rf10)

    mov(tmua, rf0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf10))  # rf10 = [1,...,1]

    add(tmud, rf10, 1)
    mov(tmua, rf0)  # data = [2,...,2]
    tmuwt()

    b(R.set_unif_addr, cond="always").unif_addr(rf0)  # unif_addr = data.addresses()[0]
    nop()
    nop()
    nop()
    L.set_unif_addr

    nop(sig=ldunifrf(rf10))  # rf10 = [data[0],...,data[0]] = [2,...,2]

    add(tmud, rf10, 1)
    mov(tmua, rf1)  # result = [3,...,3]
    tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def test_tmu_read_tmu_write_uniform_read() -> None:
    with Driver() as drv:
        code = drv.program(qpu_tmu_read_tmu_write_uniform_read)
        data: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        result: Array[np.uint32] = drv.alloc(16, dtype=np.uint32)
        unif: Array[np.uint32] = drv.alloc(3, dtype=np.uint32)

        data[:] = 1
        unif[0] = data.addresses()[0]
        unif[1] = result.addresses()[0]

        drv.execute(code, unif.addresses()[0])

        assert np.all(data == 2)
        assert np.all(result == 2)  # !? not 3 ?
