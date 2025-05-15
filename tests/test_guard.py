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
import pytest

from videocore7.assembler import *
from videocore7.assembler import AssembleError, Assembly, qpu
from videocore7.driver import Driver


@qpu
def qpu_rotate_to_broadcast(asm: Assembly) -> None:
    rotate(broadcast, rf0, 0)


@qpu
def qpu_rotate_to_tmuc(asm: Assembly) -> None:
    rotate(tmuc, rf0, 0)


@qpu
def qpu_rotate_to_tmud(asm: Assembly) -> None:
    rotate(tmud, rf0, 0)


@qpu
def qpu_rotate_to_tmua(asm: Assembly) -> None:
    rotate(tmua, rf0, 0)


@qpu
def qpu_rotate_and_mul_to_tmuc(asm: Assembly) -> None:
    rotate(rf0, rf0, 0).mov(tmuc, rf0)


@qpu
def qpu_rotate_and_mul_to_tmud(asm: Assembly) -> None:
    rotate(rf0, rf0, 0).mov(tmud, rf0)


@qpu
def qpu_rotate_and_mul_to_tmua(asm: Assembly) -> None:
    rotate(rf0, rf0, 0).mov(tmua, rf0)


def test_regs_rep() -> None:
    must_guraded_usages = [
        qpu_rotate_to_broadcast,
        qpu_rotate_to_tmuc,
        qpu_rotate_to_tmud,
        qpu_rotate_to_tmua,
        qpu_rotate_and_mul_to_tmuc,
        qpu_rotate_and_mul_to_tmud,
        qpu_rotate_and_mul_to_tmua,
    ]
    for qpu_code in must_guraded_usages:
        with Driver() as drv:
            with pytest.raises(AssembleError):
                drv.program(qpu_code)
