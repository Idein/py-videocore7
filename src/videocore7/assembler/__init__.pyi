from typing import overload

from _videocore7.assembler import ALUWithoutSMIMM as ALUWithoutSMIMM
from _videocore7.assembler import ALUWithSMIMM as ALUWithSMIMM
from _videocore7.assembler import Assembly as Assembly
from _videocore7.assembler import Branch as Branch
from _videocore7.assembler import Label as Label
from _videocore7.assembler import LabelNameSpace as LabelNameSpace
from _videocore7.assembler import Link as Link
from _videocore7.assembler import LoadSignal as LoadSignal
from _videocore7.assembler import LoopHelper
from _videocore7.assembler import Reference as Reference
from _videocore7.assembler import ReferenceHelper as ReferenceHelper
from _videocore7.assembler import Register as Register
from _videocore7.assembler import Signal as Signal
from _videocore7.assembler import TMULookUpConfig as TMULookUpConfig
from _videocore7.assembler import assemble as assemble
from _videocore7.assembler import qpu as qpu

# Structured programming helpers
loop: LoopHelper
L: Label
R: ReferenceHelper

def b(
    src: int | Register | Reference | Link | None,
    *,
    cond: str,
    absolute: bool = False,
    set_link: bool = False,
) -> Branch: ...

link: Link

def namespace(name: str) -> LabelNameSpace: ...

# Signals
thrsw: Signal
ldunif: Signal
ldunifa: Signal
ldunifrf: LoadSignal
ldunifarf: LoadSignal
ldtmu: LoadSignal
ldvary: LoadSignal
ldvpm: Signal
ldtlb: LoadSignal
ldtlbu: LoadSignal
ucb: Signal
wrtmuc: Signal

# Registers
quad: Register
null: Register
tlb: Register
tlbu: Register
unifa: Register
tmul: Register
tmud: Register
tmua: Register
tmuau: Register
vpm: Register
vpmu: Register
sync: Register
syncu: Register
syncb: Register
tmuc: Register
tmus: Register
tmut: Register
tmur: Register
tmui: Register
tmub: Register
tmudref: Register
tmuoff: Register
tmuscm: Register
tmusf: Register
tmuslod: Register
tmuhs: Register
tmuhscm: Register
tmuhsf: Register
tmuhslod: Register
rep: Register

# Register Alias
broadcast: Register  # rep
quad_broadcast: Register  # quad

rf: list[Register]
rf0: Register
rf1: Register
rf2: Register
rf3: Register
rf4: Register
rf5: Register
rf6: Register
rf7: Register
rf8: Register
rf9: Register
rf10: Register
rf11: Register
rf12: Register
rf13: Register
rf14: Register
rf15: Register
rf16: Register
rf17: Register
rf18: Register
rf19: Register
rf20: Register
rf21: Register
rf22: Register
rf23: Register
rf24: Register
rf25: Register
rf26: Register
rf27: Register
rf28: Register
rf29: Register
rf30: Register
rf31: Register
rf32: Register
rf33: Register
rf34: Register
rf35: Register
rf36: Register
rf37: Register
rf38: Register
rf39: Register
rf40: Register
rf41: Register
rf42: Register
rf43: Register
rf44: Register
rf45: Register
rf46: Register
rf47: Register
rf48: Register
rf49: Register
rf50: Register
rf51: Register
rf52: Register
rf53: Register
rf54: Register
rf55: Register
rf56: Register
rf57: Register
rf58: Register
rf59: Register
rf60: Register
rf61: Register
rf62: Register
rf63: Register

# Add ALU instructions
@overload
def fadd(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fadd(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fadd(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def faddnf(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def faddnf(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def faddnf(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vfpack(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vfpack(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vfpack(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def add(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def add(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def add(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def sub(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def sub(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def sub(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fsub(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fsub(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fsub(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def imin(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def imin(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def imin(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def imax(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def imax(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def imax(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def umin(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def umin(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def umin(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def umax(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def umax(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def umax(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def shl(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def shl(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def shl(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def shr(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def shr(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def shr(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def asr(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def asr(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def asr(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def ror(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def ror(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def ror(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fmin(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fmin(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fmin(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fmax(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fmax(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fmax(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...

#
@overload
def vfmin(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vfmin(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vfmin(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def band(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def band(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def band(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def bor(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def bor(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def bor(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def bxor(
    dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def bxor(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def bxor(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vadd(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vadd(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vadd(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vsub(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vsub(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vsub(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def bnot(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def bnot(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def neg(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def neg(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def flapush(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def flapush(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def flbpush(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def flbpush(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def flpop(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def flpop(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def clz(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def clz(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def setmsf(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def setmsf(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def setrevf(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def setrevf(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def nop(cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def tidx(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def eidx(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def lr(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def vfla(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def vflna(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def vflb(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def vflnb(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def xcd(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def ycd(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def msf(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def revf(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def iid(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def sampid(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def barrierid(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def tmuwt(dst: Register = null, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def vpmwt(dst: Register = null, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def flafirst(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def flnafirst(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def fxcd(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
def fycd(dst: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmv_in(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ldvpmv_in(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmd_in(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ldvpmd_in(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmp(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ldvpmp(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def recip(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def recip(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def rsqrt(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def rsqrt(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def exp(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def exp(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def log(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def log(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def sin(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def sin(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def rsqrt2(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def rsqrt2(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ballot(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ballot(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def bcastf(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def bcastf(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def alleq(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def alleq(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def allfeq(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def allfeq(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmg_in(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def ldvpmg_in(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def ldvpmg_in(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def stvpmv(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def stvpmv(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def stvpmd(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def stvpmd(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def stvpmp(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def stvpmp(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def fcmp(
    dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fcmp(
    dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def fcmp(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vfmax(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vfmax(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vfmax(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fround(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def fround(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftoin(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ftoin(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftrunc(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ftrunc(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftoiz(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ftoiz(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ffloor(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ffloor(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftouz(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ftouz(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def fceil(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def fceil(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftoc(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def ftoc(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def fdx(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def fdx(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def fdy(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def fdy(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def itof(dst: Register, src: int, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def itof(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def utof(dst: Register, src: int, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def utof(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def vpack(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vpack(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def vpack(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def v8pack(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def v8pack(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def v8pack(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fmov(dst: Register, src: float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def fmov(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def mov(dst: Register, src: int | float, cond: str | None = None, sig: Signal | None = None) -> ALUWithSMIMM: ...
@overload
def mov(dst: Register, src: Register, cond: str | None = None, sig: Signal | None = None) -> ALUWithoutSMIMM: ...

#
@overload
def v10pack(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def v10pack(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def v10pack(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def v11fpack(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def v11fpack(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def v11fpack(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def quad_rotate(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def quad_rotate(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def quad_rotate(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def rotate(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def rotate(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def rotate(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

#
@overload
def shuffle(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def shuffle(
    dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None
) -> ALUWithSMIMM: ...
@overload
def shuffle(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> ALUWithoutSMIMM: ...

# Mul ALU instructions
@overload
def umul24(dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None) -> None: ...
@overload
def umul24(dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None) -> None: ...
@overload
def umul24(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...

#
@overload
def vfmul(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...
@overload
def vfmul(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> None: ...
@overload
def vfmul(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...

#
@overload
def smul24(dst: Register, src1: int, src2: Register, cond: str | None = None, sig: Signal | None = None) -> None: ...
@overload
def smul24(dst: Register, src1: Register, src2: int, cond: str | None = None, sig: Signal | None = None) -> None: ...
@overload
def smul24(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...

#
@overload
def multop(
    dst: Register, src1: int | float, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...
@overload
def multop(
    dst: Register, src1: Register, src2: int | float, cond: str | None = None, sig: Signal | None = None
) -> None: ...
@overload
def multop(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...

#
def ftounorm16(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> None: ...

#
def ftosnorm16(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> None: ...

#
def vftounorm8(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> None: ...

#
def vftosnorm8(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> None: ...

#
def vftounorm10lo(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> None: ...

#
def vftounorm10hi(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> None: ...

#
@overload
def fmul(dst: Register, src1: float, src2: Register, cond: str | None = None, sig: Signal | None = None) -> None: ...
@overload
def fmul(dst: Register, src1: Register, src2: float, cond: str | None = None, sig: Signal | None = None) -> None: ...
@overload
def fmul(
    dst: Register, src1: Register, src2: Register, cond: str | None = None, sig: Signal | None = None
) -> None: ...
