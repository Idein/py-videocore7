from typing import Final, overload

from _videocore7.assembler import ALUConditionArg as ALUConditionArg
from _videocore7.assembler import ALUConditionLiteral as ALUConditionLiteral
from _videocore7.assembler import ALUWithoutSMIMM as ALUWithoutSMIMM
from _videocore7.assembler import ALUWithSMIMM as ALUWithSMIMM
from _videocore7.assembler import AssembleError as AssembleError
from _videocore7.assembler import Assembly as Assembly
from _videocore7.assembler import Branch as Branch
from _videocore7.assembler import BranchConditionLiteral as BranchConditionLiteral
from _videocore7.assembler import Label as Label
from _videocore7.assembler import LabelNameSpace as LabelNameSpace
from _videocore7.assembler import Link as Link
from _videocore7.assembler import LoadSignal as LoadSignal
from _videocore7.assembler import LoopHelper
from _videocore7.assembler import Reference as Reference
from _videocore7.assembler import ReferenceHelper as ReferenceHelper
from _videocore7.assembler import Register as Register
from _videocore7.assembler import Signal as Signal
from _videocore7.assembler import SignalArg as SignalArg
from _videocore7.assembler import TMULookUpConfig as TMULookUpConfig
from _videocore7.assembler import assemble as assemble
from _videocore7.assembler import qpu as qpu

# Structured programming helpers
loop: Final[LoopHelper]
L: Final[Label]
R: Final[ReferenceHelper]

def b(
    src: int | Register | Reference | Link | None,
    *,
    cond: BranchConditionLiteral,
    absolute: bool = False,
    set_link: bool = False,
) -> Branch: ...

link: Final[Link]

def namespace(name: str) -> LabelNameSpace: ...

# Signals
thrsw: Final[Signal]
ldunif: Final[Signal]
ldunifa: Final[Signal]
ldunifrf: Final[LoadSignal]
ldunifarf: Final[LoadSignal]
ldtmu: Final[LoadSignal]
ldvary: Final[LoadSignal]
ldvpm: Final[Signal]
ldtlb: Final[LoadSignal]
ldtlbu: Final[LoadSignal]
ucb: Final[Signal]
wrtmuc: Final[Signal]

# Registers
quad: Final[Register]
null: Final[Register]
tlb: Final[Register]
tlbu: Final[Register]
unifa: Final[Register]
tmul: Final[Register]
tmud: Final[Register]
tmua: Final[Register]
tmuau: Final[Register]
vpm: Final[Register]
vpmu: Final[Register]
sync: Final[Register]
syncu: Final[Register]
syncb: Final[Register]
tmuc: Final[Register]
tmus: Final[Register]
tmut: Final[Register]
tmur: Final[Register]
tmui: Final[Register]
tmub: Final[Register]
tmudref: Final[Register]
tmuoff: Final[Register]
tmuscm: Final[Register]
tmusf: Final[Register]
tmuslod: Final[Register]
tmuhs: Final[Register]
tmuhscm: Final[Register]
tmuhsf: Final[Register]
tmuhslod: Final[Register]
rep: Final[Register]

# Register Alias
broadcast: Final[Register]  # rep
quad_broadcast: Final[Register]  # quad

rf: Final[list[Register]]
rf0: Final[Register]
rf1: Final[Register]
rf2: Final[Register]
rf3: Final[Register]
rf4: Final[Register]
rf5: Final[Register]
rf6: Final[Register]
rf7: Final[Register]
rf8: Final[Register]
rf9: Final[Register]
rf10: Final[Register]
rf11: Final[Register]
rf12: Final[Register]
rf13: Final[Register]
rf14: Final[Register]
rf15: Final[Register]
rf16: Final[Register]
rf17: Final[Register]
rf18: Final[Register]
rf19: Final[Register]
rf20: Final[Register]
rf21: Final[Register]
rf22: Final[Register]
rf23: Final[Register]
rf24: Final[Register]
rf25: Final[Register]
rf26: Final[Register]
rf27: Final[Register]
rf28: Final[Register]
rf29: Final[Register]
rf30: Final[Register]
rf31: Final[Register]
rf32: Final[Register]
rf33: Final[Register]
rf34: Final[Register]
rf35: Final[Register]
rf36: Final[Register]
rf37: Final[Register]
rf38: Final[Register]
rf39: Final[Register]
rf40: Final[Register]
rf41: Final[Register]
rf42: Final[Register]
rf43: Final[Register]
rf44: Final[Register]
rf45: Final[Register]
rf46: Final[Register]
rf47: Final[Register]
rf48: Final[Register]
rf49: Final[Register]
rf50: Final[Register]
rf51: Final[Register]
rf52: Final[Register]
rf53: Final[Register]
rf54: Final[Register]
rf55: Final[Register]
rf56: Final[Register]
rf57: Final[Register]
rf58: Final[Register]
rf59: Final[Register]
rf60: Final[Register]
rf61: Final[Register]
rf62: Final[Register]
rf63: Final[Register]

# Add ALU instructions
@overload
def fadd(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fadd(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fadd(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def faddnf(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def faddnf(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def faddnf(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vfpack(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vfpack(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vfpack(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def add(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def add(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def add(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def sub(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def sub(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def sub(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fsub(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fsub(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fsub(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def imin(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def imin(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def imin(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def imax(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def imax(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def imax(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def umin(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def umin(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def umin(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def umax(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def umax(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def umax(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def shl(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def shl(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def shl(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def shr(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def shr(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def shr(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def asr(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def asr(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def asr(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def ror(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def ror(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def ror(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fmin(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fmin(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fmin(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fmax(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fmax(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fmax(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...

#
@overload
def vfmin(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vfmin(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vfmin(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def band(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def band(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def band(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def bor(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def bor(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def bor(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def bxor(
    dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def bxor(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def bxor(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vadd(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vadd(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vadd(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vsub(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vsub(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vsub(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def bnot(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def bnot(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def neg(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def neg(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def flapush(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def flapush(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def flbpush(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def flbpush(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def flpop(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def flpop(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def clz(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def clz(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def setmsf(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def setmsf(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def setrevf(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def setrevf(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def nop(cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def tidx(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def eidx(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def lr(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def vfla(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def vflna(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def vflb(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def vflnb(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def xcd(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def ycd(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def msf(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def revf(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def iid(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def sampid(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def barrierid(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def tmuwt(dst: Register = null, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def vpmwt(dst: Register = null, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def flafirst(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def flnafirst(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def fxcd(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
def fycd(dst: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmv_in(
    dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def ldvpmv_in(
    dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmd_in(
    dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def ldvpmd_in(
    dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmp(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ldvpmp(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def recip(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def recip(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def rsqrt(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def rsqrt(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def exp(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def exp(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def log(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def log(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def sin(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def sin(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def rsqrt2(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def rsqrt2(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ballot(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ballot(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def bcastf(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def bcastf(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def alleq(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def alleq(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def allfeq(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def allfeq(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ldvpmg_in(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def ldvpmg_in(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def ldvpmg_in(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def stvpmv(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def stvpmv(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def stvpmd(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def stvpmd(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def stvpmp(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def stvpmp(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def fcmp(
    dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fcmp(
    dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def fcmp(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def vfmax(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vfmax(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vfmax(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fround(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def fround(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftoin(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ftoin(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftrunc(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ftrunc(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftoiz(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ftoiz(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ffloor(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ffloor(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftouz(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ftouz(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def fceil(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def fceil(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def ftoc(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def ftoc(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def fdx(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def fdx(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def fdy(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def fdy(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def itof(dst: Register, src: int, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def itof(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def utof(dst: Register, src: int, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def utof(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def vpack(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vpack(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def vpack(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def v8pack(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def v8pack(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def v8pack(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def fmov(dst: Register, src: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def fmov(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def mov(dst: Register, src: int | float, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithSMIMM: ...
@overload
def mov(dst: Register, src: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> ALUWithoutSMIMM: ...

#
@overload
def v10pack(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def v10pack(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def v10pack(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def v11fpack(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def v11fpack(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def v11fpack(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def quad_rotate(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def quad_rotate(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def quad_rotate(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def rotate(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def rotate(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def rotate(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

#
@overload
def shuffle(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def shuffle(
    dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithSMIMM: ...
@overload
def shuffle(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> ALUWithoutSMIMM: ...

# Mul ALU instructions
@overload
def umul24(dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> None: ...
@overload
def umul24(dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None) -> None: ...
@overload
def umul24(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...

#
@overload
def vfmul(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...
@overload
def vfmul(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...
@overload
def vfmul(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...

#
@overload
def smul24(dst: Register, src1: int, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> None: ...
@overload
def smul24(dst: Register, src1: Register, src2: int, cond: ALUConditionArg = None, sig: SignalArg = None) -> None: ...
@overload
def smul24(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...

#
@overload
def multop(
    dst: Register, src1: int | float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...
@overload
def multop(
    dst: Register, src1: Register, src2: int | float, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...
@overload
def multop(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...

#
def ftounorm16(
    dst: Register,
    src: int | float | Register,
    cond: ALUConditionArg = None,
    sig: SignalArg = None,
) -> None: ...

#
def ftosnorm16(
    dst: Register,
    src: int | float | Register,
    cond: ALUConditionArg = None,
    sig: SignalArg = None,
) -> None: ...

#
def vftounorm8(
    dst: Register,
    src: int | float | Register,
    cond: ALUConditionArg = None,
    sig: SignalArg = None,
) -> None: ...

#
def vftosnorm8(
    dst: Register,
    src: int | float | Register,
    cond: ALUConditionArg = None,
    sig: SignalArg = None,
) -> None: ...

#
def vftounorm10lo(
    dst: Register,
    src: int | float | Register,
    cond: ALUConditionArg = None,
    sig: SignalArg = None,
) -> None: ...

#
def vftounorm10hi(
    dst: Register,
    src: int | float | Register,
    cond: ALUConditionArg = None,
    sig: SignalArg = None,
) -> None: ...

#
@overload
def fmul(dst: Register, src1: float, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None) -> None: ...
@overload
def fmul(dst: Register, src1: Register, src2: float, cond: ALUConditionArg = None, sig: SignalArg = None) -> None: ...
@overload
def fmul(
    dst: Register, src1: Register, src2: Register, cond: ALUConditionArg = None, sig: SignalArg = None
) -> None: ...
