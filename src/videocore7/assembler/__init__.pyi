from _videocore7.assembler import ALU as ALU
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
def fadd(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def faddnf(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vfpack(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def add(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def sub(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fsub(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def imin(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def imax(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def umin(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def umax(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def shl(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def shr(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def asr(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ror(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fmin(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fmax(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vfmin(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def band(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def bor(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def bxor(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vadd(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vsub(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def bnot(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def neg(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def flapush(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def flbpush(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def flpop(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def clz(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def setmsf(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def setrevf(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def nop(
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def tidx(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def eidx(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def lr(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vfla(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vflna(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vflb(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vflnb(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def xcd(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ycd(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def msf(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def revf(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def iid(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def sampid(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def barrierid(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def tmuwt(
    dst: Register = null,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vpmwt(
    dst: Register = null,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def flafirst(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def flnafirst(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fxcd(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fycd(
    dst: Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ldvpmv_in(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ldvpmd_in(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ldvpmp(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def recip(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def rsqrt(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def exp(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def log(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def sin(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def rsqrt2(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ballot(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def bcastf(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def alleq(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def allfeq(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ldvpmg_in(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def stvpmv(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def stvpmd(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def stvpmp(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fcmp(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vfmax(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fround(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftoin(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftrunc(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftoiz(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ffloor(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftouz(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fceil(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftoc(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fdx(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fdy(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def itof(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def utof(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vpack(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def v8pack(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fmov(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def mov(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def v10pack(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def v11fpack(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def quad_rotate(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def rotate(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def shuffle(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...

# Mul ALU instructions
def umul24(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vfmul(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def smul24(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def multop(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftounorm16(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def ftosnorm16(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vftounorm8(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vftosnorm8(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vftounorm10lo(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def vftounorm10hi(
    dst: Register,
    src: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
def fmul(
    dst: Register,
    src1: int | float | Register,
    src2: int | float | Register,
    cond: str | None = None,
    sig: Signal | None = None,
) -> ALU: ...
