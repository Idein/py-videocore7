# Copyright (c) 2014-2018 Broadcom
# Copyright (c) 2025- Idein Inc.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
# Street, Fifth Floor, Boston, MA 02110-1301 USA.

import mmap
import os
from collections.abc import Callable
from ctypes import c_uint32, c_void_p, cdll
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

import numpy as np


class _RegisterReaderWriter:
    _read4: Callable[[c_void_p], c_uint32]
    _write4: Callable[[c_void_p, c_uint32], None]

    def __init__(self: Self) -> None:
        stem = Path(__file__).parent / "readwrite4"
        for suffix in EXTENSION_SUFFIXES:
            try:
                lib = cdll.LoadLibrary(str(stem.with_suffix(suffix)))
            except OSError:
                continue
            else:
                break
        else:
            raise Exception("readwrite4 library is not found." + " Your installation seems to be broken.")

        self._read4 = lib.read4
        self._write4 = lib.write4
        del stem, lib

        self._read4.argtypes = [c_void_p]
        self._read4.restype = c_uint32
        self._write4.argtypes = [c_void_p, c_uint32]
        self._write4.restype = None

    def read4(self: Self, ptr: c_void_p) -> c_uint32:
        return self._read4(ptr)

    def write4(self: Self, ptr: c_void_p, value: c_uint32) -> None:
        self._write4(ptr, value)


class Register(Protocol):
    @property
    def value(self: Self) -> int: ...
    @value.setter
    def value(self: Self, value: int) -> None: ...


class HubRegister:
    _reg_rw: _RegisterReaderWriter
    _ptr: int
    _offset: int

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, offset: int) -> None:
        self._reg_rw = reg_rw
        self._ptr = ptr
        self._offset = offset

    @property
    def value(self: Self) -> int:
        return int(self._reg_rw.read4(c_void_p(self._ptr + self._offset)))

    @value.setter
    def value(self: Self, value: int) -> None:
        self._reg_rw.write4(c_void_p(self._ptr + self._offset), c_uint32(value))

    def _proof_compliant(self: Self) -> None:
        _proof_register: Register = self


class PerCoreRegister:
    _reg_rw: _RegisterReaderWriter
    _ptr: int
    _offset: int
    _core_id: int

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, offset: int, core_id: int) -> None:
        self._reg_rw = reg_rw
        self._ptr = ptr
        self._offset = offset

    @property
    def value(self: Self) -> int:
        return int(self._reg_rw.read4(c_void_p(self._ptr + self._offset)))

    @value.setter
    def value(self: Self, value: int) -> None:
        self._reg_rw.write4(c_void_p(self._ptr + self._offset), c_uint32(value))

    def _proof_compliant(self: Self) -> None:
        _proof_register: Register = self


class Field[R: Register]:
    _reg: R
    _mask: int
    _shift: int

    def __init__(self: Self, reg: R, high: int, low: int) -> None:
        self._reg = reg
        self._mask = ((1 << (high - low + 1)) - 1) << low
        self._shift = low

    @property
    def value(self: Self) -> int:
        return (self._reg.value & self._mask) >> self._shift

    @value.setter
    def value(self: Self, value: int) -> None:
        self._reg.value = (self._reg.value & ~self._mask) | ((int(value) << self._shift) & self._mask)

    def _proof_compliant(self: Self) -> None:
        _proof_register: Register = self


# V3D register definitions derived from linux/drivers/gpu/drm/v3d/v3d_regs.h


class HubIdent1(HubRegister):
    _with_mso: Field[Self]
    _with_tsy: Field[Self]
    _with_tfu: Field[Self]
    _with_l3c: Field[Self]
    _nhosts: Field[Self]
    _ncores: Field[Self]
    _rev: Field[Self]
    _tver: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int) -> None:
        super().__init__(reg_rw, ptr, 0x0000C)
        self._with_mso = Field(self, 19, 19)
        self._with_tsy = Field(self, 18, 18)
        self._with_tfu = Field(self, 17, 17)
        self._with_l3c = Field(self, 16, 16)
        self._nhosts = Field(self, 15, 12)
        self._ncores = Field(self, 11, 8)
        self._rev = Field(self, 7, 4)
        self._tver = Field(self, 3, 0)

    @property
    def WITH_MSO(self: Self) -> Field[Self]:  # noqa: N802
        return self._with_mso

    @property
    def WITH_TSY(self: Self) -> Field[Self]:  # noqa: N802
        return self._with_tsy

    @property
    def WITH_TFU(self: Self) -> Field[Self]:  # noqa: N802
        return self._with_tfu

    @property
    def WITH_L3C(self: Self) -> Field[Self]:  # noqa: N802
        return self._with_l3c

    @property
    def NHOSTS(self: Self) -> Field[Self]:  # noqa: N802
        return self._nhosts

    @property
    def NCORES(self: Self) -> Field[Self]:  # noqa: N802
        return self._ncores

    @property
    def REV(self: Self) -> Field[Self]:  # noqa: N802
        return self._rev

    @property
    def TVER(self: Self) -> Field[Self]:  # noqa: N802
        return self._tver


class HubIdent2(HubRegister):
    _mmu: Field[Self]
    _l3c_nkb: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int) -> None:
        super().__init__(reg_rw, ptr, 0x00010)
        self._mmu = Field(self, 8, 8)
        self._l3c_nkb = Field(self, 7, 0)

    @property
    def WITH_MMU(self: Self) -> Field[Self]:  # noqa: N802
        return self._mmu

    @property
    def L3C_NKB(self: Self) -> Field[Self]:  # noqa: N802
        return self._l3c_nkb


class HubIdent3(HubRegister):
    _iprev: Field[Self]
    _ipidx: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int) -> None:
        super().__init__(reg_rw, ptr, 0x00014)
        self._iprev = Field(self, 15, 8)
        self._ipidx = Field(self, 7, 0)

    @property
    def IPREV(self: Self) -> Field[Self]:  # noqa: N802
        return self._iprev

    @property
    def IPIDX(self: Self) -> Field[Self]:  # noqa: N802
        return self._ipidx


class Hub:
    _axicfg: HubRegister
    _uifcfg: HubRegister
    _ident0: HubRegister
    _ident1: HubIdent1
    _ident2: HubIdent2
    _ident3: HubIdent3
    _tfu_cs: HubRegister

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int) -> None:
        self._axicfg = HubRegister(reg_rw, ptr, 0x00000)
        self._uifcfg = HubRegister(reg_rw, ptr, 0x00004)
        self._ident0 = HubRegister(reg_rw, ptr, 0x00008)
        self._ident1 = HubIdent1(reg_rw, ptr)
        self._ident2 = HubIdent2(reg_rw, ptr)
        self._ident3 = HubIdent3(reg_rw, ptr)
        self._tfu_cs = HubRegister(reg_rw, ptr, 0x00700)

    @property
    def AXICFG(self: Self) -> HubRegister:  # noqa: N802
        return self._axicfg

    @property
    def UIFCFG(self: Self) -> HubRegister:  # noqa: N802
        return self._uifcfg

    @property
    def IDENT0(self: Self) -> HubRegister:  # noqa: N802
        return self._ident0

    @property
    def IDENT1(self: Self) -> HubIdent1:  # noqa: N802
        return self._ident1

    @property
    def IDENT2(self: Self) -> HubIdent2:  # noqa: N802
        return self._ident2

    @property
    def IDENT3(self: Self) -> HubIdent3:  # noqa: N802
        return self._ident3

    @property
    def TFU_CS(self: Self) -> HubRegister:  # noqa: N802
        return self._tfu_cs


class CoreIdent0(PerCoreRegister):
    _ver: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        super().__init__(reg_rw, ptr, 0x00000, core_id)
        self._ver = Field(self, 31, 24)

    @property
    def VER(self: Self) -> Field[Self]:  # noqa: N802
        return self._ver


class CoreIdent1(PerCoreRegister):
    _vpm_size: Field[Self]
    _nsem: Field[Self]
    _ntmu: Field[Self]
    _qups: Field[Self]
    _nslc: Field[Self]
    _rev: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        super().__init__(reg_rw, ptr, 0x00004, core_id)
        self._vpm_size = Field(self, 31, 28)
        self._nsem = Field(self, 23, 16)
        self._ntmu = Field(self, 15, 12)
        self._qups = Field(self, 11, 8)
        self._nslc = Field(self, 7, 4)
        self._rev = Field(self, 3, 0)

    @property
    def VPM_SIZE(self: Self) -> Field[Self]:  # noqa: N802
        return self._vpm_size

    @property
    def NSEM(self: Self) -> Field[Self]:  # noqa: N802
        return self._nsem

    @property
    def NTMU(self: Self) -> Field[Self]:  # noqa: N802
        return self._ntmu

    @property
    def QUPS(self: Self) -> Field[Self]:  # noqa: N802
        return self._qups

    @property
    def NSLC(self: Self) -> Field[Self]:  # noqa: N802
        return self._nslc

    @property
    def REV(self: Self) -> Field[Self]:  # noqa: N802
        return self._rev


class CoreIdent2(PerCoreRegister):
    _bcg: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        super().__init__(reg_rw, ptr, 0x00008, core_id)
        self._bcg = Field(self, 28, 28)

    @property
    def BCG(self: Self) -> Field[Self]:  # noqa: N802
        return self._bcg


class CoreMiscCfg(PerCoreRegister):
    _qrmaxcnt: Field[Self]
    _ovrtmuout: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        super().__init__(reg_rw, ptr, 0x00018, core_id)
        self._qrmaxcnt = Field(self, 3, 1)
        self._ovrtmuout = Field(self, 0, 0)

    @property
    def QRMAXCNT(self: Self) -> Field[Self]:  # noqa: N802
        return self._qrmaxcnt

    @property
    def OVRTMUOUT(self: Self) -> Field[Self]:  # noqa: N802
        return self._ovrtmuout


class CoreL2CACTL(PerCoreRegister):
    _l2cclr: Field[Self]
    _l2cdis: Field[Self]
    _l2cena: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        super().__init__(reg_rw, ptr, 0x00020, core_id)
        self._l2cclr = Field(self, 2, 2)
        self._l2cdis = Field(self, 1, 1)
        self._l2cena = Field(self, 0, 0)

    @property
    def L2CCLR(self: Self) -> Field[Self]:  # noqa: N802
        return self._l2cclr

    @property
    def L2CDIS(self: Self) -> Field[Self]:  # noqa: N802
        return self._l2cdis

    @property
    def L2CENA(self: Self) -> Field[Self]:  # noqa: N802
        return self._l2cena


class CoreSLCACTL(PerCoreRegister):
    _tvccs: Field[Self]
    _tdccs: Field[Self]
    _ucc: Field[Self]
    _icc: Field[Self]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        super().__init__(reg_rw, ptr, 0x00024, core_id)
        self._tvccs = Field(self, 27, 24)
        self._tdccs = Field(self, 19, 16)
        self._ucc = Field(self, 11, 8)
        self._icc = Field(self, 3, 0)

    @property
    def TVCCS(self: Self) -> Field[Self]:  # noqa: N802
        return self._tvccs

    @property
    def TDCCS(self: Self) -> Field[Self]:  # noqa: N802
        return self._tdccs

    @property
    def UCC(self: Self) -> Field[Self]:  # noqa: N802
        return self._ucc

    @property
    def ICC(self: Self) -> Field[Self]:  # noqa: N802
        return self._icc


class CorePctr0Src(PerCoreRegister):
    _s: list[Field[PerCoreRegister]]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int, n: int) -> None:
        super().__init__(reg_rw, ptr, 0x00660 + n, core_id)
        self._s = [
            Field(self, 7, 0),
            Field(self, 15, 8),
            Field(self, 23, 16),
            Field(self, 31, 24),
        ]

    @property
    def S(self: Self) -> list[Field[PerCoreRegister]]:  # noqa: N802
        return self._s

    @property
    def S0(self: Self) -> Field[PerCoreRegister]:  # noqa: N802
        return self._s[0]

    @property
    def S1(self: Self) -> Field[PerCoreRegister]:  # noqa: N802
        return self._s[1]

    @property
    def S2(self: Self) -> Field[PerCoreRegister]:  # noqa: N802
        return self._s[2]

    @property
    def S3(self: Self) -> Field[PerCoreRegister]:  # noqa: N802
        return self._s[3]


class Core:
    _ident0: CoreIdent0
    _ident1: CoreIdent1
    _ident2: CoreIdent2
    _misccfg: CoreMiscCfg
    _l2cactl: CoreL2CACTL
    _slcactl: CoreSLCACTL
    _pctr_0_en: PerCoreRegister
    _pctr_0_clr: PerCoreRegister
    _pctr_0_overflow: PerCoreRegister
    _pctr_0_src: list[CorePctr0Src]
    _pctr_0_pctr: list[PerCoreRegister]

    def __init__(self: Self, reg_rw: _RegisterReaderWriter, ptr: int, core_id: int) -> None:
        self._ident0 = CoreIdent0(reg_rw, ptr, core_id)
        self._ident1 = CoreIdent1(reg_rw, ptr, core_id)
        self._ident2 = CoreIdent2(reg_rw, ptr, core_id)
        self._misccfg = CoreMiscCfg(reg_rw, ptr, core_id)
        self._l2cactl = CoreL2CACTL(reg_rw, ptr, core_id)
        self._slcactl = CoreSLCACTL(reg_rw, ptr, core_id)
        self._pctr_0_en = PerCoreRegister(reg_rw, ptr, 0x00650, core_id)
        self._pctr_0_clr = PerCoreRegister(reg_rw, ptr, 0x00654, core_id)
        self._pctr_0_overflow = PerCoreRegister(reg_rw, ptr, 0x00658, core_id)
        self._pctr_0_src = [CorePctr0Src(reg_rw, ptr, core_id, i) for i in range(0, 32, 4)]
        self._pctr_0_pctr = [PerCoreRegister(reg_rw, ptr, 0x00680 + 4 * i, core_id) for i in range(32)]

    @property
    def IDENT0(self: Self) -> CoreIdent0:  # noqa: N802
        return self._ident0

    @property
    def IDENT1(self: Self) -> CoreIdent1:  # noqa: N802
        return self._ident1

    @property
    def IDENT2(self: Self) -> CoreIdent2:  # noqa: N802
        return self._ident2

    @property
    def MISCCFG(self: Self) -> CoreMiscCfg:  # noqa: N802
        return self._misccfg

    @property
    def L2CACTL(self: Self) -> CoreL2CACTL:  # noqa: N802
        return self._l2cactl

    @property
    def SLCACTL(self: Self) -> CoreSLCACTL:  # noqa: N802
        return self._slcactl

    @property
    def PCTR_0_EN(self: Self) -> PerCoreRegister:  # noqa: N802
        return self._pctr_0_en

    @property
    def PCTR_0_CLR(self: Self) -> PerCoreRegister:  # noqa: N802
        return self._pctr_0_clr

    @property
    def PCTR_0_OVERFLOW(self: Self) -> PerCoreRegister:  # noqa: N802
        return self._pctr_0_overflow

    @property
    def PCTR_0_SRC(  # noqa: N802
        self: Self,
    ) -> list[CorePctr0Src]:
        return self._pctr_0_src

    @property
    def PCTR_0_SRC_0_3(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[0]

    @property
    def PCTR_0_SRC_4_7(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[1]

    @property
    def PCTR_0_SRC_8_11(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[2]

    @property
    def PCTR_0_SRC_12_15(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[3]

    @property
    def PCTR_0_SRC_16_19(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[4]

    @property
    def PCTR_0_SRC_20_23(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[5]

    @property
    def PCTR_0_SRC_24_27(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[6]

    @property
    def PCTR_0_SRC_28_31(self: Self) -> CorePctr0Src:  # noqa: N802
        return self._pctr_0_src[7]

    @property
    def PCTR_0_PCTR(  # noqa: N802
        self: Self,
    ) -> list[PerCoreRegister]:
        return self._pctr_0_pctr


CORE_PCTR_CYCLE_COUNT: Final[int] = 0


class RegisterMapping:
    NUM_OF_CORES: Final[int] = 1

    _hub: Hub
    _core: list[Core]
    _map: list[mmap.mmap]

    def __init__(self: Self) -> None:
        self._core = []
        self._map = []

        fd = os.open("/dev/mem", os.O_RDWR)

        path: str
        names: list[str]
        reg_addrs: dict[str, tuple[int, int]] = {}
        with open("/proc/device-tree/__symbols__/v3d", "rb") as f:
            path = f.read().decode("utf-8").split("\0")[0]
        with open("/proc/device-tree" + path + "/reg-names", "rb") as f:
            names = [name for name in f.read().decode("utf-8").split("\0") if len(name) > 0]
        with open("/proc/device-tree" + path + "/reg", "rb") as f:
            for name in names:
                addr = int.from_bytes(f.read(8))
                size = int.from_bytes(f.read(8))
                reg_addrs[name] = (addr, size)

        if "hub" not in reg_addrs:
            raise RuntimeError("Failed to get hub register addresses.")
        reg_rw = _RegisterReaderWriter()
        map_hub = mmap.mmap(
            offset=reg_addrs["hub"][0],
            length=reg_addrs["hub"][1],
            fileno=fd,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        ptr_hub = np.frombuffer(map_hub).ctypes.data
        self._hub = Hub(reg_rw, ptr_hub)
        self._map.append(map_hub)

        self._core = []
        for core in range(RegisterMapping.NUM_OF_CORES):
            if f"core{core}" not in reg_addrs:
                raise RuntimeError(f"Failed to get core{core} register addresses.")
            map_core = mmap.mmap(
                offset=reg_addrs[f"core{core}"][0],
                length=reg_addrs[f"core{core}"][1],
                fileno=fd,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            ptr_core = np.frombuffer(map_core).ctypes.data
            self._core.append(Core(reg_rw, ptr_core, core))
            self._map.append(map_core)

        os.close(fd)

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> None:
        for m in self._map:
            m.close()

    @property
    def HUB(self: Self) -> Hub:  # noqa: N802
        return self._hub

    @property
    def CORE(self: Self) -> list[Core]:  # noqa: N802
        return self._core


class PerformanceCounter:
    _reg: RegisterMapping
    _srcs: list[int]
    _core: int
    _mask: int

    def __init__(self: Self, _regmap: RegisterMapping, srcs: list[int], core: int = 0) -> None:
        self._reg = _regmap
        self._srcs = srcs
        self._core = core  # Sufficient for now.
        self._mask = (1 << len(self._srcs)) - 1

    def __enter__(self: Self) -> Self:
        pctr_srcs = self._reg.CORE[self._core].PCTR_0_SRC
        for reg, value in zip(sum([s.S for s in pctr_srcs], []), self._srcs):
            reg.value = value

        self._reg.CORE[self._core].PCTR_0_EN.value = self._mask
        self._reg.CORE[self._core].PCTR_0_CLR.value = self._mask
        self._reg.CORE[self._core].PCTR_0_OVERFLOW.value = self._mask

        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[TracebackType] | None,
    ) -> None:
        self._reg.CORE[self._core].PCTR_0_EN.value = 0
        self._reg.CORE[self._core].PCTR_0_CLR.value = self._mask
        self._reg.CORE[self._core].PCTR_0_OVERFLOW.value = self._mask

    def result(self: Self) -> list[int]:
        return [reg.value for reg in self._reg.CORE[self._core].PCTR_0_PCTR[: len(self._srcs)]]
