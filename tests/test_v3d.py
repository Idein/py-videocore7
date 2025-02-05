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
from _videocore7.drm_v3d import DRM_V3D
from videocore7.v3d import *


def test_v3d_regs() -> None:
    with DRM_V3D() as drm:
        try:
            with RegisterMapping() as reg:
                assert reg.HUB.UIFCFG.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_UIFCFG)
                assert reg.HUB.IDENT1.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT1)
                assert reg.HUB.IDENT2.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT2)
                assert reg.HUB.IDENT3.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_HUB_IDENT3)
                assert reg.CORE[0].IDENT0.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT0)
                assert reg.CORE[0].IDENT1.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT1)
                assert reg.CORE[0].IDENT2.value == drm.v3d_get_param(DRM_V3D.V3D_PARAM_V3D_CORE0_IDENT2)
        except PermissionError:
            print("Skipping tests because of a lack of root privilege")
