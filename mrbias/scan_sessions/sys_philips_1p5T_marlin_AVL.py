"""
Copyright (c) 2021 James Korte, Zachary Chin

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--------------------------------------------------------------------------------
Change Log:
--------------------------------------------------------------------------------
  21-June-2021  : (Zachary Chin, James Korte) : Initial code from masters project
02-August-2021  :               (James Korte) : Updated for MR-BIAS code v0.0
  23-June-2022  :               (James Korte) : GitHub Release   MR-BIAS v1.0
   06-Dec-2024  :               (James Korte) : Refactoring  MR-BIAS v1.0
"""

import pandas as pd
pd.options.mode.chained_assignment = 'raise' # DEBUG: for finding SettingWithCopyWarning

# Code to handle running each module as a test case (from within the module)
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
from mrbias import misc_utils as mu
from mrbias.misc_utils import LogLevels
from mrbias.scan_sessions.abstract_scan_sessions import SystemSessionAbstract, ImageCatetory


class SystemSessionAVLPhilipsMarlinNoGeo(SystemSessionAbstract):
    def __init__(self, dicom_dir, display_unknown_series=True):
        super().__init__(dicom_dir,
                         force_geometry_imageset=ImageCatetory.T1_VFA,
                         display_unknown_series=display_unknown_series)

    def get_geometric_series_UIDs(self):
        return None

    def get_t1_vir_series_UIDs(self):
        df_se_2D = super().get_2D_series()
        df_se_ir_2D = super().get_inversion_recovery_series(df_se_2D)
        df_se_ir_2D_MIR = df_se_ir_2D[df_se_ir_2D.ImageType.str.contains("M_IR", na=False)]
        df_se_ir_2D_MIR = df_se_ir_2D_MIR.sort_values(by=["InversionTime"])
        return df_se_ir_2D_MIR.index

    def get_t1_vfa_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_vfa = df_ge_3D[df_ge_3D.ImageType.str.contains("M_FFE", na=False)]
        df_ge_3D_vfa = df_ge_3D_vfa[df_ge_3D_vfa.SeriesDescription.str.contains("tFA", na=False)]
        df_ge_3D_vfa = df_ge_3D_vfa[df_ge_3D_vfa.SliceThickness <= 10]
        df_ge_3D_vfa = df_ge_3D_vfa.sort_values(by=["FlipAngle"])
        return df_ge_3D_vfa.index

    def get_t2_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        return df_se_2D.index

    def get_proton_density_series_UIDs(self):
        return None

    def get_t2star_series_UIDs(self):
        return None