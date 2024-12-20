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
02-August-2021  :               (James Korte) : Updated for           MR-BIAS code v0.0
  23-June-2022  :               (James Korte) : GitHub Release        MR-BIAS v1.0
   16-Jan-2023  :               (James Korte) : Goodness of fit added MR-BIAS v1.0.1
   20-Dec-2024  :               (James Korte) : Refactoring
"""
import numpy as np
from collections import OrderedDict


# Code to handle running each module as a test case (from within the module)
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
from mrbias.curve_fit_models.curve_fit_abstract import CurveFitAbstract
import mrbias.misc_utils as mu
from mrbias.misc_utils import LogLevels


class T1VFACurveFitAbstract2Param(CurveFitAbstract):
    def __init__(self, imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                 angle_exclusion_list=None, exclusion_label=None, use_2D_roi=False, centre_offset_2D_list=[0]):
        self.repetition_time = imageset.repetition_time_list[0]
        self.eqn_param_map = OrderedDict()
        self.eqn_param_map["M0"] = ("Equilibrium magnetisation", "max(S(alpha))", "0.0", "inf")
        self.eqn_param_map["T1"] = ("T1 relaxation time", "800.0", "0.0", "inf")
        self.eqn_param_map["alpha"] = ("Flip Angle", "as measured", "-", "-")
        self.eqn_param_map["TR"] = ("Repetition time", "as measured", "-", "-")
        super().__init__(imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                         angle_exclusion_list, exclusion_label,
                         use_2D_roi=use_2D_roi, centre_offset_2D_list=centre_offset_2D_list)

    def get_model_name(self):
        return "T1VFACurveFit2param"

    def get_meas_parameter_name(self):
        return 'FlipAngle'

    def get_symbol_of_interest(self):
        return 'T1'
    def get_ordered_parameter_symbols(self):
        return ['M0', 'T1']
    def get_meas_parameter_symbol(self):
        return 'alpha'

    def fit_function(self, alpha, M0, T1):
        # alpha = flip angle
        # M0 = Weighted equilibrium magnetisation
        # TR = Repetition rate
        # T1 = T1 relaxation time
        TR = self.repetition_time
        alpha_radians = np.deg2rad(alpha)
        return (np.sin(alpha_radians) * M0 * (1.0 - np.exp(-TR/T1))) / (1.0 - np.cos(alpha_radians) * np.exp(-TR/T1))

    def get_initial_parameters(self, roi_dx, voxel_dx):
        cf_roi = self.cf_rois[roi_dx]
        vox_series = cf_roi.get_voxel_series(voxel_dx)
        if cf_roi.is_normalised:
            return [1., 800.]
        else:
            return [np.max(vox_series), 800.]

    def estimate_cf_start_point(self, meas_vec, av_sig, init_val, cf_roi):
        return 800.0

    def fit_parameter_bounds(self):
        return (0., 10.), (np.inf, 3000.)

    def get_model_eqn_strs(self):
        eqn_strs = ["                               ( 1.0       -       exp(-TR/T1) )",
                    "S(alpha)  =  M0 * sin(alpha) * ---------------------------------",
                    "                               ( 1.0 - cos(alpha) * exp(-TR/T1) )"]
        return eqn_strs