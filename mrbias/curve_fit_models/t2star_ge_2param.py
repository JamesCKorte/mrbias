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
   25-Jan-2023  :              (Arpita Dutta) : T2 Star signal model added
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


class T2StarCurveFitAbstract2Param(CurveFitAbstract):
    def __init__(self, imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                 echo_exclusion_list=None, exclusion_label=None):
        self.eqn_param_map = OrderedDict()
        self.eqn_param_map["M0"] = ("Equilibrium magnetisation", "max(S(TE))", "0.0", "inf")
        self.eqn_param_map["T2Star"] = ("T2Star relaxation time", "TE_median", "0.0", "inf")
        self.eqn_param_map["TE"] = ("Echo time", "as measured", "-", "-")
        self.eqn_param_map["TE_median"] = ("The median measured echo time", "from signal", "-", "-")
        super().__init__(imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                         echo_exclusion_list, exclusion_label)

    def get_model_name(self):
        return "T2StarCurveFit2param"

    def get_meas_parameter_name(self):
        return 'EchoTime'

    def get_symbol_of_interest(self):
        return 'T2Star'
    def get_ordered_parameter_symbols(self):
        return ['M0', 'T2Star']
    def get_meas_parameter_symbol(self):
        return 'TE'

    def fit_function(self, TE, M0, T2Star):
        # M0 = Weighted equilibrium magnetisation
        # T2Star = T2Star relaxation time
        # TE = echo time
        return M0 * np.exp(-TE/T2Star)

    def get_initial_parameters(self, roi_dx, voxel_dx):
        cf_roi = self.cf_rois[roi_dx]
        vox_series = cf_roi.get_voxel_series(voxel_dx)
        init_val = cf_roi.initialisation_value #self.initialisation_phantom.get_roi_by_dx(roi_dx)
        if cf_roi.is_normalised:
            return [1.,  init_val]
        else:
            return [np.max(vox_series),  init_val]

    def fit_parameter_bounds(self):
        return (0., 0.), (np.inf, np.inf)

    def estimate_cf_start_point(self, meas_vec, av_sig, init_val, cf_roi):
        try:
            # half_val = np.max(av_sig)*0.368
            # half_val_idx = np.argmin(np.abs(av_sig-half_val))
            # return meas_vec[half_val_idx]
            # use median
            return np.median(cf_roi.meas_var_vector)
        except:
            mu.log("T2StarCurveFitAbstract2Param::estimate_cf_start_point(): failed to estimate start point, using "
                   "default values of %.3f" % init_val, LogLevels.LOG_WARNING)
            return init_val

    def get_model_eqn_strs(self):
        eqn_strs = ["S(TE) = M0 * exp(-TE/T2Star)"]
        return eqn_strs