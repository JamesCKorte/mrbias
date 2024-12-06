"""
Copyright (c) 2024 James Korte, Zachary Chin, Stanley Norris

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
02-August-2021  :               (James Korte) : Initial code for MR-BIAS v0.0
  23-June-2022  :               (James Korte) : GitHub Release   MR-BIAS v1.0
   05-Dec-2024  :               (James Korte) : Diffusion Update MR-BIAS v2.0
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code to add the parent directory to allow importing mrbias core modules
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
base_dir = os.path.join(root, "..")
if str(base_dir) not in sys.path:
    sys.path.insert(1, str(base_dir))
# import required mrbias modules
from mrbias import MRBIAS

# DEFINE THE TEST TOLERANCES
MAX_ADC_ERROR_UM2pS = 5.00 # um2/s
# SWITCH THE VISUALISATION ON/OFF
VISUALISE_RESULTS = False


# specify the configuration file to control the analysis
configuration_filename = "e2e_config.yaml"
# specific the dicom directories to analyse
dicom_directory = os.path.join(base_dir, "data", "mrbias_testset_C")


# -------------------------------------------------------------------------------------------------------
# RUN THE PROGRAM
# -------------------------------------------------------------------------------------------------------
# create a MRBIAS analysis object
mrb = MRBIAS(configuration_filename, write_to_screen=True)
# run the analysis (output will be created in the "output_directory" specified in the configuration file)
mrb.analyse(dicom_directory)


# -------------------------------------------------------------------------------------------------------
# COMPARE THE RESULTS TO A STABLE VERSION OF THE SOFTWARE
# -------------------------------------------------------------------------------------------------------
# ground truth datafiles
gt_modelfit_datafile_list = []
for dw_dx in range(5):
    gt_modelfit_datafile_list.append(os.path.join("ComparisonData", "dw_%03d_model_fit_summary.csv" % dw_dx))
# new results datafiles
now_modelfit_datafile_list = []
for dw_dx in range(5):
    now_modelfit_datafile_list.append(os.path.join("VCCC - Parkville_Siemens Healthineers-MAGNETOM Skyra-3p0T-46069_20241010-120550",
                                                   "dw_%03d_DWCurveFit2param_AvROI_NrmROIMax" % dw_dx,
                                                   "model_fit_summary.csv"))


# # ! Might need to update the folder selection to use REGEX in the future (as per previous issues)


# # - we added a regex as the "time" at the end of the foldername may be prone to change with slightly different
# #   ScanSession filtering rules (a different series may be used - but the date should be stable) ...
# #   unless the data was taken over two dates (i.e. a midnight scanning session)
# bdir_dset_1 = None
# bdir_dset_2 = None
# regex_1 = re.compile('^PMCC_SIEMENS-Skyra-3p0T-46069_20190914*')
# regex_2 = re.compile('^PMCC_SIEMENS-Skyra-3p0T-46069_20200706*')
# for root, dirs, files in os.walk("."):
#   for d in dirs:
#     if regex_1.match(d):
#         bdir_dset_1 = d
#     if regex_2.match(d):
#         bdir_dset_2 = d
# assert ((bdir_dset_1 is not None) and (bdir_dset_2 is not None)), "Unable to find to two expected (recently generated) output directories\n\t-%s\n\t-%s" \
#                                                                   % ("./PMCC_SIEMENS-Skyra-3p0T-46069_20190914...", "./PMCC_SIEMENS-Skyra-3p0T-46069_20200706...")

if VISUALISE_RESULTS:
    f, ax_list = plt.subplots(5, 1)
else:
    ax_list = [None, None, None, None, None]

pass_vec = []
for dw_dx, (dset_gt, dset_now, ax) in enumerate(zip(gt_modelfit_datafile_list,
                                                    now_modelfit_datafile_list,
                                                    ax_list)):
    # compare the DWresults
    df_gt  = pd.read_csv(dset_gt)
    df_now = pd.read_csv(dset_now)
    # add a analysis tag to both frames
    df_gt.loc[:, "Analysis"]  = "GroundTruth"
    df_now.loc[:, "Analysis"] = "Current"
    # concatenate and visualise
    if VISUALISE_RESULTS:
        df = pd.concat([df_gt, df_now])
        sns.stripplot(x="RoiLabel", y="D (mean)", hue="Analysis", data=df,
                        jitter=True, ax=ax)
        plt.pause(0.01)
    # compare results
    DW_diff_per_roi = df_now["D (mean)"] - df_gt["D (mean)"]
    DW_PASSED = np.max(np.abs(DW_diff_per_roi)) < MAX_ADC_ERROR_UM2pS

    # print the test results
    print("DW difference [dw_%03d]: ADC : average = %.5f ms {min (%.5f um^2/s), max (%.5f um^2/s)} : Test if max < %.5f um^2/s ? [Pass = %s]" %
          (dw_dx, np.mean(np.abs(DW_diff_per_roi)), np.min(DW_diff_per_roi), np.max(DW_diff_per_roi), MAX_ADC_ERROR_UM2pS, str(DW_PASSED)))

    pass_vec.append(DW_PASSED)

    if VISUALISE_RESULTS:
        plt.pause(0.01)

assert not (False in pass_vec), "End to end test: FAILED!!!"

print("End to end test: PASSED")
if VISUALISE_RESULTS:
    plt.show()