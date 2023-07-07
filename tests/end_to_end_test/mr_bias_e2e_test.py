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
02-August-2021  :               (James Korte) : Initial code for MR-BIAS v0.0
  23-June-2022  :               (James Korte) : GitHub Release   MR-BIAS v1.0
"""
import os
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
MAX_T1_ERROR_MS = 5.00 # ms
MAX_T2_ERROR_MS = 5.00 # ms


# specify the configuration file to control the analysis
configuration_filename = "e2e_config.yaml"
# specific the dicom directories to analyse
dicom_directory_a = os.path.join(base_dir, "data", "mrbias_testset_A")
dicom_directory_b = os.path.join(base_dir, "data", "mrbias_testset_B")


# -------------------------------------------------------------------------------------------------------
# RUN THE PROGRAM
# -------------------------------------------------------------------------------------------------------
# create a MRBIAS analysis object
mrb = MRBIAS(configuration_filename, write_to_screen=True)
# run the analysis (output will be created in the "output_directory" specified in the configuration file)
mrb.analyse(dicom_directory_a)
# run analysis on a second dicom directory (with the same analysis settings)
mrb.analyse(dicom_directory_b)


# -------------------------------------------------------------------------------------------------------
# COMPARE THE RESULTS TO A STABLE VERSION OF THE SOFTWARE
# -------------------------------------------------------------------------------------------------------
# ground truth directories
bdir_dset_gt_1 = os.path.join("ComparisonData","PMCC_46069_20190914-160825")
bdir_dset_gt_2 = os.path.join("ComparisonData","PMCC_46069_20200706-201432")
# the recent results for checkings
bdir_dset_1 = "PMCC_SIEMENS-Skyra-3p0T-46069_20190914-160825"
bdir_dset_2 = "PMCC_SIEMENS-Skyra-3p0T-46069_20200706-201432"

for bdir_dset_gt, bdir_dset in zip([bdir_dset_gt_1, bdir_dset_gt_2],
                                   [bdir_dset_1, bdir_dset_2]):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1)

    # compare the T1-VIR results
    df_gt  = pd.read_csv(os.path.join(bdir_dset_gt, "t1-vir_model_fit_summary.csv"))
    df_now = pd.read_csv(os.path.join(bdir_dset, "t1_vir_000_T1VIRCurveFit4param_AvROI_NrmVoxMax_ExclClip-100pct_no1500ms", "model_fit_summary.csv"))
    # add a analysis tag to both frames
    df_gt["Analysis"]  = "GroundTruth"
    df_now["Analysis"] = "Current"
    # concatenate and visualise
    df = pd.concat([df_gt, df_now])
    sns.stripplot(x="RoiLabel", y="T1 (mean)", hue="Analysis", data=df,
                    jitter=True, ax=ax1)
    plt.pause(0.01)
    # compare results
    T1_vir_diff_per_roi = df_now["T1 (mean)"] - df_gt["T1 (mean)"]
    VIR_PASSED = np.abs(np.max(T1_vir_diff_per_roi)) < MAX_T1_ERROR_MS

    # compare the T1-VFA results
    df_gt = pd.read_csv(os.path.join(bdir_dset_gt, "t1-vfa_model_fit_summary.csv"))
    df_now = pd.read_csv(
        os.path.join(bdir_dset, "t1_vfa_000_T1VFACurveFit2param_AvROI_NrmVoxMax_ExclClip-100pct_no15deg",
                     "model_fit_summary.csv"))
    # add a analysis tag to both frames
    df_gt["Analysis"] = "GroundTruth"
    df_now["Analysis"] = "Current"
    # concatenate and visualise
    df = pd.concat([df_gt, df_now])
    sns.stripplot(x="RoiLabel", y="T1 (mean)", hue="Analysis", data=df,
                  jitter=True, ax=ax2)
    plt.pause(0.01)
    # compare results
    T1_vfa_diff_per_roi = df_now["T1 (mean)"] - df_gt["T1 (mean)"]
    VFA_PASSED = np.abs(np.max(T1_vfa_diff_per_roi)) < MAX_T1_ERROR_MS

    # compare the T2-MSE results
    df_gt = pd.read_csv(os.path.join(bdir_dset_gt, "t2-mse_model_fit_summary.csv"))
    df_now = pd.read_csv(
        os.path.join(bdir_dset, "t2_mse_000_T2SECurveFit3param_AvROI_NrmVoxMax_ExclClip-100pct_no10ms",
                     "model_fit_summary.csv"))
    # add a analysis tag to both frames
    df_gt["Analysis"] = "GroundTruth"
    df_now["Analysis"] = "Current"
    # concatenate and visualise
    df = pd.concat([df_gt, df_now])
    sns.stripplot(x="RoiLabel", y="T2 (mean)", hue="Analysis", data=df,
                  jitter=True, ax=ax3)
    plt.pause(0.01)
    # compare results
    T2_mse_diff_per_roi = df_now["T2 (mean)"] - df_gt["T2 (mean)"]
    MSE_PASSED = np.abs(np.max(T2_mse_diff_per_roi)) < MAX_T2_ERROR_MS

    # print the test results
    print("T1-VIR difference: average = %.5f ms {min (%.5f ms), max (%.5f ms)} : Test if max < %.5f ms ? [Pass = %s]" %
          (np.mean(np.abs(T1_vir_diff_per_roi)), np.min(T1_vir_diff_per_roi), np.max(T1_vir_diff_per_roi), MAX_T1_ERROR_MS, str(VIR_PASSED)))
    print("T1-VFA difference: average = %.5f ms {min (%.5f ms), max (%.5f ms)} : Test if max < %.5f ms ? [Pass = %s]" %
          (np.mean(np.abs(T1_vfa_diff_per_roi)), np.min(T1_vfa_diff_per_roi), np.max(T1_vfa_diff_per_roi), MAX_T1_ERROR_MS, str(VFA_PASSED)))
    print("T2-MSE difference: average = %.5f ms {min (%.5f ms), max (%.5f ms)} : Test if max < %.5f ms ? [Pass = %s]" %
          (np.mean(np.abs(T2_mse_diff_per_roi)), np.min(T2_mse_diff_per_roi), np.max(T2_mse_diff_per_roi), MAX_T2_ERROR_MS, str(MSE_PASSED)))

    assert VIR_PASSED and VFA_PASSED and MSE_PASSED, "End to end test: FAILED!!!"

    plt.pause(0.01)

print("End to end test: PASSED")
plt.show()