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
import yaml
import gc
from mrbias import MRBIAS

DATA_BASE_DIR = r"I:\JK\MR-BIAS\EMBRACE\EMBRACE_DATA2_ALL"

DATA_CONFIG_DICT = {"InstituteA": (os.path.join(DATA_BASE_DIR, "InstA_phantomQA"), # Philips 1.5T Ingenia
                                   ("config/embrace/diffusion_philips_config.yaml", None),
                                   None,
                                   1.5),
                    "InstituteB": (os.path.join(DATA_BASE_DIR, "InstB_phantomQA"), # Philips 1.5T Ingenia
                                   ("config/embrace/diffusion_philips_config.yaml", None),
                                   None,
                                   1.5),
                    "InstituteC": (os.path.join(DATA_BASE_DIR, "InstC_phantomQA"), # Philips 1.5T Ingenia
                                   ("config/embrace/diffusion_philips_config.yaml", None),
                                   None,
                                   1.5),
                    "InstituteD": (os.path.join(DATA_BASE_DIR, "InstD_phantomQA"), # Siemens 1.5T Aera
                                   ("config/embrace/diffusion_siemens_config.yaml", None),
                                   None,
                                   1.5),
                    "InstituteE": (os.path.join(DATA_BASE_DIR, "InstE_phantomQA"), # GE 1.5T Optima
                                   ("config/embrace/diffusion_GE_config.yaml", [11]),
                                   None,
                                   1.5),
                    # "InstituteF": (os.path.join(DATA_BASE_DIR, "InstF_phantomQA"),  # Philips 3.0T Ingenia (Enhanced/MultiFrame DICOM)
                    #                "config/embrace/diffusion_philips_config.yaml",
                    #                None,
                    #                3.0),
                    "InstituteG": (os.path.join(DATA_BASE_DIR, "InstG_phantomQA"),  # Philips 3.0T Ingenia
                                   ("config/embrace/diffusion_philips_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteH": (os.path.join(DATA_BASE_DIR, "InstH_phantomQA"),  # Philips 3.0T Ingenia
                                   ("config/embrace/diffusion_philips_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteI": (os.path.join(DATA_BASE_DIR, "InstI_phantomQA"),  # Philips 3.0T Ingenia
                                   ("config/embrace/diffusion_philips_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteJ": (os.path.join(DATA_BASE_DIR, "InstJ_phantomQA"), # Siemens 3.0T Skyra
                                   ("config/embrace/diffusion_siemens_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteK": (os.path.join(DATA_BASE_DIR, "InstK_phantomQA"), # Siemens 3.0T PrismaFit?
                                   ("config/embrace/diffusion_siemens_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteL": (os.path.join(DATA_BASE_DIR, "InstL_phantomQA"), # Siemens 3.0T Skyra
                                   ("config/embrace/diffusion_siemens_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteM": (os.path.join(DATA_BASE_DIR, "InstM_phantomQA"), # Siemens 3.0T Verio
                                   ("config/embrace/diffusion_siemens_config.yaml", None),
                                   None,
                                   3.0),
                    "InstituteN": (os.path.join(DATA_BASE_DIR, "InstN_phantomQA"), # GE 3.0T Discovery, Trial Series Not Used (due to 2 degrees increase)
                                   ("config/embrace/diffusion_GE_discovery_config.yaml", None),
                                   None,
                                   3.0)}


# DATA_CONFIG_DICT = {"InstituteK": (os.path.join(DATA_BASE_DIR, "InstK_phantomQA"),  # Siemens 3.0T Skyra (Missing Data?)
#                                    ("config/embrace/diffusion_siemens_config.yaml", None),
#                                    None,
#                                    3.0)}

# DATA_CONFIG_DICT = {"InstituteA": (os.path.join(DATA_BASE_DIR, "InstA_phantomQA"), # Philips 1.5T Ingenia
#                                    ("config/embrace/diffusion_philips_config.yaml", None),
#                                    None,
#                                    1.5)}

# DATA_CONFIG_DICT = {"InstituteB": (os.path.join(DATA_BASE_DIR, "InstB_phantomQA"), # Philips 1.5T Ingenia
#                                    ("config/embrace/diffusion_philips_config.yaml", None),
#                                    None,
#                                    1.5)}
#
# # D has ADC maps associated (for checking derived maps settings)
# DATA_CONFIG_DICT = {"InstituteD": (os.path.join(DATA_BASE_DIR, "InstD_phantomQA"), # Siemens 1.5T Aera
#                                    ("config/embrace/diffusion_siemens_config.yaml", None),
#                                    None,
#                                    1.5)}

TURN_ON_SLICE_AVERAGING = False
TURN_ON_ROI_FINE_TUNING = True
FIT_ALL_VOXELS = False

tmp_filename = 'temp.yaml'

# loop over the institutes and analyse both diffusion and T1/T2 phantom
for inst_label, (data_dir, dw_config, estar_config, field_str) in DATA_CONFIG_DICT.items():
    # load up the base config, modify the output directory
    dw_config_file, dw_cap_dir_series_numbers = dw_config
    conf = yaml.full_load(open(dw_config_file))
    base_out_dir = conf['global']['output_directory'] + "_II_r5_h10_ft" #"_II_r8_h10_ft"
    conf["curve_fitting"]['dw_options']['use_2D_roi'] = False
    if TURN_ON_SLICE_AVERAGING:
        base_out_dir = conf['global']['output_directory'] + "_II_r8_ft_sliceAv"
        conf["curve_fitting"]['dw_options']['use_2D_roi'] = False
        conf["curve_fitting"]['averaging'] = "voxels_in_slice"
    if FIT_ALL_VOXELS:
        base_out_dir = conf['global']['output_directory'] + "_II_r8_h35_ft_allVox"
        conf["curve_fitting"]['averaging'] = None
    if TURN_ON_ROI_FINE_TUNING:
        conf['roi_detection']['shape_fine_tune'] = True


    inst_path = base_out_dir + '/' + inst_label
    if not os.path.isdir(inst_path):
        os.mkdir(inst_path)
    conf['global']['output_directory'] = inst_path + "/allBvals"
    conf['phantom_experiment']['field_strength_tesla'] = field_str
    if(dw_cap_dir_series_numbers is not None):
        conf['roi_detection']['flip_cap_series_numbers'] = dw_cap_dir_series_numbers
    yaml.dump(conf, open(tmp_filename, mode="w+"))
    # analyse DW data
    mrb = MRBIAS(tmp_filename, write_to_screen=True)
    mrb.analyse(data_dir)

    # analyse again, but exclude b-values to match EMBRACE analysis for the institutional protocols
    if inst_label == "InstituteA":
        # exclude specfic b-values to use only [200, 1000]
        conf['global']['output_directory'] = base_out_dir + '/' + inst_label + "/specBVals"
        conf["curve_fitting"]['dw_options']['bval_exclusion_list'] = [0.0, 600.0e-6] # bvals are 0,200, 600, 1000
        conf["curve_fitting"]['dw_options']['bval_exclusion_label'] = "specBVals"
    elif inst_label == "InstituteC":
        # exclude specfic b-values to use only [200, 1000]
        conf['global']['output_directory'] = base_out_dir + '/' + inst_label + "/specBVals"
        conf["curve_fitting"]['dw_options']['bval_exclusion_list'] = [0.0, 50.0e-6, 700.0e-6] # bvals are 0, 50, 200, 700, 1000
        conf["curve_fitting"]['dw_options']['bval_exclusion_label'] = "specBVals"
    elif inst_label == "InstituteL":
        # exclude specfic b-values to use only [200, 1000]
        conf['global']['output_directory'] = base_out_dir + '/' + inst_label + "/specBVals"
        conf["curve_fitting"]['dw_options']['bval_exclusion_list'] = [0.0, 800.0e-6] # bvals are 0, 200, 800, 1000
        conf["curve_fitting"]['dw_options']['bval_exclusion_label'] = "specBVals"
    else:
        # exclude the b=0 value
        conf['global']['output_directory'] = base_out_dir + '/' + inst_label + "/noBval0"
        conf["curve_fitting"]['dw_options']['bval_exclusion_list'] = [0.0]
        conf["curve_fitting"]['dw_options']['bval_exclusion_label'] = "excludeBval0"
    yaml.dump(conf, open(tmp_filename, mode="w+"))
    mrb = MRBIAS(tmp_filename, write_to_screen=True)
    mrb.analyse(data_dir)


    # clear the temporary configuration file
    os.remove(tmp_filename)
    # free memory just in case
    del mrb
    gc.collect()




