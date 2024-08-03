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
from mrbias import MRBIAS


# specify the configuration file to control the analysis
configuration_filename = os.path.join("config", "validation_diffusion_config_jk.yaml")
base_monthly_config_filename = os.path.join(r"config\monthly_diffusion", "validation_diffusion_config_monthX.yaml")

# specific the dicom directories to analyse
dicom_directory_str = r"I:\JK\MR-BIAS\Data_From_Maddie\Carr2022_data\%02d_MONTH"

# analyse central slices with a manual ROI placement
for month_num in range(1, 13):
    # load up the base config, modify and save for current month
    manual_conf_filename = os.path.join(r"config\monthly_diffusion",
                                        "validation_diffusion_config_month%d.yaml" % month_num)
    conf = yaml.full_load(open(base_monthly_config_filename))
    conf['roi_detection']['manual_roi_dw_filepath'] = "config/monthly_diffusion/manual_roi_files/MONTH_%02d.yaml" % month_num
    yaml.dump(conf, open(manual_conf_filename, mode="w+"))
    # analyse
    mrb_month = MRBIAS(manual_conf_filename, write_to_screen=True)
    mrb_month.analyse(dicom_directory_str % month_num)
    # clear the temporary configuration file
    os.remove(manual_conf_filename)

# create MRBIAS analysis objects to pull out all voxel data in a large ROI
mrb = MRBIAS(configuration_filename, write_to_screen=True)
for month_num in  range(1, 13):
    mrb.analyse(dicom_directory_str % month_num)

