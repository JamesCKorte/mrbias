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
10-July-2023  :               (James Korte) : Initial code to help users add a region of interest detection template
"""
# load up relevant modules
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Code to add the parent directory to allow importing mrbias core modules
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
import mrbias.roi_detect as roi_detect
import mrbias.image_sets as image_sets
import mrbias.misc_utils as mu



#################################################################################
# INSTRUCTIONS
#################################################################################
# Set the following variable:
# -----------------------------------------------------------------------------
# - ROI_TEMPLATE_NAME: the name of the sub-directory you have put the new ROI template in
#                    : this is expected to be a sub-directory of mrbias/roi_detection_templates
#
# Check the results:
# -------------------------------------------------------------------------------
# Visually check the figure which is drawn, showing the ROIs on the template image
#################################################################################
ROI_TEMPLATE_NAME = "siemens_diffusion_no_ice"
#################################################################################



# define the test template
TEST_TEMPLATE_DIR = os.path.join(mu.reference_template_directory(), ROI_TEMPLATE_NAME)
# create the geometric image
sitk_im = mu.get_sitk_image_from_dicom_image_folder(os.path.join(TEST_TEMPLATE_DIR, "dicom"))
test_geom_im = image_sets.ImageGeometric("TemplateImage", sitk_im)

# create a roi detector
roi_detector = roi_detect.ROIDetector(test_geom_im, TEST_TEMPLATE_DIR)

# draw it
f, (ax1) = plt.subplots(1, 1)
roi_detector.visualise_fixed_DW_rois(ax1)

print("------------- FIN ------------------")
plt.show()
