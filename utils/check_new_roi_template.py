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
10-July-2023  :               (James Korte) : Initial code to help users test a new detection template
"""
# load up relevant modules
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape

# Code to add the parent directory to allow importing mrbias core modules
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
base_dir = root
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
import mrbias.roi_detect as roi_detect
import mrbias.scan_session as ss
import mrbias.image_sets as image_sets
import mrbias.misc_utils as mu



#################################################################################
# INSTRUCTIONS
#################################################################################
#
# Set the following variables:
# -----------------------------------------------------------------------------
# - ROI_TEMPLATE_NAME: the name of the sub-directory you have put the new ROI template in
#                    : this is expected to be a sub-directory of mrbias/roi_detection_templates
# - DICOM_DIRECTORY:     the directory of dicom images you want to detect ROIs on
#
#
# Create your new ScanSession<YourClassName> class for testing:
# --------------------------------------------------------------------------------
# i.e. scan_session = ss.ScanSessionYourClassName(DICOM_DIRECTORY)
#
#
# Check the results:
# -------------------------------------------------------------------------------
#   The script will log information to the terminal while it processes the data
# you can read this to check things are working correctly.
#   There is also a pdf generated ("check_roi_detect.pdf") which you can
# visually inspect to check the ROIs are in the correct locations
#################################################################################
ROI_TEMPLATE_NAME = "siemens_skyra_3p0T"
DICOM_DIRECTORY = os.path.join(base_dir, "data", "mrbias_testset_B")
scan_session = ss.SystemSessionSiemensSkyra(DICOM_DIRECTORY)
registration_method = roi_detect.RegistrationOptions.TWOSTAGE_MSMEGS_CORELGD # default
partial_fov = False
#################################################################################



# setup a pdf to test the pdf reporting
pdf = mu.PDFSettings()
c = canvas.Canvas("check_roi_detect.pdf", landscape(pdf.page_size))
scan_session.write_pdf_summary_page(c)                      # write to pdf for checking
# get the T1 and T2 imagesets
geometric_images = scan_session.get_geometric_images()
t1_vir_imagesets = scan_session.get_t1_vir_image_sets()
t1_vfa_imagesets = scan_session.get_t1_vfa_image_sets()
t2_mse_imagesets = scan_session.get_t2_mse_image_sets()
t2star_imagesets = scan_session.get_t2star_image_sets()
dw_imagesets = scan_session.get_dw_image_sets()


# define the test template
TEST_TEMPLATE_DIR = os.path.join(mu.reference_template_directory(), ROI_TEMPLATE_NAME)
# create the geometric image
sitk_im = mu.get_sitk_image_from_dicom_image_folder(os.path.join(TEST_TEMPLATE_DIR, "dicom"))
test_geom_im = image_sets.ImageGeometric("TemplateImage", sitk_im)

# Perform ROI detection on all geometric images
# - this stores the result for use by the related imagesets
for geometric_image in geometric_images:
    # detect the ROIs and return the masks on the target image
    roi_detector = roi_detect.ROIDetector(geometric_image, TEST_TEMPLATE_DIR,
                                          registration_method=registration_method,
                                          partial_fov=partial_fov)
    roi_detector.detect()
    # output to pdf
    roi_detector.write_pdf_summary_page(c, "ROI TEST: %s" % geometric_image.label)

# for each of the image datasets show the detected ROIs
for t1_vir_imageset in t1_vir_imagesets:
    t1_vir_imageset.update_ROI_mask()  # trigger a mask update
    t1_vir_imageset.write_roi_pdf_page(c)
for t1_vfa_imageset in t1_vfa_imagesets:
    t1_vfa_imageset.update_ROI_mask()  # trigger a mask update
    t1_vfa_imageset.write_roi_pdf_page(c)
for t2_mse_imageset in t2_mse_imagesets:
    t2_mse_imageset.update_ROI_mask()  # trigger a mask update
    t2_mse_imageset.write_roi_pdf_page(c)
for t2star_imageset in t2star_imagesets:
    t2star_imageset.update_ROI_mask()  # trigger a mask update
    t2star_imageset.write_roi_pdf_page(c)
for dw_imageset in dw_imagesets:
    dw_imageset.update_ROI_mask()  # trigger a mask update
    dw_imageset.write_roi_pdf_page(c)



print("------------- FIN ------------------")
c.save()
