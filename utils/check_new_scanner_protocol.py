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
07-July-2023  :               (James Korte) : Initial code to help users test their new scanner protocols
"""
import os
# Code to add the parent directory to allow importing mrbias core modules
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
base_dir = root
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
import mrbias.scan_session as ss
from mrbias import misc_utils as mu
from mrbias.misc_utils import LogLevels

# for pdf output
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas


#################################################################################
# INSTRUCTIONS
#################################################################################
#
# Set the following variables:
# -----------------------------------------------------------------------------
# - DICOM_DIRECTORY:     the directory of dicom images you want to search
# - OUTPUT_LOG_FILENAME: the filename for the output log
# - OUTPUT_PDF_FILENAME: the filename for the output pdf
#
# Create your new ScanSession<YourClassName> class for testing:
# --------------------------------------------------------------------------------
# i.e. scan_session = ss.ScanSessionYourClassName(DICOM_DIRECTORY)
#
# Check the results:
# -------------------------------------------------------------------------------
#   The script will log information to the terminal while it processes the data
# you can read this to check things are working correctly. In particular you
# can check that the imageset objects which are created match you expectation
# (i.e. there are 5 VFA images created if you expect five from the dataset)
#   There is also a pdf generated ("scan_session_test.pdf") which you can
# visually inspect to check the images have been correctly identified
#
#################################################################################
DICOM_DIRECTORY = os.path.join(base_dir, "data", "mrbias_testset_B")
OUTPUT_LOG_FILENAME = "scan_session_test.log"
OUTPUT_PDF_FILENAME = "scan_session_test.pdf"
scan_session = ss.SystemSessionSiemensSkyra(DICOM_DIRECTORY)
#################################################################################


# setup the logger to write to file
mu.initialise_logger(OUTPUT_LOG_FILENAME, force_overwrite=True, write_to_screen=True)
# setup a pdf to test the pdf reporting
pdf = mu.PDFSettings()
c = canvas.Canvas(OUTPUT_PDF_FILENAME, landscape(pdf.page_size))

mu.log("="*100, LogLevels.LOG_INFO)
mu.log("SCANNING DICOM DIR: %s" % DICOM_DIRECTORY, LogLevels.LOG_INFO)
mu.log("="*100, LogLevels.LOG_INFO)
# parse the DICOM directory and filter image sets
scan_session.write_pdf_summary_page(c)
# save the pdf report
c.save()

geometric_images = scan_session.get_geometric_images()
for geom_image in geometric_images:
    mu.log("Found GEO: %s" % type(geom_image), LogLevels.LOG_INFO)
    mu.log("\t\t%s" % str(geom_image), LogLevels.LOG_INFO)

pd_images = scan_session.get_proton_density_images()
for pd_image in pd_images:
    mu.log("Found PD: %s" % type(pd_image), LogLevels.LOG_INFO)
    mu.log("\t\t%s" % str(pd_image), LogLevels.LOG_INFO)

t1_vir_imagesets = scan_session.get_t1_vir_image_sets()
for t1_vir_imageset in t1_vir_imagesets:
    mu.log("Found T1(VIR): %s" % type(t1_vir_imageset), LogLevels.LOG_INFO)
    mu.log("\t\t%s" % str(t1_vir_imageset), LogLevels.LOG_INFO)

t1_vfa_imagesets = scan_session.get_t1_vfa_image_sets()
for t1_vfa_imageset in t1_vfa_imagesets:
    mu.log("Found T1(VFA): %s" % type(t1_vfa_imageset), LogLevels.LOG_INFO)
    mu.log("\t\t%s" % str(t1_vfa_imageset), LogLevels.LOG_INFO)

t2_mse_imagesets = scan_session.get_t2_mse_image_sets()
for t2_mse_imageset in t2_mse_imagesets:
    mu.log("Found T2(MSE): %s" % type(t2_mse_imageset), LogLevels.LOG_INFO)
    mu.log("\t\t%s" % str(t2_mse_imageset), LogLevels.LOG_INFO)

t2star_imagesets = scan_session.get_t2star_image_sets()
for t2star_imageset in t2star_imagesets:
    mu.log("Found T2*: %s" % type(t2star_imageset), LogLevels.LOG_INFO)
    mu.log("\t\t%s" % str(t2star_imageset), LogLevels.LOG_INFO)

# give a visual break in the log
mu.log("", LogLevels.LOG_INFO)






