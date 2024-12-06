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

from abc import ABC, abstractmethod

import os
from collections import OrderedDict
from enum import IntEnum

import pydicom as dcm

import pandas as pd
pd.options.mode.chained_assignment = 'raise' # DEBUG: for finding SettingWithCopyWarning
import SimpleITK as sitk
import numpy as np

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

# System phantom sessions
from mrbias.scan_sessions.sys_philips_1p5T_marlin import SystemSessionPhilipsMarlin
from mrbias.scan_sessions.sys_philips_1p5T_marlin_AVL import SystemSessionAVLPhilipsMarlinNoGeo
from mrbias.scan_sessions.sys_philips_1p5T_ingeniaX import SystemSessionPhilipsIngeniaAmbitionX
from mrbias.scan_sessions.sys_siemens_3T_skyra import SystemSessionSiemensSkyra
from mrbias.scan_sessions.sys_siemens_3T_skyra_erin import SystemSessionSiemensSkyraErin
from mrbias.scan_sessions.sys_siemens_auckland_CAM import SystemSessionAucklandCAM

# Diffusion phantom sessions
from mrbias.scan_sessions.diff_GE_1p5T_optima import DiffusionSessionGEOptima
from mrbias.scan_sessions.diff_GE_3T_discovery import DiffusionSessionGEDiscovery
from mrbias.scan_sessions.diff_philips_1p5T_ingeniaX import DiffusionSessionPhilipsIngeniaAmbitionX
from mrbias.scan_sessions.diff_philips_3T_ingenia import DiffusionSessionPhilipsIngenia
from mrbias.scan_sessions.diff_siemens_3T_skyra import DiffusionSessionSiemensSkyra


# for pdf output
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas


def main():

    # -------------------------------
    # Basic test
    # ------------------------------
    # Run the ScanSessionSiemensSkyra class on two test dicom directories
    # Visually inspect output to ensure all types of images are located
    # todo: change to function with a boolean pass criteria

    # Skyra test
    dcm_dir_a = os.path.join(mu.reference_data_directory(), "mrbias_testset_A")
    dcm_dir_b = os.path.join(mu.reference_data_directory(), "mrbias_testset_B")
    test_dcm_dir_list = [dcm_dir_a, dcm_dir_b]
    test_ss_config_list = ["SiemensSkyra", "SiemensSkyra"]

    # Skyra DW test
    # test_ss_config_list = []
    # test_dcm_dir_list = []
    test_dcm_dir_list.append(os.path.join(mu.reference_data_directory(), "mrbias_testset_C"))
    test_ss_config_list.append("DiffSiemensSkyra")


    # setup the logger to write to file
    mu.initialise_logger("scan_session.log", force_overwrite=True, write_to_screen=True)
    # setup a pdf to test the pdf reporting
    pdf = mu.PDFSettings()
    c = canvas.Canvas("scan_session.pdf", landscape(pdf.page_size))


    for dcm_dir, ss_type in zip(test_dcm_dir_list, test_ss_config_list):
        mu.log("="*100, LogLevels.LOG_INFO)
        mu.log("SCANNING DICOM DIR: %s" % dcm_dir, LogLevels.LOG_INFO)
        mu.log("="*100, LogLevels.LOG_INFO)
        # parse the DICOM directory and filter image sets
        if ss_type == "PhilipsMarlin":
            scan_session = SystemSessionPhilipsMarlin(dcm_dir)
        if ss_type == "SiemensSkyra":
            scan_session = SystemSessionSiemensSkyra(dcm_dir)
        elif ss_type == "PhilipsIngeniaAmbitionX":
            scan_session = SystemSessionPhilipsIngeniaAmbitionX(dcm_dir)
        elif ss_type == "DiffPhilipsIngeniaAmbitionX":
            scan_session = DiffusionSessionPhilipsIngeniaAmbitionX(dcm_dir)
        elif ss_type == "DiffSiemensSkyra":
            scan_session = DiffusionSessionSiemensSkyra(dcm_dir)
        scan_session.write_pdf_summary_page(c)


        # geometric_images = scan_session.get_geometric_images()
        # for geom_image in geometric_images:
        #     mu.log("Found GEO: %s" % type(geom_image), LogLevels.LOG_INFO)
        #     mu.log("\t\t%s" % str(geom_image), LogLevels.LOG_INFO)
        #
        # pd_images = scan_session.get_proton_density_images()
        # for pd_image in pd_images:
        #     mu.log("Found PD: %s" % type(pd_image), LogLevels.LOG_INFO)
        #     mu.log("\t\t%s" % str(pd_image), LogLevels.LOG_INFO)
        #
        # t1_vir_imagesets = scan_session.get_t1_vir_image_sets()
        # for t1_vir_imageset in t1_vir_imagesets:
        #     mu.log("Found T1(VIR): %s" % type(t1_vir_imageset), LogLevels.LOG_INFO)
        #     mu.log("\t\t%s" % str(t1_vir_imageset), LogLevels.LOG_INFO)
        #
        # t1_vfa_imagesets = scan_session.get_t1_vfa_image_sets()
        # for t1_vfa_imageset in t1_vfa_imagesets:
        #     mu.log("Found T1(VFA): %s" % type(t1_vfa_imageset), LogLevels.LOG_INFO)
        #     mu.log("\t\t%s" % str(t1_vfa_imageset), LogLevels.LOG_INFO)
        #
        # t2_mse_imagesets = scan_session.get_t2_mse_image_sets()
        # for t2_mse_imageset in t2_mse_imagesets:
        #     mu.log("Found T2(MSE): %s" % type(t2_mse_imageset), LogLevels.LOG_INFO)
        #     mu.log("\t\t%s" % str(t2_mse_imageset), LogLevels.LOG_INFO)
        #
        # dw_imagesets = scan_session.get_dw_image_sets()
        # for dw_imageset in dw_imagesets:
        #     mu.log("Found DW: %s" % type(dw_imageset), LogLevels.LOG_INFO)
        #     mu.log("\t\t%s" % str(dw_imageset), LogLevels.LOG_INFO)
        # # give a visual break in the log
        # mu.log("", LogLevels.LOG_INFO)

    # save the pdf report
    c.save()
    mu.log("------ FIN -------", LogLevels.LOG_INFO)


if __name__ == "__main__":
    main()