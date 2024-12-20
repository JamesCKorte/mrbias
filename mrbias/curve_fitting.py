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
"""

import os
import copy
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as spopt
import lmfit
from scipy.stats import linregress
import seaborn as sns

# for pdf output
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Code to handle running each module as a test case (from within the module)
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
import mrbias.scan_session as scan_session
import mrbias.roi_detect as roi_detect
from mrbias.roi_detection_methods.detection_methods import DetectionOptions
import mrbias.image_sets as imset
import mrbias.phantom_reference as phantom
import mrbias.misc_utils as mu
from mrbias.misc_utils import LogLevels, PhantomOptions

from mrbias.curve_fit_models.curve_fit_abstract import NormalisationOptions, AveragingOptions, ExclusionOptions, OptiOptions
from mrbias.curve_fit_models.curve_fit_abstract import NORM_SETTING_STR_ENUM_MAP, AV_SETTING_STR_ENUM_MAP, EXCL_SETTING_STR_ENUM_MAP, OPTI_SETTING_STR_ENUM_MAP

from mrbias.curve_fit_models.t1_vir_2param import T1VIRCurveFitAbstract2Param
from mrbias.curve_fit_models.t1_vir_3param import T1VIRCurveFitAbstract3Param
from mrbias.curve_fit_models.t1_vir_4param import T1VIRCurveFitAbstract4Param
from mrbias.curve_fit_models.t1_vfa_2param import T1VFACurveFitAbstract2Param
from mrbias.curve_fit_models.t2_se_3param import T2SECurveFitAbstract3Param
from mrbias.curve_fit_models.dw_2param import DWCurveFitAbstract2Param



def main():
    run_phantom(PhantomOptions.RELAX_SYSTEM) # run a system phantom
    run_phantom(PhantomOptions.DIFFUSION_NIST)  # run a diffusion phantom

def run_phantom(phan_option):
    # setup output filenames
    file_prefix = "curve_fit"
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        file_prefix = "curve_fit_sys"
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        file_prefix = "curve_fit_diff"

    # setup the logger to write to file
    mu.initialise_logger("%s.log" % file_prefix, force_overwrite=True, write_to_screen=True)
    # setup a pdf to test the pdf reporting
    pdf = mu.PDFSettings()
    c = canvas.Canvas("%s.pdf" % file_prefix, landscape(pdf.page_size))

    # target images to test
    ss = None
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        dcm_dir_a = os.path.join(mu.reference_data_directory(), "mrbias_testset_A")
        ss = scan_session.SystemSessionSiemensSkyra(dcm_dir_a)
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        dcm_dir_a = os.path.join(mu.reference_data_directory(), "mrbias_testset_C")
        ss = scan_session.DiffusionSessionSiemensSkyra(dcm_dir_a)

    test_geometric_images = ss.get_geometric_images()
    test_geo_vec = [test_geometric_images[0]]
    case_name_vec = ["test_image_0"]

    if phan_option == PhantomOptions.RELAX_SYSTEM:
        # get the T1 and T2 imagesets
        t1_vir_imagesets = ss.get_t1_vir_image_sets()
        t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
        t2_mse_imagesets = ss.get_t2_mse_image_sets()
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        # get the diffusion weighted imagesets
        dw_imagesets = ss.get_dw_image_sets()

    # do full preparation with standard pipeline...
    # -----------------------------------------------------
    roi_template_dir = None
    detection_method = None
    ref_phan, init_phan = None, None
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        # Reference phantom and curve fit initialisation phantom
        ref_phan = phantom.ReferencePhantomCalibreSystem2(field_strength=3.0,  # Tesla
                                                          temperature=20.0,
                                                          serial_number="130-0093")  # Celsius
        init_phan = phantom.ReferencePhantomCalibreSystemFitInit(field_strength=3.0,  # Tesla
                                                                 temperature=20.0)  # Celsius
        # detection method
        roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_skyra_3p0T")
        detection_method = DetectionOptions.TWOSTAGE_MSEGS_CORELGD
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        # Reference phantom and curve fit initialisation phantom
        ref_phan = phantom.ReferencePhantomDiffusion1(field_strength=3.0,  # Tesla
                                                      temperature=0.0)  # Celsius
        init_phan = phantom.ReferencePhantomDiffusionFitInit(field_strength=3.0,  # Tesla
                                                             temperature=0.0)  # Celsius
        # detection method
        roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_diffusion_no_ice")
        detection_method = DetectionOptions.SHAPE_DIFFUSION_NIST

    # if a valid scan protocol found load up relevant image sets
    if ss is not None:
        mu.log(" loading image sets ..." , LogLevels.LOG_INFO)
        geometric_images = ss.get_geometric_images()
        t1_vir_imagesets = ss.get_t1_vir_image_sets()
        t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
        t2_mse_imagesets = ss.get_t2_mse_image_sets()
        dw_imagesets = ss.get_dw_image_sets()
        ss.write_pdf_summary_page(c)
    # exclude any geometric images that are not reference in curve fit data
    mu.log(" searching for linked geometric images ...", LogLevels.LOG_INFO)
    geometric_images_linked = set()
    for geometric_image in geometric_images:
        for fit_imagesets in [t1_vir_imagesets, t1_vfa_imagesets, t2_mse_imagesets, dw_imagesets]:
            for imageset in fit_imagesets:
                g = imageset.get_geometry_image()
                if g.get_label() == geometric_image.get_label():
                    geometric_images_linked.add(geometric_image)
                    mu.log("found geometric image (%s) linked with with imageset (%s)" %
                           (geometric_image.get_label(), imageset.get_set_label()),
                           LogLevels.LOG_INFO)
                    mu.log("found geometric image (%s) linked with with imageset.geoimage (%s)" %
                           (repr(geometric_image), repr(imageset.get_geometry_image())),
                           LogLevels.LOG_INFO)


    # detect the ROIs for curve fitting
    for geom_image in geometric_images_linked:
        roi_detector = roi_detect.ROIDetector(geom_image,
                                              roi_template_dir,
                                              detection_method=detection_method,
                                              partial_fov=False)
        # detect the ROIs and store the masks on the target image
        roi_detector.detect()
        # add a summary page to the PDF
        roi_detector.write_pdf_summary_page(c)


    preproc_dict_a = {'normalise': NormalisationOptions.VOXEL_MAX,
                      'average': AveragingOptions.AVERAGE_ROI,
                      'exclude': ExclusionOptions.CLIPPED_VALUES}
    preproc_dict_b = {'normalise': NormalisationOptions.VOXEL_MAX,
                      'exclude': ExclusionOptions.CLIPPED_VALUES}
    # test all the models
    for preproc_dict in [preproc_dict_a, preproc_dict_b]:
        # T1 VFA check
        if len(t1_vfa_imagesets):
            t1_vfa_imageset = t1_vfa_imagesets[0]
            t1_vfa_imageset.update_ROI_mask()  # trigger a mask update
            # 2 parameter model:
            test_t1_vfa_curvefit2p = T1VFACurveFitAbstract2Param(imageset=t1_vfa_imageset,
                                                                reference_phantom=ref_phan,
                                                                initialisation_phantom=init_phan,
                                                                preprocessing_options=preproc_dict,
                                                                angle_exclusion_list=[15.0],
                                                                exclusion_label="no15deg")

        # T1 VIR check
        if len(t1_vir_imagesets):
            t1_vir_imageset = t1_vir_imagesets[0]
            t1_vir_imageset.update_ROI_mask() # trigger a mask update
            # 2 parameter model:
            test_t1_curvefit2p = T1VIRCurveFitAbstract2Param(imageset=t1_vir_imageset,
                                                            reference_phantom=ref_phan,
                                                            initialisation_phantom=init_phan,
                                                            preprocessing_options=preproc_dict,
                                                            inversion_exclusion_list=[1500],
                                                            exclusion_label="no1500ms")
            # 3 parameter model:
            test_t1_curvefit3p = T1VIRCurveFitAbstract3Param(imageset=t1_vir_imageset,
                                                            reference_phantom=ref_phan,
                                                            initialisation_phantom=init_phan,
                                                            preprocessing_options=preproc_dict)
            # 4 parameter model:
            test_t1_curvefit4p = T1VIRCurveFitAbstract4Param(imageset=t1_vir_imageset,
                                                            reference_phantom=ref_phan,
                                                            initialisation_phantom=init_phan,
                                                            preprocessing_options=preproc_dict)

        # T2 MSE check
        if len(t2_mse_imagesets):
            t2_mse_imageset = t2_mse_imagesets[0]
            t2_mse_imageset.update_ROI_mask() # trigger a mask update
            test_t2_curvefit3p = T2SECurveFitAbstract3Param(imageset=t2_mse_imageset,
                                                            reference_phantom=ref_phan,
                                                            initialisation_phantom=init_phan,
                                                            preprocessing_options=preproc_dict,
                                                            echo_exclusion_list=[10, 20.],
                                                            exclusion_label="less20deg")
        
        if len(dw_imagesets):
            dw_imageset = dw_imagesets[0]
            dw_imageset.update_ROI_mask() # trigger a mask update
            test_dw_curvefit2p = DWCurveFitAbstract2Param(imageset=dw_imageset,
                                                          reference_phantom=ref_phan,
                                                          initialisation_phantom=init_phan,
                                                          preprocessing_options=preproc_dict)

        # add summary pages to PDF
        if len(t1_vfa_imagesets):
            for model in [test_t1_vfa_curvefit2p]:
                # create an output directory
                model_out_dir = model.get_imset_model_preproc_name()
                if not os.path.isdir(model_out_dir):
                    os.mkdir(model_out_dir)
                model.write_data(model_out_dir, write_voxel_data=True)
                model.write_pdf_summary_pages(c, is_system=True)
        if len(t1_vir_imagesets):
            for model in [test_t1_curvefit2p, test_t1_curvefit3p, test_t1_curvefit4p]:
                # create an output directory
                model_out_dir = model.get_imset_model_preproc_name()
                if not os.path.isdir(model_out_dir):
                    os.mkdir(model_out_dir)
                model.write_data(model_out_dir, write_voxel_data=True)
                model.write_pdf_summary_pages(c, is_system=True)
        if len(t2_mse_imagesets):
            for model in [test_t2_curvefit3p]:
                # create an output directory
                model_out_dir = model.get_imset_model_preproc_name()
                if not os.path.isdir(model_out_dir):
                    os.mkdir(model_out_dir)
                model.write_data(model_out_dir, write_voxel_data=True)
                model.write_pdf_summary_pages(c, is_system=True)
        if len(dw_imagesets):
            for model in [test_dw_curvefit2p]:
                # create an output directory
                model_out_dir = model.get_imset_model_preproc_name()
                if not os.path.isdir(model_out_dir):
                    os.mkdir(model_out_dir)
                model.write_data(model_out_dir, write_voxel_data=True)
                model.write_pdf_summary_pages(c, is_system=False)


    # save the pdf report
    c.save()
    mu.log("------ FIN -------", LogLevels.LOG_INFO)





if __name__ == '__main__':
    main()
