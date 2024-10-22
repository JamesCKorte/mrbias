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
21-September-2024  :               (James Korte) : Refactoring   MR-BIAS v1.0
"""

import os
import copy
from enum import IntEnum
from collections import OrderedDict
from abc import ABC, abstractmethod

import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from mrbias import misc_utils as mu
from mrbias.misc_utils import LogLevels, OrientationOptions, PhantomOptions
import mrbias.scan_session as scan_session
import mrbias.image_sets as image_sets
from mrbias.roi_detection_methods.roi_template import ROITemplate
from mrbias.roi_detection_methods.detection_methods import DetectionOptions
from mrbias.roi_detection_methods.register_none import RegistrationNone
from mrbias.roi_detection_methods.register_correl_gd import RegistrationCorrelationGradientDescent
from mrbias.roi_detection_methods.register_mmi_gd import RegistrationMutualInformationGradientDescent
from mrbias.roi_detection_methods.register_mse_gridsearch import RegistrationMSEGridSearch
from mrbias.roi_detection_methods.register_two_stage import RegistrationTwoStage
from mrbias.roi_detection_methods.register_axi_gridsearch_correl_gd import RegistrationAxialRotGridThenNBestGradDecnt
from mrbias.roi_detection_methods.shape_diffusion_nist import ShapeDiffusionNIST


def main():
    #run_phantom(PhantomOptions.RELAX_SYSTEM) # run a system phantom
    run_phantom(PhantomOptions.DIFFUSION_NIST) # run a diffusion phantom

def run_phantom(phan_option):
    # setup output filenames
    file_prefix = "roi_detect"
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        file_prefix = "roi_detect_sys"
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        file_prefix = "roi_detect_diff"
    # setup the logger to write to file
    mu.initialise_logger("%s.log" % file_prefix, force_overwrite=True, write_to_screen=True)
    # setup a pdf to test the pdf reporting
    pdf = mu.PDFSettings()
    c = canvas.Canvas("%s.pdf" % file_prefix, landscape(pdf.page_size))

    # target images to test
    ss = None
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        dcm_dir_a = os.path.join(mu.reference_data_directory(), "mrbias_testset_B")
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

    # set the test template
    test_template_dir = None
    detection_method = None
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        test_template_dir = os.path.join(mu.reference_template_directory(), "siemens_skyra_3p0T")
        detection_method = DetectionOptions.TWOSTAGE_MSEGS_CORELGD
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        test_template_dir = os.path.join(mu.reference_template_directory(), "siemens_diffusion_no_ice")
        detection_method = DetectionOptions.SHAPE_DIFFUSION_NIST

    for test_target_im, case_str in zip(test_geo_vec, case_name_vec):
        # create a roi detector
        roi_detector = ROIDetector(test_target_im, test_template_dir,
                                   detection_method=detection_method,
                                   partial_fov=False)
        # detect the ROIs and return the masks on the target image
        roi_detector.detect()

        # output to pdf
        figure_title = "ROI DETECTOR : %s" % case_str
        roi_detector.write_pdf_summary_page(c, figure_title)

    # for each of the image datasets show the detected ROIs
    if phan_option == PhantomOptions.RELAX_SYSTEM:
        for t1_vir_imageset in t1_vir_imagesets:
            t1_vir_imageset.update_ROI_mask()  # trigger a mask update
            t1_vir_imageset.write_roi_pdf_page(c, include_pmap_pages=True)
        for t1_vfa_imageset in t1_vfa_imagesets:
            t1_vfa_imageset.update_ROI_mask()  # trigger a mask update
            t1_vfa_imageset.write_roi_pdf_page(c, include_pmap_pages=True)
        for t2_mse_imageset in t2_mse_imagesets:
            t2_mse_imageset.update_ROI_mask()  # trigger a mask update
            t2_mse_imageset.write_roi_pdf_page(c, include_pmap_pages=True)
    elif phan_option == PhantomOptions.DIFFUSION_NIST:
        for dw_imageset in dw_imagesets:
            dw_imageset.update_ROI_mask()  # trigger a mask update
            dw_imageset.write_roi_pdf_page(c, include_pmap_pages=True)

    # save the pdf report
    c.save()
    mu.log("------ FIN -------", LogLevels.LOG_INFO)
    plt.show()


class ROIDetector(object):
    """
    A class for detecting regions of interest (ROIs) on a 3D T1 weighted image of the ISMRM/NIST qMRI System Phantom.

    The target image is registered to a template image which has associated template ROIs. The ROIs are then warped
    from the template image space onto the target image space.

    Attributes:
        target_geo_im (ImageSet.ImageGeometric): the target image on which we want to detect ROIs
        detect_method (DetectionOptions): the detection method to link the target and template images
        fixed_geom_im (SimpleITK.Image): the template T1 weighted 3D MRI image
        detector (DetectionMethodAbstract): the detection method to link the target and template images
    """

    def __init__(self,
                 target_geometry_image,
                 template_directory,
                 detection_method=DetectionOptions.TWOSTAGE_MSEGS_CORELGD,
                 partial_fov=False,
                 kwargs=None):
        """
        Class constructor stores the target image, registration method choice and loads the template image.

        Args:
            target_geometry_image (ImageSet.ImageGeometric): the target image on which we want to detect ROIs
            template_directory (directory): a template directory with a DICOM folder and ROI yaml files
            registration_method (RegistrationOptions): a flag representing the registration method to use
        """
        assert isinstance(target_geometry_image, image_sets.ImageGeometric), \
            "ROIDetector::init(): target_geometry_image is expected as a image_sets.ImageGeometric (not %s)" % \
            type(target_geometry_image)
        self.target_geo_im = target_geometry_image
        self.detect_method = detection_method
        self.partial_fov = partial_fov
        self.roi_template = ROITemplate(template_directory)
        self.fixed_geom_im = self.roi_template.get_image()
        assert isinstance(self.fixed_geom_im, sitk.Image), \
            "ROIDetector::init(): self.fixed_geom_im (template image) is expected as a SimpleITK image (not %s)" % \
            type(self.fixed_geom_im)
        self.kwargs = kwargs
        self.detector = None

    @staticmethod
    def generate_detection_instance(reg_method, fixed_geom_im, target_geo_im, roi_template, partial_fov=False, kwargs=None):
        rego = None
        centre_images = not partial_fov
        if reg_method == DetectionOptions.NONE:
            rego = RegistrationNone(target_geo_im, fixed_geom_im, roi_template)
        elif reg_method == DetectionOptions.COREL_GRADIENTDESCENT:
            rego = RegistrationCorrelationGradientDescent(target_geo_im, fixed_geom_im, roi_template,
                                                          centre_images=centre_images)
        elif reg_method == DetectionOptions.MSE_GRIDSEARCH:
            rego = RegistrationMSEGridSearch(target_geo_im, fixed_geom_im, roi_template)
        elif reg_method == DetectionOptions.TWOSTAGE_MSEGS_CORELGD:
            rego = RegistrationTwoStage(target_geo_im, fixed_geom_im, roi_template,
                                        DetectionOptions.MSE_GRIDSEARCH,
                                        DetectionOptions.COREL_GRADIENTDESCENT)
        elif reg_method == DetectionOptions.MMI_GRADIENTDESCENT:
            rego = RegistrationMutualInformationGradientDescent(target_geo_im, fixed_geom_im, roi_template,
                                                                centre_images=centre_images)
        elif reg_method == DetectionOptions.COREL_AXIGRID_NBEST_GRADIENTDESCENT:
            rego = RegistrationAxialRotGridThenNBestGradDecnt(target_geo_im, fixed_geom_im, roi_template,
                                                                centre_images=centre_images)
        elif reg_method == DetectionOptions.SHAPE_DIFFUSION_NIST:
            flip_cap_dir = False
            debug_vis = False
            if kwargs is not None:
                if 'flip_cap_dir' in kwargs.keys():
                    flip_cap_dir = kwargs['flip_cap_dir']
                if 'debug_vis' in kwargs.keys():
                    debug_vis = kwargs['debug_vis']
            rego = ShapeDiffusionNIST(target_geo_im, fixed_geom_im, roi_template,
                                      flip_cap_dir=flip_cap_dir,
                                      debug_vis=debug_vis)
        assert rego is not None, "ROIDetector::generate_detection_instance(): " \
                                 "invalid detection method, %s" % str(reg_method)
        return rego

    def detect(self):
        """ Detect the ROIs on the target image, by registering the target image to template image."""
        self.detector = ROIDetector.generate_detection_instance(self.detect_method,
                                                                self.fixed_geom_im,
                                                                self.target_geo_im,
                                                                self.roi_template,
                                                                self.partial_fov,
                                                                self.kwargs)
        mu.log("ROIDetector::detect():",
               LogLevels.LOG_INFO)
        self.detector.detect()

        # store the resulting masks on the geometric image
        t1_mask, t1_prop_dict = self.detector.get_detected_T1_mask()
        self.target_geo_im.set_T1_mask(t1_mask, t1_prop_dict)
        t2_mask, t2_prop_dict = self.detector.get_detected_T2_mask()
        self.target_geo_im.set_T2_mask(t2_mask, t2_prop_dict)
        dw_mask, dw_prop_dict = self.detector.get_detected_DW_mask()
        self.target_geo_im.set_DW_mask(dw_mask, dw_prop_dict)

    def copy_detector(self, d):
        assert issubclass(type(d), ROIDetector), \
            "ROIDetector::copy_registration(d): d (the ROI detector being copied from) is expected as type ROIDetector (not %s)" % \
            type(d)
        # copy the detection information from the passed ROI detector (shallow)
        #self.transform = copy.deepcopy(d.transform) # cant deepcopy a pyswig object
        self.detector = d.detector # TODO: review this hack

        # store the resulting masks on the geometric image
        t1_mask, t1_prop_dict = self.detector.get_detected_T1_mask()
        self.target_geo_im.set_T1_mask(t1_mask, t1_prop_dict)
        t2_mask, t2_prop_dict = self.detector.get_detected_T2_mask()
        self.target_geo_im.set_T2_mask(t2_mask, t2_prop_dict)
        dw_mask, dw_prop_dict = self.detector.get_detected_DW_mask()
        self.target_geo_im.set_DW_mask(dw_mask, dw_prop_dict)

    def write_pdf_summary_page(self, c, sup_title=None):
        self.detector.write_pdf_summary_page(c, sup_title=None)




if __name__ == "__main__":
    main()
