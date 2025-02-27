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

import tempfile
import shutil

from abc import ABC, abstractmethod

import SimpleITK as sitk
import yaml
from collections import OrderedDict
import numbers
import pandas as pd

from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas

# Code to handle running each module as a test case (from within the module)
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
import mrbias.misc_utils as mu
from mrbias.misc_utils import LogLevels
import mrbias.scan_session as scan_session
import mrbias.curve_fitting as curve_fit
import mrbias.roi_detect as roi_detect
import mrbias.phantom_reference as phantom


def main():

    # specify the configuration file to control the analysis
    test_configuration_file = os.path.join(os.getcwd(), "..", "config", "example_AVL_Philips_config_allvials_nogeo.yaml")
    # specific the dicom directories to analyse
    test_dicom_directory_a = os.path.join(os.getcwd(), "..", "Newdata", "M231213A")

    # create a MRBIAS analysis object
    mrb = MRBIAS(test_configuration_file, write_to_screen=True)
    # run the analysis (output will be created in the "output_directory" specified in the configuration file)
    mrb.analyse(test_dicom_directory_a)

    mu.log("------ FIN -------", LogLevels.LOG_INFO)


class MRBIAS(object):
    def __init__(self, config_filename, write_to_screen=True):
        mu.log("MR-BIAS::__init__(): parsing config file : %s" % config_filename, LogLevels.LOG_INFO)
        self.config_filename = config_filename
        self.global_config     = MRIBiasGlobalConfig(config_filename)
        self.experiment_config = MRIBiasExperimentConfig(config_filename)
        self.sorting_config    = MRIBiasDICOMSortConfig(config_filename)
        self.detect_config     = MRIBiasROIDetectConfig(config_filename)
        self.cf_config         = MRIBiasCurveFitConfig(config_filename)
        self.write_to_screen = write_to_screen
        # setup (global) class configuration from file
        self.output_dir = self.global_config.get_output_directory()
        self.overwrite_existing_output = self.global_config.get_overwrite_existing_output()


    def analyse(self, dicom_directory):
        assert os.path.isdir(dicom_directory), "MR-BIAS::analyse(): invalid dicom_directory: %s" % dicom_directory
        mu.log("="*100, LogLevels.LOG_INFO)
        mu.log("MR-BIAS::analyse()", LogLevels.LOG_INFO)
        mu.log("="*100, LogLevels.LOG_INFO)

        # ===================================================================================================
        # Setup the output directory
        # ===================================================================================================
        mu.log("MR-BIAS::analyse():   scan DICOM directory for basic information", LogLevels.LOG_INFO)
        output_subdir_name = MRBIAS.get_directory_name_from_dcm_metadata(dicom_directory)
        mu.log("MR-BIAS::analyse():  setup the output base directory: %s" % self.output_dir,
               LogLevels.LOG_INFO)
        mu.log("MR-BIAS::analyse():                and sub-directory: %s" % output_subdir_name,
               LogLevels.LOG_INFO)
        out_dir = os.path.join(self.output_dir, output_subdir_name)
        dir_created = mu.safe_dir_create(self.output_dir, "MR-BIAS::analyse():")
        dir_created = mu.safe_dir_create(out_dir, "MR-BIAS::analyse():")
        # if it already exists, and overwrite is disabled then cancel the analysis
        if (not dir_created) and (not self.overwrite_existing_output):
            mu.log("MR-BIAS::analyse(): skipping analysis as output directory already exists: %s" % out_dir,
                   LogLevels.LOG_WARNING)
            return None
        # setup the log to file
        mu.initialise_logger(os.path.join(out_dir, "mri_bias.log"),
                             force_overwrite=True,
                             write_to_screen=self.write_to_screen)
        # setup the summary pdf
        pdf = mu.PDFSettings()
        c = canvas.Canvas(os.path.join(out_dir, "mri_bias.pdf"),
                          landscape(pdf.page_size))
        self.write_pdf_title_page(c)


        # ===================================================================================================
        # Scan and sort the DICOM directory
        # ===================================================================================================
        mu.log("-" * 100, LogLevels.LOG_INFO)
        mu.log("MR-BIAS::analyse() : Scan and sort the DICOM directory", LogLevels.LOG_INFO)
        mu.log("-" * 100, LogLevels.LOG_INFO)
        scan_protocol = self.sorting_config.get_scan_protocol_for_sorting()
        show_unknown_series = self.sorting_config.get_show_unknown_series()
        ss = None
        if scan_protocol == "siemens_skyra_3p0T":
            ss = scan_session.SystemSessionSiemensSkyra(dicom_directory,
                                                        display_unknown_series=show_unknown_series)
        elif scan_protocol == "philips_marlin_1p5T":
            ss = scan_session.SystemSessionPhilipsMarlin(dicom_directory,
                                                         display_unknown_series=show_unknown_series)
        elif scan_protocol == "auckland_cam_3p0T":
            ss = scan_session.SystemSessionAucklandCAM(dicom_directory,
                                                       display_unknown_series=show_unknown_series)
        elif scan_protocol == "siemens_skyra_erin_3p0T":
            ss = scan_session.SystemSessionSiemensSkyraErin(dicom_directory,
                                                            display_unknown_series=show_unknown_series)
        elif scan_protocol == "philips_ingenia_ambitionX":
            ss = scan_session.SystemSessionPhilipsIngeniaAmbitionX(dicom_directory,
                                                                   display_unknown_series=show_unknown_series)
        elif scan_protocol == "philips_marlin_1p5T_avl":
            ss = scan_session.SystemSessionAVLPhilipsMarlinNoGeo(dicom_directory,
                                                                 display_unknown_series=show_unknown_series)
        elif scan_protocol == "diff_philips_ingenia_ambitionX":
            ss = scan_session.DiffusionSessionPhilipsIngeniaAmbitionX(dicom_directory,
                                                                      display_unknown_series=show_unknown_series)
        elif scan_protocol == "diff_siemens_skyra":
            ss = scan_session.DiffusionSessionSiemensSkyra(dicom_directory,
                                                           display_unknown_series=show_unknown_series)
        elif scan_protocol == "diff_philips_ingenia":
            ss = scan_session.DiffusionSessionPhilipsIngenia(dicom_directory,
                                                             display_unknown_series=show_unknown_series)
        elif scan_protocol == "diff_ge_optima":
            ss = scan_session.DiffusionSessionGEOptima(dicom_directory,
                                                       display_unknown_series=show_unknown_series)
        elif scan_protocol == "diff_ge_discovery":
            ss = scan_session.DiffusionSessionGEDiscovery(dicom_directory,
                                                          display_unknown_series=show_unknown_series)
        else:
            mu.log("MR-BIAS::analyse(): skipping analysis as unknown 'scan_protocol' defined for DICOM sorting",
                   LogLevels.LOG_WARNING)
        # if a valid scan protocol found load up relevant image sets
        geometric_images = []
        pd_images = []
        t1_vir_imagesets = []
        t1_vfa_imagesets = []
        t2_mse_imagesets = []
        t2_star_imagesets = []
        dw_imagesets = []
        if ss is not None:
            geometric_images = ss.get_geometric_images()
            #pd_images = ss.get_proton_density_images()
            t1_vir_imagesets = ss.get_t1_vir_image_sets()
            t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
            t2_mse_imagesets = ss.get_t2_mse_image_sets()
            t2_star_imagesets = ss.get_t2star_image_sets()
            dw_imagesets = ss.get_dw_image_sets()
            ss.write_pdf_summary_page(c)
        # log some basic details of the imagesets
        for t1_vir_imageset in t1_vir_imagesets:
            mu.log("Found T1(VIR): %s" % type(t1_vir_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t1_vir_imageset), LogLevels.LOG_INFO)
        for t1_vfa_imageset in t1_vfa_imagesets:
            mu.log("Found T1(VFA): %s" % type(t1_vfa_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t1_vfa_imageset), LogLevels.LOG_INFO)
        for t2_mse_imageset in t2_mse_imagesets:
            mu.log("Found T2(MSE): %s" % type(t2_mse_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t2_mse_imageset), LogLevels.LOG_INFO)
        for t2_star_imageset in t2_star_imagesets:
            mu.log("Found T2(Star): %s" % type(t2_star_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t2_star_imageset), LogLevels.LOG_INFO)
        for dw_imageset in dw_imagesets:
            mu.log("Found DW: %s" % type(dw_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(dw_imageset), LogLevels.LOG_INFO)

        geometric_images_linked = OrderedDict() # using the ordered dictionary keys as an ordered set
        if ss is not None:
            # exclude any geometric images that are not reference in curve fit data
            mu.log("MR-BIAS::analyse(): Identify linked geometric images ...", LogLevels.LOG_INFO)
            for geometric_image in geometric_images:
                for fit_imagesets in [t1_vir_imagesets, t1_vfa_imagesets, t2_mse_imagesets, t2_star_imagesets, dw_imagesets]:
                    for imageset in fit_imagesets:
                        g = imageset.get_geometry_image()
                        if g.get_label() == geometric_image.get_label():
                            geometric_images_linked[geometric_image] = None
                            mu.log("\tfound geometric image (%s) linked with with imageset (%s)" %
                                   (geometric_image.get_label(), imageset.get_set_label()),
                                   LogLevels.LOG_INFO)
                            mu.log("\tfound geometric image (%s) linked with with imageset.geoimage (%s)" %
                                   (repr(geometric_image), repr(imageset.get_geometry_image())),
                                   LogLevels.LOG_INFO)

        # ===================================================================================================
        # Dectect the ROIs on each geometry image (only ones used in a fit)
        # ===================================================================================================
        mu.log("-" * 100, LogLevels.LOG_INFO)
        mu.log("MR-BIAS::analyse() : Detect the ROIs on each geometry image ...", LogLevels.LOG_INFO)
        mu.log("-" * 100, LogLevels.LOG_INFO)
        roi_template = self.detect_config.get_template()
        roi_detect_method = self.detect_config.get_detection_method()
        roi_shape_fine_tune = self.detect_config.get_shape_fine_tune()
        roi_flip_cap_series_numbers = self.detect_config.get_flip_cap_series_numbers()
        roi_is_partial_fov = self.detect_config.get_registration_partial_fov()
        roi_use_first_detection_for_all = self.detect_config.get_use_first_detection_only()
        roi_use_manual_roi = self.detect_config.get_use_manual_roi()
        roi_T1_manual_filepath = self.detect_config.get_manual_T1_roi_filepath()
        roi_T2_manual_filepath = self.detect_config.get_manual_T2_roi_filepath()
        roi_DW_manual_filepath = self.detect_config.get_manual_DW_roi_filepath()
        roi_template_dir, detect_method = None, None
        temporary_dir, roi_template_dicom_dir = None, None
        if roi_use_manual_roi:
            # make a temporary template directory
            roi_template = "manual"
            temporary_dir = os.path.join(os.getcwd(), "tmp")
            if not os.path.isdir(temporary_dir):
                os.mkdir(temporary_dir)
            roi_template_dir = temporary_dir
            # no registration required just place the ROI on the image
            detect_method = roi_detect.DetectionOptions.NONE
        else:
            if roi_template == "siemens_skyra_3p0T":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_skyra_3p0T")
            elif roi_template == "systemlite_siemens_vida_3p0T":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "systemlite_siemens_vida_3p0T")
            elif roi_template == "systemlite_siemens_vida_3p0T_180degrees":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "systemlite_siemens_vida_3p0T_180degrees")
            elif roi_template == "philips_ingenia_1p5T":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "philips_ingenia_1p5T")
            elif roi_template == "eurospin_philips_1p5T_allvials":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "eurospin_philips_1p5T_allvials")
            elif roi_template == "siemens_diffusion":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_diffusion")
            elif roi_template == "siemens_diffusion_no_ice":
                roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_diffusion_no_ice")
            # ... add others

            # determine which detection method
            if roi_detect_method == "none":
                detect_method = roi_detect.DetectionOptions.NONE
            elif roi_detect_method == "two_stage_msme-GS_correl-GD":
                detect_method = roi_detect.DetectionOptions.TWOSTAGE_MSEGS_CORELGD
            elif roi_detect_method == "mattesMI-GD":
                detect_method = roi_detect.DetectionOptions.MMI_GRADIENTDESCENT
            elif roi_detect_method == "correl-GD":
                detect_method = roi_detect.DetectionOptions.COREL_GRADIENTDESCENT
            elif roi_detect_method == "correl-axigrid_nbest-GD":
                detect_method = roi_detect.DetectionOptions.COREL_AXIGRID_NBEST_GRADIENTDESCENT
            elif roi_detect_method == "shape_diffusion_nist":
                detect_method = roi_detect.DetectionOptions.SHAPE_DIFFUSION_NIST


        # create the detector if appropriate information available
        if roi_template is None:
            mu.log("MR-BIAS::analyse(): skipping analysis as unknown 'roi_template' defined for ROI detection",
                   LogLevels.LOG_ERROR)
            return None
        elif not (len(geometric_images_linked.keys()) > 0):
            mu.log("MR-BIAS::analyse(): skipping analysis as no linked geometry imaged found for ROI detection",
                   LogLevels.LOG_ERROR)
            return None
        else:
            assert detect_method is not None, "MR-BIAS::analyse(): invalid ROI registration method selected - please check your configuration file"
            roi_detectors = OrderedDict()
            first_geo_im, first_detector  = None, None
            for g_dx, geom_image in enumerate(geometric_images_linked.keys()):
                if roi_use_manual_roi:
                    # make a copy of the geometric image and ROI definition file into the temporary directory
                    roi_template_dicom_dir = os.path.join(temporary_dir, "dicom")
                    assert ((roi_T1_manual_filepath is not None) and os.path.isfile(roi_T1_manual_filepath)) or \
                           ((roi_T2_manual_filepath is not None) and os.path.isfile(roi_T2_manual_filepath)) or \
                           ((roi_DW_manual_filepath is not None) and os.path.isfile(roi_DW_manual_filepath)), \
                        "MR-BIAS::analyse(): roi_manual_filepath not found (%s) - please check your configuration file" \
                        "\n\tT1: %s\n\tT2: %s\n\tDW: %s" % (roi_T1_manual_filepath, roi_T2_manual_filepath, roi_DW_manual_filepath)
                    file_list = []
                    # add the dicom files
                    for f in geom_image.get_filepath_list():
                        file_list.append(f)

                    if not os.path.isdir(roi_template_dicom_dir):
                        os.mkdir(roi_template_dicom_dir)
                    # Try to create symbolic links to the original files from our tmpdir
                    try:
                        if roi_T1_manual_filepath is not None:
                            os.symlink(os.path.abspath(roi_T1_manual_filepath), os.path.join(roi_template_dir, "default_T1_rois.yaml"))
                        if roi_T2_manual_filepath is not None:
                            os.symlink(os.path.abspath(roi_T2_manual_filepath), os.path.join(roi_template_dir, "default_T2_rois.yaml"))
                        if roi_DW_manual_filepath is not None:
                            os.symlink(os.path.abspath(roi_DW_manual_filepath), os.path.join(roi_template_dir, "default_DW_rois.yaml"))
                        for f in file_list:
                            os.symlink(os.path.abspath(f), os.path.join(roi_template_dicom_dir, os.path.basename(f)))
                    except:
                        # if it fails (permissions etc.)
                        # copy the original files to a tmpdir
                        if roi_T1_manual_filepath is not None:
                            shutil.copy(os.path.abspath(roi_T1_manual_filepath), os.path.join(roi_template_dir, "default_T1_rois.yaml"))
                        if roi_T2_manual_filepath is not None:
                            shutil.copy(os.path.abspath(roi_T2_manual_filepath), os.path.join(roi_template_dir, "default_T2_rois.yaml"))
                        if roi_DW_manual_filepath is not None:
                            shutil.copy(os.path.abspath(roi_DW_manual_filepath), os.path.join(roi_template_dir, "default_DW_rois.yaml"))
                        for f in file_list:
                            shutil.copy(os.path.abspath(f), os.path.join(roi_template_dicom_dir, os.path.basename(f)))

                # create a roi detector
                detect_kwargs = {'flip_cap_dir': False,
                                 'debug_vis': False, # TODO: add parameter in config file and link here
                                 'inner_ring_only': True, # TODO: add parameter in config file for diffusion ROI shape method
                                 'fine_tune_rois': roi_shape_fine_tune}
                if geom_image.series_number in roi_flip_cap_series_numbers:
                    detect_kwargs['flip_cap_dir'] = True
                roi_detector = roi_detect.ROIDetector(geom_image, roi_template_dir,
                                                      detection_method=detect_method,
                                                      partial_fov=roi_is_partial_fov,
                                                      kwargs=detect_kwargs)
                if g_dx == 0:
                    first_geo_im = geom_image
                    first_detector = roi_detector
                if roi_use_first_detection_for_all and (g_dx > 0):
                    mu.log("MR-BIAS::analyse() : ROI detection from %s being copied to %s ..." %
                           (first_geo_im.label, geom_image.label),
                           LogLevels.LOG_INFO)
                    roi_detector.copy_registration(first_detector)
                else:
                    # detect the ROIs and store the masks on the target image
                    roi_detector.detect()
                # add a summary page to the PDF
                roi_detector.write_pdf_summary_page(c)
                # store detector
                roi_detectors[geom_image.get_label()] = roi_detector

                # clear up temporary sub-directory
                if (roi_template_dicom_dir is not None) and os.path.isdir(roi_template_dicom_dir):
                    shutil.rmtree(roi_template_dicom_dir)
            # clear up temporary directory
            if (temporary_dir is not None) and os.path.isdir(temporary_dir):
                shutil.rmtree(temporary_dir)

        # ===================================================================================================
        # Fit parametric models to the raw voxel data
        # ===================================================================================================
        mu.log("-" * 100, LogLevels.LOG_INFO)
        mu.log("MR-BIAS::analyse() : fit parametric models to the detected ROIs on each imageset ...", LogLevels.LOG_INFO)
        mu.log("-" * 100, LogLevels.LOG_INFO)
        # --------------------------------------------------------------------
        # get phantom/experiment details from the configuration file
        # --------------------------------------------------------------------
        # phantom details
        phantom_maker = self.experiment_config.get_phantom_manufacturer()
        phantom_type = self.experiment_config.get_phantom_type()
        phantom_sn = self.experiment_config.get_phantom_serial_number()
        ph_model_num, ph_item_num = None, None
        if (phantom_maker == "caliber_mri"):
            try:
                ph_model_num, ph_item_num = phantom_sn.split("-")
            except ValueError:
                mu.log(
                    "MR-BIAS::analyse(): for CailberMRI phantoms the software expects a 'phantom_serial_number' in the configuration.yaml file "
                    "which has the format 'model-sn' for example '130-001' for a diffusion phantom (130) and the first one made (001)",
                    LogLevels.LOG_ERROR)
                return None
        if not ((phantom_maker == "caliber_mri") and (phantom_type =="system_phantom") and (ph_model_num == "130")) and \
            not ((phantom_maker == "caliber_mri") and (phantom_type =="diffusion_phantom") and (ph_model_num == "128")) and \
                not ((phantom_maker == "eurospin") and (phantom_type =="relaxometry")):
            mu.log("MR-BIAS::analyse(): only supports phantom [caliber_mri:system_phantom(130), caliber_mri:diffusion_phantom(128) and eurospin:testobject5] (not [%s:%s(%s)]) "
                   "skipping analysis... " % (phantom_maker, phan_config, ph_model_num), LogLevels.LOG_ERROR)
            return None
        
        #experiment details
        field_strength = self.experiment_config.get_field_strength_tesla()
        temperature_celsius = self.experiment_config.get_temperature_celsius()
        if phantom_type == "system_phantom":
            init_phan = phantom.ReferencePhantomCalibreSystemFitInit(field_strength=field_strength,  # Tesla
                                                                     temperature=temperature_celsius)  # Celsius
        if phantom_type == "relaxometry":
            init_phan = phantom.ReferencePhantomEurospinRelaxometryFitInit(field_strength=field_strength,  # Tesla
                                                                           temperature=temperature_celsius)  # Celsius
        if phantom_type == "diffusion_phantom":
            init_phan = phantom.ReferencePhantomDiffusionFitInit(field_strength=field_strength,  # Tesla
                                                                 temperature=temperature_celsius)  # Celsius
        mu.log("init_phan: type, value, etc...", LogLevels.LOG_ERROR)
        mu.log(type(init_phan), LogLevels.LOG_ERROR)
        mu.log(init_phan, LogLevels.LOG_ERROR)

        if phantom_maker == "caliber_mri":
            # select the reference system phantom based on phantom serial number
            if ph_model_num == "130":
                ph_item_num = int(ph_item_num)
                ref_phan = None
                if ph_item_num < 42:
                    ref_phan = phantom.ReferencePhantomCalibreSystem1(field_strength=field_strength,  # Tesla
                                                                    temperature=temperature_celsius,
                                                                    serial_number=phantom_sn)  # Celsius
                elif ph_item_num < 133:
                    ref_phan = phantom.ReferencePhantomCalibreSystem2(field_strength=field_strength,  # Tesla
                                                                    temperature=temperature_celsius,
                                                                    serial_number=phantom_sn)  # Celsius
                else: # ph_item_num >= 133
                    ref_phan = phantom.ReferencePhantomCalibreSystem2p5(field_strength=field_strength,  # Tesla
                                                                        temperature=temperature_celsius,
                                                                        serial_number=phantom_sn)  # Celsius
            # select the reference diffusion phantom
            elif ph_model_num == "128":
                ref_phan = phantom.ReferencePhantomDiffusion1(field_strength=field_strength,  # Tesla
                                                              temperature=temperature_celsius,
                                                              serial_number=phantom_sn)  # Celsius
        elif phantom_maker == "eurospin":
            if phantom_type == "relaxometry":
                ref_phan = phantom.ReferencePhantomEurospinRelaxometry1(field_strength=field_strength,  # Tesla
                                                                        temperature=temperature_celsius,
                                                                        serial_number=phantom_sn)  # Celsius

        # --------------------------------------------------------------------
        # get curve fitting details from the configuration file
        # --------------------------------------------------------------------
        include_roi_pmap_pages = self.cf_config.get_include_roi_pmaps()
        cf_normal = self.cf_config.get_normalisation()
        cf_averaging = self.cf_config.get_averaging()
        cf_exclude = self.cf_config.get_exclude()
        cf_percent_clipped_threshold = self.cf_config.get_percent_clipped_threshold()
        cf_write_vox_data = self.cf_config.get_save_voxel_data()
        preproc_dict = {}
        if cf_normal in curve_fit.NORM_SETTING_STR_ENUM_MAP.keys():
            preproc_dict['normalise'] = curve_fit.NORM_SETTING_STR_ENUM_MAP[cf_normal]
        if cf_averaging in curve_fit.AV_SETTING_STR_ENUM_MAP.keys():
            preproc_dict['average'] = curve_fit.AV_SETTING_STR_ENUM_MAP[cf_averaging]
        if cf_exclude in curve_fit.EXCL_SETTING_STR_ENUM_MAP.keys():
            preproc_dict['exclude'] = curve_fit.EXCL_SETTING_STR_ENUM_MAP[cf_exclude]
        assert isinstance(cf_percent_clipped_threshold, numbers.Number), "Please check config file 'percent_clipped_threshold' needs to be a number (detected type:%s)" % type(cf_percent_clipped_threshold)
        preproc_dict['percent_clipped_threshold'] = cf_percent_clipped_threshold
        # prepare a vector to make a mapping between the image data and the analysis folders
        data_map_num = 0
        data_map_filename = os.path.join(out_dir, "data_map_%03d.csv" % data_map_num)
        while(os.path.isfile(data_map_filename)):
            data_map_num += 1
            data_map_filename = os.path.join(out_dir, "data_map_%03d.csv" % data_map_num)
        t1vir_map_df = None
        t1vfa_map_df = None
        t2mse_map_df = None
        t2star_map_df = None
        dw_map_df = None
        t1vir_map_d_arr = []
        t1vfa_map_d_arr = []
        t2mse_map_d_arr = []
        t2star_map_d_arr = []
        dw_map_d_arr = []
        # ----------------------------------------------------
        # T1 Variable Inversion Recovery
        for t1_vir_imageset in t1_vir_imagesets:
            t1_vir_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t1_vir_model_list = self.cf_config.get_t1_vir_models()
            inversion_exclusion_list = self.cf_config.get_t1_vir_exclusion_list()
            exclusion_label = self.cf_config.get_t1_vir_exclusion_label()
            for t1_vir_model_str in t1_vir_model_list:
                mdl = None
                if t1_vir_model_str == "2_param":
                    mdl = curve_fit.T1VIRCurveFitAbstract2Param(imageset=t1_vir_imageset,
                                                                reference_phantom=ref_phan,
                                                                initialisation_phantom=init_phan,
                                                                preprocessing_options=preproc_dict,
                                                                inversion_exclusion_list=inversion_exclusion_list,
                                                                exclusion_label=exclusion_label)
                elif t1_vir_model_str == "3_param":
                    mdl = curve_fit.T1VIRCurveFitAbstract3Param(imageset=t1_vir_imageset,
                                                                reference_phantom=ref_phan,
                                                                initialisation_phantom=init_phan,
                                                                preprocessing_options=preproc_dict,
                                                                inversion_exclusion_list=inversion_exclusion_list,
                                                                exclusion_label=exclusion_label)
                elif t1_vir_model_str == "4_param":
                    mdl = curve_fit.T1VIRCurveFitAbstract4Param(imageset=t1_vir_imageset,
                                                                reference_phantom=ref_phan,
                                                                initialisation_phantom=init_phan,
                                                                preprocessing_options=preproc_dict,
                                                                inversion_exclusion_list=inversion_exclusion_list,
                                                                exclusion_label=exclusion_label)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c, is_system=True,
                                                include_pmap_pages=include_roi_pmap_pages)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
                    # add data to the map to link dicom images to analysis folders
                    for series_uid, series_num, TI, TR in zip(t1_vir_imageset.series_instance_UIDs,
                                                              t1_vir_imageset.series_numbers,
                                                              t1_vir_imageset.meas_var_list,
                                                              t1_vir_imageset.repetition_time_list):
                        exclude_TI = TI in inversion_exclusion_list
                        t1vir_map_d_arr.append([t1_vir_imageset.label, t1_vir_model_str, series_num, TI, TR,
                                                cf_normal, cf_averaging, cf_exclude, cf_percent_clipped_threshold,
                                                exclude_TI, exclusion_label,
                                                series_uid, d_dir])
        if len(t1vir_map_d_arr):
            t1vir_map_col_names = ["Label", "Model", "SeriesNumber",
                                   "%s (%s)" % (t1_vir_imagesets[0].meas_var_name, t1_vir_imagesets[0].meas_var_units),
                                   "RepetitionTime",
                                   "Normalise", "Average", "ExcludeClipped", "ClipPcntThreshold",
                                   "Excluded", "ExclusionLabel",
                                   "SeriesInstanceUID", "AnalysisDir"]
            t1vir_map_df = pd.DataFrame(t1vir_map_d_arr, columns=t1vir_map_col_names)
        # ----------------------------------------------------
        # T1 Variable Flip Angle Recovery
        for t1_vfa_imageset in t1_vfa_imagesets:
            t1_vfa_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t1_vfa_model_list = self.cf_config.get_t1_vfa_models()
            angle_exclusion_list = self.cf_config.get_t1_vfa_exclusion_list()
            exclusion_label = self.cf_config.get_t1_vfa_exclusion_label()
            use_2D_roi = self.cf_config.get_t1_vfa_2D_roi_setting()
            centre_offset_2D_list = self.cf_config.get_t1_vfa_2D_slice_offset_list()
            if use_2D_roi:
                mu.log("MR-BIAS::analyse() : \tT1-VFA use a 2D ROI ...", LogLevels.LOG_INFO)
            for t1_vfa_model_str in t1_vfa_model_list:
                mdl = None
                if t1_vfa_model_str == "2_param":
                    mdl = curve_fit.T1VFACurveFitAbstract2Param(imageset=t1_vfa_imageset,
                                                                reference_phantom=ref_phan,
                                                                initialisation_phantom=init_phan,
                                                                preprocessing_options=preproc_dict,
                                                                angle_exclusion_list=angle_exclusion_list,
                                                                exclusion_label=exclusion_label,
                                                                use_2D_roi=use_2D_roi,
                                                                centre_offset_2D_list=centre_offset_2D_list)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c, is_system=True,
                                                include_pmap_pages=include_roi_pmap_pages)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
                    # add data to the map to link dicom images to analysis folders
                    for series_uid, series_num, FA, TR in zip(t1_vfa_imageset.series_instance_UIDs,
                                                              t1_vfa_imageset.series_numbers,
                                                              t1_vfa_imageset.meas_var_list,
                                                              t1_vfa_imageset.repetition_time_list):
                        exclude_FA = FA in angle_exclusion_list
                        t1vfa_map_d_arr.append([t1_vfa_imageset.label, t1_vfa_model_str, series_num, FA, TR,
                                                cf_normal, cf_averaging, cf_exclude, cf_percent_clipped_threshold,
                                                exclude_FA, exclusion_label, use_2D_roi, centre_offset_2D_list,
                                                series_uid, d_dir])
        if len(t1vfa_map_d_arr):
            t1vfa_map_col_names = ["Label", "Model", "SeriesNumber",
                                   "%s (%s)" % (t1_vfa_imagesets[0].meas_var_name, t1_vfa_imagesets[0].meas_var_units),
                                   "RepetitionTime",
                                   "Normalise", "Average", "ExcludeClipped", "ClipPcntThreshold",
                                   "Excluded", "ExclusionLabel", "Use2DROI", "2DSliceOffsets",
                                   "SeriesInstanceUID", "AnalysisDir"]
            t1vfa_map_df = pd.DataFrame(t1vfa_map_d_arr, columns=t1vfa_map_col_names)
        # ----------------------------------------------------
        # T2 Multiple Spin-echo
        for t2_mse_imageset in t2_mse_imagesets:
            t2_mse_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t2_mse_model_list = self.cf_config.get_t2_mse_models()
            echo_exclusion_list = self.cf_config.get_t2_mse_exclusion_list()
            exclusion_label = self.cf_config.get_t2_mse_exclusion_label()
            for t2_mse_model_str in t2_mse_model_list:
                mdl = None
                if t2_mse_model_str == "3_param":
                    mdl = curve_fit.T2SECurveFitAbstract3Param(imageset=t2_mse_imageset,
                                                               reference_phantom=ref_phan,
                                                               initialisation_phantom=init_phan,
                                                               preprocessing_options=preproc_dict,
                                                               echo_exclusion_list=echo_exclusion_list,
                                                               exclusion_label=exclusion_label)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c, is_system=True,
                                                include_pmap_pages=include_roi_pmap_pages)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
                    # add data to the map to link dicom images to analysis folders
                    for series_uid, series_num, TE, TR in zip(t2_mse_imageset.series_instance_UIDs,
                                                              t2_mse_imageset.series_numbers,
                                                              t2_mse_imageset.meas_var_list,
                                                              t2_mse_imageset.repetition_time_list):
                        exclude_TE = TE in echo_exclusion_list
                        t2mse_map_d_arr.append([t2_mse_imageset.label, t2_mse_model_str, series_num, TE, TR,
                                                cf_normal, cf_averaging, cf_exclude, cf_percent_clipped_threshold,
                                                exclude_TE, exclusion_label,
                                                series_uid, d_dir])

        if len(t2mse_map_d_arr):
            t2mse_map_col_names = ["Label", "Model", "SeriesNumber",
                                   "%s (%s)" % (t2_mse_imagesets[0].meas_var_name, t2_mse_imagesets[0].meas_var_units),
                                   "RepetitionTime",
                                   "Normalise", "Average", "ExcludeClipped", "ClipPcntThreshold",
                                   "Excluded", "ExclusionLabel",
                                   "SeriesInstanceUID", "AnalysisDir"]
            t2mse_map_df = pd.DataFrame(t2mse_map_d_arr, columns=t2mse_map_col_names)
        # ----------------------------------------------------
        # T2Star Gradient Echo
        for t2_star_imageset in t2_star_imagesets:
            t2_star_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t2_star_model_list = self.cf_config.get_t2_star_ge_models()
            echo_exclusion_list = self.cf_config.get_t2_star_ge_exclusion_list()
            exclusion_label = self.cf_config.get_t2_star_ge_exclusion_label()
            for t2_star_model_str in t2_star_model_list:
                mdl = None
                if t2_star_model_str == "2_param":
                    mdl = curve_fit.T2StarCurveFitAbstract2Param(imageset=t2_star_imageset,
                                                                 reference_phantom=ref_phan,
                                                                 initialisation_phantom=init_phan,
                                                                 preprocessing_options=preproc_dict,
                                                                 echo_exclusion_list=echo_exclusion_list,
                                                                 exclusion_label=exclusion_label)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c, is_system=True,
                                                include_pmap_pages=include_roi_pmap_pages)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
                    # add data to the map to link dicom images to analysis folders
                    for series_uid, series_num, TE, TR in zip(t2_star_imageset.series_instance_UIDs,
                                                              t2_star_imageset.series_numbers,
                                                              t2_star_imageset.meas_var_list,
                                                              t2_star_imageset.repetition_time_list):
                        exclude_TE = TE in echo_exclusion_list
                        t2star_map_d_arr.append([t2_star_imageset.label, t2_star_model_str, series_num, TE, TR,
                                                 cf_normal, cf_averaging, cf_exclude, cf_percent_clipped_threshold,
                                                 exclude_TE, exclusion_label,
                                                 series_uid, d_dir])

        if len(t2star_map_d_arr):
            t2star_map_col_names = ["Label", "Model", "SeriesNumber",
                                    "%s (%s)" % (
                                        t2_star_imagesets[0].meas_var_name, t2_star_imagesets[0].meas_var_units),
                                    "RepetitionTime",
                                    "Normalise", "Average", "ExcludeClipped", "ClipPcntThreshold",
                                    "Excluded", "ExclusionLabel",
                                    "SeriesInstanceUID", "AnalysisDir"]
            t2star_map_df = pd.DataFrame(t2star_map_d_arr, columns=t2star_map_col_names)
        # ----------------------------------------------------
        # DWI
        dw_2param_mdl_list = []
        for dw_imageset in dw_imagesets:
            dw_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            dw_model_list = self.cf_config.get_dw_models()
            bval_exclusion_list = self.cf_config.get_dw_exclusion_list()
            exclusion_label = self.cf_config.get_dw_exclusion_label()
            use_2D_roi = self.cf_config.get_dw_2D_roi_setting()
            if use_2D_roi:
                mu.log("MR-BIAS::analyse() : \tDW use a 2D ROI ...", LogLevels.LOG_INFO)
                centre_offset_2D_list = self.cf_config.get_dw_2D_slice_offset_list()
            else:
                centre_offset_2D_list = None
            for dw_model_str in dw_model_list:
                mdl = None
                if dw_model_str == "2_param":
                    mdl = curve_fit.DWCurveFitAbstract2Param(imageset=dw_imageset,
                                                             reference_phantom=ref_phan,
                                                             initialisation_phantom=init_phan,
                                                             preprocessing_options=preproc_dict,
                                                             bval_exclusion_list=bval_exclusion_list,
                                                             exclusion_label=exclusion_label,
                                                             use_2D_roi=use_2D_roi,
                                                             centre_offset_2D_list=centre_offset_2D_list)
                    dw_2param_mdl_list.append(mdl)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c,
                                                is_system=False,
                                                include_pmap_pages=include_roi_pmap_pages)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
                    # add data to the map to link dicom images to analysis folders
                    for series_uid, series_num, bval, TR in zip(dw_imageset.series_instance_UIDs,
                                                                dw_imageset.series_numbers,
                                                                dw_imageset.meas_var_list,
                                                                dw_imageset.repetition_time_list):
                        exclude_b = bval in bval_exclusion_list
                        dw_map_d_arr.append([dw_imageset.label, dw_model_str, series_num, bval, TR,
                                             cf_normal, cf_averaging, cf_exclude, cf_percent_clipped_threshold,
                                             exclude_b, exclusion_label, use_2D_roi, centre_offset_2D_list,
                                             series_uid, d_dir])
        if len(dw_map_d_arr):
            dw_map_col_names = ["Label", "Model", "SeriesNumber",
                                "%s (%s)" % (dw_imagesets[0].meas_var_name, dw_imagesets[0].meas_var_units),
                                "RepetitionTime",
                                "Normalise", "Average", "ExcludeClipped", "ClipPcntThreshold",
                                "Excluded", "ExclusionLabel", "Use2DROI", "2DSliceOffsets",
                                "SeriesInstanceUID", "AnalysisDir"]
            dw_map_df = pd.DataFrame(dw_map_d_arr, columns=dw_map_col_names)

        # join the data mapping frames and save to disk
        map_vec = []
        for m in [t1vir_map_df, t1vfa_map_df, t2mse_map_df, t2star_map_df, dw_map_df]:
            if m is not None:
                map_vec.append(m)
        if len(map_vec):
            df_data_analysis_map = pd.concat([t1vir_map_df, t1vfa_map_df, t2mse_map_df, t2star_map_df, dw_map_df],
                                             axis=0, join='outer')
            df_data_analysis_map.to_csv(data_map_filename)

        if len(dw_2param_mdl_list):
            #summary page in the pdf per model
            self.write_pdf_aggregated_diffusion_page(dw_2param_mdl_list, c, out_dir)
        # close the summary pdf
        c.save()
        # close the logger
        mu.detatch_logger()

    def write_pdf_aggregated_diffusion_page(self, mdl_list, c, data_dir):
        df_list = []
        model_names = []
        for mdl in mdl_list:
            df_list.append(mdl.get_summary_dataframe())
            temp_sup_title = "- CurveFit [%s - %s] <%s>" % (mdl.get_model_name(),
                                                 mdl.get_preproc_name(),
                                                 mdl.get_imageset_name())
            model_names.append(temp_sup_title)
        # calculate summary metrics and create summary df
        comb_df = pd.concat(df_list, ignore_index=True)
        sorted_df = comb_df.sort_values(by='RoiIndex', ascending=True)
        pooled_df = sorted_df.groupby('RoiLabel')['ADC (mean)'].agg(['mean', 'std'])
        pooled_df = pooled_df.sort_values(by='RoiLabel', key=lambda x: x.str.extract('(\d+)', expand=False).astype(int))
        pooled_df['RC_st'] = 2.77 * pooled_df.iloc[:, 1]  # Assuming column 1 is ADC
        pooled_df['CV_st'] = 100 * pooled_df.iloc[:, 1] / pooled_df.iloc[:, 0]
        df_list[0] = df_list[0].set_index('RoiLabel')
        pooled_df['ref'] = df_list[0]['ADC_reference']

        pooled_df['bias (%)'] = 100 * ((pooled_df['mean'] - pooled_df['ref']) / pooled_df['ref'])
        pooled_df.reset_index(inplace=True)
        pooled_df = pooled_df[['RoiLabel', 'ref', 'mean', 'std', 'RC_st', 'CV_st', 'bias (%)']]


        # print the pooled dataframe
        mu.log("Summary metric dataframe \n (%s)" %
                   pooled_df, LogLevels.LOG_INFO)

        # write summary metrics to table in PDF
        # prepare PDF
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.font_size)
        sup_title = "Repeatability metrics across 4 short-term scan repeats:"
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin, sup_title)
        off_dx = 7.5
        for name in model_names:
            off_dx = off_dx + 14
            c.drawString(pdf.left_margin*3, pdf.page_height - pdf.top_margin - off_dx, name)

        c.setFont(pdf.font_name, pdf.small_font_size)
        
        # need to define the 3 components for the table writing -> temp
        col_names = ["RoiLabel", "ref", "mean", "std", "RC_st", "CV_st", "bias (%)"]
        header_str = "| ROI LABEL |"
        row_fmt_str = "| %9s |"

        for quant in col_names:
            if quant != "RoiLabel":
                header_str = "%s %s |" % (header_str, " %8s " % quant)
                row_fmt_str = row_fmt_str + " %10.2f |"
        # table writing
        table_width = len(header_str)

        off_dx = 9.5
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - off_dx * pdf.small_line_width,
                         "=" * table_width)
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - (off_dx + 1) * pdf.small_line_width, header_str)
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - (off_dx + 2) * pdf.small_line_width,
                         "=" * table_width)
        off_dx = off_dx + 3

        # table content
        for line_dx, (idx, r) in enumerate(pooled_df.iterrows()):
            val_list = []
            for name in col_names:
                val_list.append(r[name])
            c.drawString(pdf.left_margin,
                             pdf.page_height - pdf.top_margin - pdf.small_line_width * (line_dx + off_dx),
                             row_fmt_str % tuple(val_list))
        # final borderline
        c.drawString(pdf.left_margin,
                         pdf.page_height - pdf.top_margin - pdf.small_line_width * (line_dx + off_dx + 1),
                         "=" * table_width)

        s_dx = line_dx + off_dx + 8

        # -------------------------------------------------------------
        # TABLE WITH EQUATION DETAILS
        # -------------------------------------------------------------
        # write the model equation and a list of parameters
        off_dx = 0
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx),
                        "SUMMARY METRICS:")
        off_dx += 2
        off_dx = off_dx + 1
        # table of parameters
        header_str = "|  Metric  |  Description                                                 |                      Symbol                      |"
        table_width = len(header_str)
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx), "-" * table_width)
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx + 1), header_str)
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx + 2), "-" * table_width)
        off_dx = off_dx + 3
        metrics = [("ref", "Reference ADC value as measured by NIST", "ADC_ref"), 
                   ("mean", "Mean ADC value across 4 short-term repeats", "ADC_mean"), 
                   ("std", "Standard deviation of ADC values across 4 short-term repeats", "SD"), 
                   ("RC_st", "Repeatability coefficient", "RC_st = 2.77 * SD"), 
                   ("CV_st", "Coefficient of variation", "CV_st = 100% * (SD/ADC_mean)"), 
                   ("bias (%)", "%Bias", "bias(%) = 100% * (ADC_mean - ADC_ref)/ADC_ref")]
        for p_name, descr, eqn in metrics:
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx),
                            "| %8s | %-60s | %-48s |" % (p_name, descr, eqn))
            off_dx = off_dx + 1
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx), "-" * table_width)
        s_dx += off_dx + 6

        c.showPage()  # new page

        # write summary metrics df to disk as csv
        # pooled_df
        summary_datafilename = os.path.join(data_dir, "repeatability_metrics.csv")
        mu.log("Writing repeatability metrics data to %s" %
                (summary_datafilename), LogLevels.LOG_INFO)
        pooled_df.to_csv(summary_datafilename)


    def write_pdf_title_page(self, c):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.font_size)  # set to a fixed width font

        # Create a banner
        c.drawString(2*pdf.left_margin,
                     pdf.page_height - pdf.page_height/8. + pdf.line_width,
                     "="*130)
        logo_str_list = ["      ___           ___                                             ___           ___         ",
                         "     /__/\         /  /\                 _____        ___          /  /\         /  /\        ",
                         "    |  |::\       /  /::\               /  /::\      /  /\        /  /::\       /  /:/_       ",
                         "    |  |:|:\     /  /:/\:\             /  /:/\:\    /  /:/       /  /:/\:\     /  /:/ /\      ",
                         "  __|__|:|\:\   /  /:/~/:/            /  /:/~/::\  /__/::\      /  /:/~/::\   /  /:/ /::\     ",
                         " /__/::::| \:\ /__/:/ /:/___   ____  /__/:/ /:/\:| \__\/\:\__  /__/:/ /:/\:\ /__/:/ /:/\:\    ",
                         " \  \:\~~\__\/ \  \:\/:::::/  /___/\ \  \:\/:/~/:/    \  \:\/\ \  \:\/:/__\/ \  \:\/:/~/:/    ",
                         "  \  \:\        \  \::/~~~~   \___\/  \  \::/ /:/      \__\::/  \  \::/       \  \::/ /:/     ",
                         "   \  \:\        \  \:\                \  \:\/:/       /__/:/    \  \:\        \__\/ /:/      ",
                         "    \  \:\        \  \:\                \  \::/        \__\/      \  \:\         /__/:/       ",
                         "     \__\/         \__\/                 \__\/                     \__\/         \__\/        "]
        for logo_dx, logo_str in enumerate(logo_str_list):
            c.drawString(pdf.left_margin + pdf.page_width/6.,
                         pdf.page_height - pdf.page_height/8. - logo_dx*pdf.line_width,
                         logo_str)
        # c.drawString(pdf.left_margin,
        #              pdf.page_height - pdf.page_height/6. - (len(logo_str_list)+1)*pdf.line_width,
        #              "-"*130)

        line_start_pos = pdf.page_height - pdf.page_height/8. - (len(logo_str_list)+2)*pdf.line_width
        # write the version information
        c.drawString(pdf.left_margin + pdf.page_width/6.,
                     line_start_pos,
                     "                   MR-BIAS v%s (released on %s)" % (mu.MRBIAS_VERSION_NUMBER, mu.MRBIAS_VERSION_DATE))
        c.drawString(pdf.left_margin + pdf.page_width/6.,
                     line_start_pos - 1.5*pdf.line_width,
                     "               Source code: %s" % (mu.MRBIAS_URL))
        c.linkURL("%s" % mu.MRBIAS_URL,
                  (pdf.left_margin + pdf.page_width/6. + 15*pdf.line_width,
                   line_start_pos - 0.5*pdf.line_width,
                   pdf.left_margin + pdf.page_width/6. + 40*pdf.line_width,
                   line_start_pos - 2.0*pdf.line_width),
                   relative=0,
                   thickness=0)
        c.drawString(2*pdf.left_margin,
                     line_start_pos - 3*pdf.line_width,
                     "="*130)

        line_start_pos_a = line_start_pos - 8.0 * pdf.line_width
        cite_str_relx = ["      TITLE: \"Magnetic resonance biomarker assessment software (MR-BIAS): an ",
                         "               automated open-source tool for the ISMRM/NIST system phantom\"",
                         "    AUTHORS: James C Korte, Zachary Chin, Madeline Carr, Lois Holloway, Rick Franich",
                         "    JOURNAL: Physics in Medicine & Biology",
                         "       YEAR: 2023",
                         "        DOI: https://doi.org/10.1088/1361-6560/acbcbb"]
        doi_relax = mu.MRBIAS_RELAXOMETRY_DOI_URL
        line_start_pos_b = line_start_pos - 20.0 * pdf.line_width
        cite_str_diff = ["      TITLE: \"Open-source quality assurance for multi-parametric MRI: a diffusion analysis",
                         "               update for the magnetic resonance biomarker assessment software (MR-BIAS)\"",
                         "    AUTHORS: James C Korte, Stanley A Norris, Madeline E Carr, Lois Holloway, Glenn D Cahoon",
                         "             Ben Neijndorff, Petra van Houdt, Rick Franich",
                         "    JOURNAL: UNDER REVIEW",
                         "       YEAR: UNDER REVIEW",
                         "        DOI: https://doi.org/TBD"]
        doi_diffusion = mu.MRBIAS_DIFFUSION_DOI_URL

        # write the citation/reference details (relaxometry phantoms)
        for line_start_pos, cite_str_list, doi_url, phantom_type in zip([line_start_pos_a, line_start_pos_b],
                                                                        [cite_str_relx, cite_str_diff],
                                                                        [doi_relax, doi_diffusion],
                                                                        ["relaxometry", "diffusion"]):
            c.drawString(pdf.left_margin + pdf.page_width / 8., line_start_pos - 2 * pdf.line_width,
                         "-" * 90)
            c.drawString(pdf.left_margin, line_start_pos - 3*pdf.line_width,
                         "                  Please cite the following publication (for %s):" % phantom_type)
            c.drawString(pdf.left_margin + pdf.page_width/8., line_start_pos - 4*pdf.line_width,
                         "-"*90)
            for cite_dx, cite_str in enumerate(cite_str_list):
                c.drawString(pdf.left_margin + pdf.page_width/8.,
                             line_start_pos - (5+cite_dx)*pdf.line_width,
                             cite_str)
            # link the DOI to the publication URL
            c.linkURL("%s" % doi_url,
                      (pdf.left_margin + pdf.page_width/8. + 5*pdf.line_width,
                       line_start_pos - (3+len(cite_str_list))*pdf.line_width,
                       pdf.left_margin + pdf.page_width/8. + 30*pdf.line_width,
                       line_start_pos - (3+len(cite_str_list))*pdf.line_width - 2.0*pdf.line_width),
                       relative=0,
                       thickness=0)
            c.drawString(pdf.left_margin + pdf.page_width/8., line_start_pos - (5+len(cite_str_list))*pdf.line_width,
                         "-"*90)

        c.showPage()  # new page


    @staticmethod
    def get_directory_name_from_dcm_metadata(dicom_directory):
        dcm_search = mu.DICOMSearch(dicom_directory, read_single_file=True)
        dcm_df = dcm_search.get_df()
        meta_data = OrderedDict()
        for attrib, missing_val in [("InstitutionName", "Inst"),
                                    ("Manufacturer", "Manufact"),
                                    ("ManufacturerModelName", "Model"),
                                    ("MagneticFieldStrength", 0.0),
                                    ("DeviceSerialNumber", 123456),
                                    ("SeriesDate", 17071986),
                                    ("SeriesTime", 162000)]:
            if attrib in dcm_df.columns:
                meta_data[attrib] = dcm_df[attrib].iloc[0]
            else:
                meta_data[attrib] = missing_val
        output_subdir_name = "%s_%s-%s-%sT-%s_%s-%s" % (meta_data["InstitutionName"],
                                                        meta_data["Manufacturer"],
                                                        meta_data["ManufacturerModelName"],
                                                        ("%0.1f" % meta_data["MagneticFieldStrength"]).
                                                        replace(".", "p"),
                                                        str(meta_data["DeviceSerialNumber"]),
                                                        str(meta_data["SeriesDate"]),
                                                        str(meta_data["SeriesTime"]))
        return output_subdir_name


class MRIBIASConfiguration(ABC):
    def __init__(self, config_filename):
        assert os.path.isfile(config_filename), "MR-BIASConfiguration::__init__(): couldn't locate config_file: %s" % config_filename
        mu.log("MR-BIASConfiguration::__init__(): parsing config file : %s" % config_filename, LogLevels.LOG_INFO)
        self.config = yaml.full_load(open(config_filename))

    def get_global_config(self):
        return MRIBIASConfiguration.__safe_get("global", self.config)
    def get_phantom_experiment_config(self):
        return MRIBIASConfiguration.__safe_get("phantom_experiment", self.config)
    def get_dicom_sorting_config(self):
        return MRIBIASConfiguration.__safe_get("dicom_sorting", self.config)
    def get_roi_detection_config(self):
        return MRIBIASConfiguration.__safe_get("roi_detection", self.config)
    def get_curve_fitting_config(self):
        return MRIBIASConfiguration.__safe_get("curve_fitting", self.config)

    def get(self, param_name, default_value):
        sub_config = self.get_sub_config()
        if sub_config is not None:
            if param_name in sub_config.keys():
                return sub_config[param_name]
        # not found, return a default value
        mu.log("%s::get(%s): not found in configuration file, "
               "using default value : %s" % (type(self).__name__, param_name, str(default_value)), LogLevels.LOG_WARNING)
        return default_value

    @abstractmethod
    def get_sub_config(self):
        return None

    @staticmethod
    def __safe_get(keyname, d):
        if keyname in d.keys():
            return d[keyname]
        return None

class MRIBiasGlobalConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_sub_config(self):
        return super().get_global_config()

    def get_output_directory(self):
        return self.get("output_directory", os.path.join(os.getcwd(), "output"))
    def get_overwrite_existing_output(self):
        return self.get("overwrite_existing_output", False)

class MRIBiasDICOMSortConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_sub_config(self):
        return super().get_dicom_sorting_config()

    def get_scan_protocol_for_sorting(self):
        return self.get("scan_protocol", None)
    def get_show_unknown_series(self):
        return self.get("show_unknown_series", True)


class MRIBiasROIDetectConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_sub_config(self):
        return super().get_roi_detection_config()

    def get_template(self):
        return self.get("template_name", None)
    def get_detection_method(self):
        return self.get("registration_method", "two_stage_msme-GS_correl-GD")
    def get_flip_cap_series_numbers(self):
        return self.get("flip_cap_series_numbers", [])
    def get_shape_fine_tune(self):
        return self.get("shape_fine_tune", False)
    def get_registration_partial_fov(self):
        return self.get("partial_fov", False)
    def get_use_first_detection_only(self):
        return self.get("use_first_detection_only", False)
    def get_use_manual_roi(self):
        return self.get("use_manual_roi", False)
    def get_manual_T1_roi_filepath(self):
        return self.get("manual_roi_t1_filepath", None)
    def get_manual_T2_roi_filepath(self):
        return self.get("manual_roi_t2_filepath", None)
    def get_manual_DW_roi_filepath(self):
        return self.get("manual_roi_dw_filepath", None)


class MRIBiasExperimentConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_sub_config(self):
        return super().get_phantom_experiment_config()

    def get_phantom_manufacturer(self):
        return self.get("phantom_manufacturer", "caliber_mri")
    def get_phantom_type(self):
        return self.get("phantom_type", "system_phantom")
    def get_phantom_serial_number(self):
        return self.get("phantom_serial_number", "130-0093")
    def get_field_strength_tesla(self):
        return self.get("field_strength_tesla", 3.0)
    def get_temperature_celsius(self):
        return self.get("temperature_celsius", 20.0)


class MRIBiasCurveFitConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_sub_config(self):
        return super().get_curve_fitting_config()

    # VISUALISATION
    def get_include_roi_pmaps(self):
        return self.get("roi_pmaps_in_pdf", False)

    # PREPROCESSING
    def get_averaging(self):
        return self.get("averaging", None)
    def get_normalisation(self):
        return self.get("normalisation", "voxel_max")
    def get_exclude(self):
        return self.get("exclude", "clipped")
    def get_percent_clipped_threshold(self):
        return self.get("percent_clipped_threshold", 200)  # no parital clipping

    # T1 VIR SETTINGS
    def __get_t1_vir(self, param_name, default_value):
        return self.__get_nestled("t1_vir_options", param_name, default_value)
    def get_t1_vir_models(self):
        return self.__get_t1_vir("fitting_models", ["3_param"])
    def get_t1_vir_exclusion_list(self):
        return self.__get_t1_vir("inversion_exclusion_list", [])
    def get_t1_vir_exclusion_label(self):
        return self.__get_t1_vir("inversion_exclusion_label", "user_IR_excld")
    def get_t1_vir_2D_slice_offset_list(self):
        return self.__get_2D_slice_offset_list("t1_vir_options")

    # T1 VFA SETTINGS
    def __get_t1_vfa(self, param_name, default_value):
        return self.__get_nestled("t1_vfa_options", param_name, default_value)
    def get_t1_vfa_models(self):
        return self.__get_t1_vfa("fitting_models", ["2_param"])
    def get_t1_vfa_exclusion_list(self):
        return self.__get_t1_vfa("angle_exclusion_list", [])
    def get_t1_vfa_exclusion_label(self):
        return self.__get_t1_vfa("angle_exclusion_label", "user_angle_excld")
    def get_t1_vfa_2D_roi_setting(self):
        return self.__get_t1_vfa("use_2D_roi", False)
    def get_t1_vfa_2D_slice_offset_list(self):
        return self.__get_2D_slice_offset_list("t1_vfa_options")

    # T2 MSE SETTINGS
    def __get_t2_mse(self, param_name, default_value):
        return self.__get_nestled("t2_mse_options", param_name, default_value)
    def get_t2_mse_models(self):
        return self.__get_t2_mse("fitting_models", ["3_param"])
    def get_t2_mse_exclusion_list(self):
        return self.__get_t2_mse("echo_exclusion_list", [])
    def get_t2_mse_exclusion_label(self):
        return self.__get_t2_mse("echo_exclusion_label", "user_angle_excld")
    def get_t2_mse_2D_slice_offset_list(self):
        return self.__get_2D_slice_offset_list("t2_mse_options")

    # T2 Star GE SETTINGS
    def __get_t2_star_ge(self, param_name, default_value):
        return self.__get_nestled("t2_star_ge_options", param_name, default_value)
    def get_t2_star_ge_models(self):
        return self.__get_t2_star_ge("fitting_models", ["2_param"])
    def get_t2_star_ge_exclusion_list(self):
        return self.__get_t2_star_ge("echo_exclusion_list", [])
    def get_t2_star_ge_exclusion_label(self):
        return self.__get_t2_star_ge("echo_exclusion_label", "user_angle_excld")
    def get_t2_star_ge_2D_slice_offset_list(self):
        return self.__get_2D_slice_offset_list("t2_star_ge_options")

    # DW SETTINGS
    def __get_dw(self, param_name, default_value):
        return self.__get_nestled("dw_options", param_name, default_value)
    def get_dw_models(self):
        return self.__get_dw("fitting_models", ["2_param"])
    def get_dw_exclusion_list(self):
        return self.__get_dw("bval_exclusion_list", [])
    def get_dw_exclusion_label(self):
        return self.__get_dw("bval_exclusion_label", "user_bval_excld")
    def get_dw_2D_roi_setting(self):
        return self.__get_dw("use_2D_roi", False)
    def get_dw_2D_slice_offset_list(self):
        return self.__get_2D_slice_offset_list("dw_options")

    # OUTPUT SETTINGS
    def get_save_voxel_data(self):
        return self.get("save_voxel_data", False)
    def get_save_parameter_maps(self):
        return self.get("save_parameter_maps", False)

    # helper functions
    def __get_nestled(self, option_name, param_name, default_value):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if option_name in cf_config.keys():
                opts = cf_config[option_name]
                if param_name in opts.keys():
                    return opts[param_name]
        # not found, return a default value
        mu.log("MRIBiasCurveFitConfig::get(%s -> %s): not found in configuration file, "
               "using default value : %s" % (option_name, param_name, str(default_value)), LogLevels.LOG_WARNING)
        return default_value
    def __get_2D_slice_offset_list(self, model_option):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if model_option in cf_config.keys():
                opts = cf_config[model_option]
                if "slice_offset_2D" in opts.keys():
                    return [opts["slice_offset_2D"]] # put single value in list
                if "slice_offset_2D_list" in opts.keys():
                    return opts["slice_offset_2D_list"]
        # not found, return a default value
        default_value = [0]
        mu.log("MRIBiasCurveFitConfig::get_dw_2D_slice_offset_list(%s): not found in configuration file, "
               "using default value : %s" % (model_option, str(default_value)), LogLevels.LOG_WARNING)
        return default_value

if __name__ == "__main__":
    main()