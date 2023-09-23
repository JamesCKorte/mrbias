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
from collections import OrderedDict
import numbers

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
    test_configuration_file = os.path.join(os.getcwd(), "..", "config", "example_config.yaml")
    # specific the dicom directories to analyse
    test_dicom_directory_a = os.path.join(os.getcwd(), "..", "data", "mrbias_testset_A")

    # create a MRBIAS analysis object
    mrb = MRBIAS(test_configuration_file, write_to_screen=True)
    # run the analysis (output will be created in the "output_directory" specified in the configuration file)
    mrb.analyse(test_dicom_directory_a)

    mu.log("------ FIN -------", LogLevels.LOG_INFO)


class MRBIAS(object):
    def __init__(self, config_filename, write_to_screen=True):
        mu.log("MR-BIAS::__init__(): parsing config file : %s" % config_filename, LogLevels.LOG_INFO)
        self.config_filename = config_filename
        self.conf = MRIBIASConfiguration(config_filename)
        self.write_to_screen = write_to_screen
        # setup (global) class configuration from file
        self.output_dir = self.conf.get_output_directory()
        self.overwrite_existing_output = self.conf.get_overwrite_existing_output()


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
        scan_protocol = self.conf.get_scan_protocol_for_sorting()
        ss = None
        if scan_protocol == "siemens_skyra_3p0T":
            ss = scan_session.SystemSessionSiemensSkyra(dicom_directory)
        elif scan_protocol == "philips_marlin_1p5T":
            ss = scan_session.SystemSessionPhilipsMarlin(dicom_directory)
        elif scan_protocol == "auckland_cam_3p0T":
            ss = scan_session.SystemSessionAucklandCAM(dicom_directory)
        elif scan_protocol == "siemens_skyra_erin_3p0T":
            ss = scan_session.SystemSessionSiemensSkyraErin(dicom_directory)
        elif scan_protocol == "philips_ingenia_ambitionX":
            ss = scan_session.SystemSessionPhilipsIngeniaAmbitionX(dicom_directory)
        elif scan_protocol == "diff_philips_ingenia_ambitionX":
            ss = scan_session.DiffusionSessionPhilipsIngeniaAmbitionX(dicom_directory)
        else:
            mu.log("MR-BIAS::analyse(): skipping analysis as unknown 'scan_protocol' defined for DICOM sorting",
                   LogLevels.LOG_WARNING)
        # if a valid scan protocol found load up relevant image sets
        geometric_images = []
        pd_images = []
        t1_vir_imagesets = []
        t1_vfa_imagesets = []
        t2_mse_imagesets = []
        dw_imagesets = []
        if ss is not None:
            geometric_images = ss.get_geometric_images()
            #pd_images = ss.get_proton_density_images()
            t1_vir_imagesets = ss.get_t1_vir_image_sets()
            t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
            t2_mse_imagesets = ss.get_t2_mse_image_sets()
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
        for dw_imageset in dw_imagesets:
            mu.log("Found DW: %s" % type(dw_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(dw_imageset), LogLevels.LOG_INFO)

        geometric_images_linked = set()
        if ss is not None:
            # exclude any geometric images that are not reference in curve fit data
            mu.log("MR-BIAS::analyse(): Identify linked geometric images ...", LogLevels.LOG_INFO)
            for geometric_image in geometric_images:
                for fit_imagesets in [t1_vir_imagesets, t1_vfa_imagesets, t2_mse_imagesets, dw_imagesets]:
                    for imageset in fit_imagesets:
                        g = imageset.get_geometry_image()
                        if g.get_label() == geometric_image.get_label():
                            geometric_images_linked.add(geometric_image)
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
        roi_template = self.conf.get_roi_template()
        roi_reg_method = self.conf.get_roi_registration_method()
        roi_is_partial_fov = self.conf.get_roi_registration_partial_fov()
        roi_template_dir = None
        if roi_template == "siemens_skyra_3p0T":
            roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_skyra_3p0T")
        elif roi_template == "systemlite_siemens_vida_3p0T":
            roi_template_dir = os.path.join(mu.reference_template_directory(), "systemlite_siemens_vida_3p0T")
        elif roi_template == "systemlite_siemens_vida_3p0T_180degrees":
            roi_template_dir = os.path.join(mu.reference_template_directory(), "systemlite_siemens_vida_3p0T_180degrees")
        elif roi_template == "philips_ingenia_1p5T":
            roi_template_dir = os.path.join(mu.reference_template_directory(), "philips_ingenia_1p5T")
        # ... add others
        if roi_template is None:
            mu.log("MR-BIAS::analyse(): skipping analysis as unknown 'roi_template' defined for ROI detection",
                   LogLevels.LOG_ERROR)
            return None
        elif not (len(geometric_images_linked) > 0):
            mu.log("MR-BIAS::analyse(): skipping analysis as no linked geometry imaged found for ROI detection",
                   LogLevels.LOG_ERROR)
            return None
        else:
            roi_detectors = OrderedDict()
            for geom_image in geometric_images_linked:
                # create a roi detector
                reg_method = None
                if roi_reg_method == "two_stage_msme-GS_correl-GD":
                    reg_method = roi_detect.RegistrationOptions.TWOSTAGE_MSMEGS_CORELGD
                elif roi_reg_method == "mattesMI-GD":
                    reg_method = roi_detect.RegistrationOptions.MMI_GRADIENTDESCENT
                elif roi_reg_method == "correl-GD":
                    reg_method = roi_detect.RegistrationOptions.COREL_GRADIENTDESCENT
                assert reg_method is not None, "MR-BIAS::analyse(): invalid ROI registration method selected - please check your configuration file"
                roi_detector = roi_detect.ROIDetector(geom_image, roi_template_dir,
                                                      registration_method=reg_method,
                                                      partial_fov=roi_is_partial_fov)
                # detect the ROIs and store the masks on the target image
                roi_detector.detect()
                # add a summary page to the PDF
                roi_detector.write_pdf_summary_page(c)
                # store detector
                roi_detectors[geom_image.get_label()] = roi_detector


        # ===================================================================================================
        # Fit parametric models to the raw voxel data
        # ===================================================================================================
        mu.log("-" * 100, LogLevels.LOG_INFO)
        mu.log("MR-BIAS::analyse() : fit parametric models to the detected ROIs on each imageset ...", LogLevels.LOG_INFO)
        mu.log("-" * 100, LogLevels.LOG_INFO)
        # --------------------------------------------------------------------
        # get phantom/experiment details from the configuration file
        # --------------------------------------------------------------------
        phan_config = MRIBiasExperimentConfig(self.config_filename)
        # phantom details
        phantom_maker = phan_config.get_phantom_manufacturer()
        phantom_type = phan_config.get_phantom_type()
        phantom_sn = phan_config.get_phantom_serial_number()
        ph_model_num, ph_item_num = phantom_sn.split("-")
        if not ((phantom_maker == "caliber_mri") and (phantom_type =="system_phantom") and (ph_model_num == "130")) and \
            not ((phantom_maker == "caliber_mri") and (phantom_type =="diffusion_phantom") and (ph_model_num == "128")):
            mu.log("MR-BIAS::analyse(): only supports phantom [caliber_mri:system_phantom(130) and caliber_mri:diffusion_phantom(128)] (not [%s:%s()]) "
                   "skipping analysis... " % (phantom_maker, phan_config, ph_model_num), LogLevels.LOG_ERROR)
            return None
        
        #experiment details
        field_strength = phan_config.get_field_strength_tesla()
        temperature_celsius = phan_config.get_temperature_celsius()
        if phantom_type == "system_phantom":
            init_phan = phantom.ReferencePhantomCalibreSystemFitInit(field_strength=field_strength,  # Tesla
                                                                 temperature=temperature_celsius)  # Celsius
        if phantom_type == "diffusion_phantom":
            init_phan = phantom.ReferencePhantomDiffusionFitInit(field_strength=field_strength,  # Tesla
                                                                 temperature=temperature_celsius)  # Celsius
        mu.log("init_phan: type, value, etc...", LogLevels.LOG_ERROR)
        mu.log(type(init_phan), LogLevels.LOG_ERROR)
        mu.log(init_phan, LogLevels.LOG_ERROR)

        # select the reference phantom based on phantom serial number
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
        
        elif ph_model_num == "128":
            ph_item_num = int(ph_item_num)
            ref_phan = None
            ref_phan = phantom.ReferencePhantomDiffusion1(field_strength=field_strength,  # Tesla
                                                                temperature=temperature_celsius,
                                                                serial_number=phantom_sn)  # Celsius
                

        # --------------------------------------------------------------------
        # get curve fitting details from the configuration file
        # --------------------------------------------------------------------
        cf_config = MRIBiasCurveFitConfig(self.config_filename)
        cf_normal = cf_config.get_normalisation()
        cf_averaging = cf_config.get_averaging()
        cf_exclude = cf_config.get_exclude()
        cf_percent_clipped_threshold = cf_config.get_percent_clipped_threshold()
        cf_write_vox_data = cf_config.get_save_voxel_data()
        preproc_dict = {}
        if cf_normal in curve_fit.NORM_SETTING_STR_ENUM_MAP.keys():
            preproc_dict['normalise'] = curve_fit.NORM_SETTING_STR_ENUM_MAP[cf_normal]
        if cf_averaging in curve_fit.AV_SETTING_STR_ENUM_MAP.keys():
            preproc_dict['average'] = curve_fit.AV_SETTING_STR_ENUM_MAP[cf_averaging]
        if cf_exclude in curve_fit.EXCL_SETTING_STR_ENUM_MAP.keys():
            preproc_dict['exclude'] = curve_fit.EXCL_SETTING_STR_ENUM_MAP[cf_exclude]
        assert isinstance(cf_percent_clipped_threshold, numbers.Number), "Please check config file 'percent_clipped_threshold' needs to be a number (detected type:%s)" % type(cf_percent_clipped_threshold)
        preproc_dict['percent_clipped_threshold'] = cf_percent_clipped_threshold
        # ----------------------------------------------------
        # T1 Variable Inversion Recovery
        for t1_vir_imageset in t1_vir_imagesets:
            t1_vir_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t1_vir_model_list = cf_config.get_t1_vir_models()
            inversion_exclusion_list = cf_config.get_t1_vir_exclusion_list()
            exclusion_label = cf_config.get_t1_vir_exclusion_label()
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
                    mdl.write_pdf_summary_pages(c)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
        # ----------------------------------------------------
        # T1 Variable Flip Angle Recovery
        for t1_vfa_imageset in t1_vfa_imagesets:
            t1_vfa_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t1_vfa_model_list = cf_config.get_t1_vfa_models()
            angle_exclusion_list = cf_config.get_t1_vfa_exclusion_list()
            exclusion_label = cf_config.get_t1_vfa_exclusion_label()
            use_2D_roi = cf_config.get_t1_vfa_2D_roi_setting()
            centre_offset_2D = cf_config.get_t1_vfa_2D_slice_offset()
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
                                                                centre_offset_2D=centre_offset_2D)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
        # ----------------------------------------------------
        # T2 Multiple Spin-echo
        for t2_mse_imageset in t2_mse_imagesets:
            t2_mse_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t2_mse_model_list = cf_config.get_t2_mse_models()
            echo_exclusion_list = cf_config.get_t2_mse_exclusion_list()
            exclusion_label = cf_config.get_t2_mse_exclusion_label()
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
                    mdl.write_pdf_summary_pages(c)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
        # ----------------------------------------------------
        # DWI
        for dw_imageset in dw_imagesets:
            dw_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            dw_model_list = cf_config.get_dw_models()
            for dw_model_str in dw_model_list:
                mdl = None
                if dw_model_str == "2_param":
                    mdl = curve_fit.DWCurveFitAbstract2Param(imageset=dw_imageset,
                                                                reference_phantom=ref_phan,
                                                                initialisation_phantom=init_phan,
                                                                preprocessing_options=preproc_dict)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages_dw(c)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)


        # close the summary pdf
        c.save()
        # close the logger
        mu.detatch_logger()


    def write_pdf_title_page(self, c):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.font_size)  # set to a fixed width font

        # Create a banner
        c.drawString(pdf.left_margin,
                     pdf.page_height - pdf.page_height/6. + pdf.line_width,
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
            c.drawString(pdf.left_margin + pdf.page_width/8.,
                         pdf.page_height - pdf.page_height/6. - logo_dx*pdf.line_width,
                         logo_str)
        # c.drawString(pdf.left_margin,
        #              pdf.page_height - pdf.page_height/6. - (len(logo_str_list)+1)*pdf.line_width,
        #              "-"*130)

        line_start_pos = pdf.page_height - pdf.page_height/6. - (len(logo_str_list)+2)*pdf.line_width
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
        c.drawString(pdf.left_margin,
                     line_start_pos - 3*pdf.line_width,
                     "="*130)

        # write the citation/reference details
        line_start_pos = line_start_pos - 10.0*pdf.line_width
        c.drawString(pdf.left_margin + pdf.page_width / 8., line_start_pos - 2 * pdf.line_width,
                     "-" * 90)
        c.drawString(pdf.left_margin, line_start_pos - 3*pdf.line_width,
                     "                  Please cite the following publication:")
        c.drawString(pdf.left_margin + pdf.page_width/8., line_start_pos - 4*pdf.line_width,
                     "-"*90)
        cite_str_list = ["      TITLE: \"Magnetic resonance biomarker assessment software (MR-BIAS): an ",
                         "               automated open-source tool for the ISMRM/NIST system phantom\"",
                         "    AUTHORS: James C Korte, Zachary Chin, Madeline Carr, Lois Holloway, Rick Franich",
                         "    JOURNAL: Physics in Medicine & Biology",
                         "       YEAR: 2023",
                         "        DOI: https://doi.org/10.1088/1361-6560/acbcbb"]
        for cite_dx, cite_str in enumerate(cite_str_list):
            c.drawString(pdf.left_margin + pdf.page_width/8.,
                         line_start_pos - (5+cite_dx)*pdf.line_width,
                         cite_str)
        # link the DOI to the publication URL
        c.linkURL("%s" % mu.MRBIAS_DOI_URL,
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
        dcm_search = scan_session.DICOMSearch(dicom_directory, read_single_file=True)
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


class MRIBIASConfiguration(object):
    def __init__(self, config_filename):
        assert os.path.isfile(config_filename), "MR-BIASConfiguration::__init__(): couldn't locate config_file: %s" % config_filename
        mu.log("MR-BIASConfiguration::__init__(): parsing config file : %s" % config_filename, LogLevels.LOG_INFO)
        self.config = yaml.full_load(open(config_filename))

    def get_output_directory(self):
        glob_config = self.__get_global_config()
        if glob_config is not None:
            if "output_directory" in glob_config.keys():
                return glob_config["output_directory"]
        # not found, return a default value
        default_value = os.path.join(os.getcwd(), "output")
        mu.log("MR-BIASConfiguration::get_output_directory(): not found in configuration file, using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_overwrite_existing_output(self):
        glob_config = self.__get_global_config()
        if glob_config is not None:
            if "overwrite_existing_output" in glob_config.keys():
                return glob_config["overwrite_existing_output"]
        # not found, return a default value
        default_value = False
        mu.log("MR-BIASConfiguration::get_overwrite_existing_output(): not found in configuration file, using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_scan_protocol_for_sorting(self):
        sort_config = self.__get_dicom_sorting_config()
        if sort_config is not None:
            if "scan_protocol" in sort_config.keys():
                x = sort_config["scan_protocol"]
                # todo: check with options (from another YAML file?)
                return x
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASConfiguration::get_scan_protocol_for_sorting(): not found in configuration file, using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_roi_template(self):
        detect_config = self.__get_roi_detection_config()
        if detect_config is not None:
            if "template_name" in detect_config.keys():
                x = detect_config["template_name"]
                # todo: check with options (from another YAML file?)
                return x
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASConfiguration::get_roi_template(): not found in configuration file, using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_roi_registration_method(self):
        detect_config = self.__get_roi_detection_config()
        if detect_config is not None:
            if "registration_method" in detect_config.keys():
                x = detect_config["registration_method"]
                # todo: check with options (from another YAML file?)
                return x
        # not found, return a default value
        default_value = "two_stage_msme-GS_correl-GD"
        mu.log("MR-BIASConfiguration::get_roi_registration_method(): not found in configuration file, using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_roi_registration_partial_fov(self):
        detect_config = self.__get_roi_detection_config()
        if detect_config is not None:
            if "partial_fov" in detect_config.keys():
                x = detect_config["partial_fov"]
                # todo: check with options (from another YAML file?)
                return x
        # not found, return a default value
        default_value = False
        mu.log("MR-BIASConfiguration::get_roi_registration_partial_fov(): not found in configuration file, using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value




    def __get_global_config(self):
        return MRIBIASConfiguration.__safe_get("global", self.config)
    def get_phantom_experiment_config(self):
        return MRIBIASConfiguration.__safe_get("phantom_experiment", self.config)
    def __get_dicom_sorting_config(self):
        return MRIBIASConfiguration.__safe_get("dicom_sorting", self.config)
    def __get_roi_detection_config(self):
        return MRIBIASConfiguration.__safe_get("roi_detection", self.config)
    def get_curve_fitting_config(self):
        return MRIBIASConfiguration.__safe_get("curve_fitting", self.config)

    @staticmethod
    def __safe_get(keyname, d):
        if keyname in d.keys():
            return d[keyname]
        return None

class MRIBiasExperimentConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_phantom_manufacturer(self):
        phan_config = super().get_phantom_experiment_config()
        if phan_config is not None:
            if "phantom_manufacturer" in phan_config.keys():
                return phan_config["phantom_manufacturer"]
        # not found, return a default value
        default_value = "caliber_mri"
        mu.log("MR-BIASExperimentConfig::get_phantom_manufacturer(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_phantom_type(self):
        phan_config = super().get_phantom_experiment_config()
        if phan_config is not None:
            if "phantom_type" in phan_config.keys():
                return phan_config["phantom_type"]
        # not found, return a default value
        default_value = "system_phantom"
        mu.log("MR-BIASExperimentConfig::get_phantom_type(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_phantom_serial_number(self):
        phan_config = super().get_phantom_experiment_config()
        if phan_config is not None:
            if "phantom_serial_number" in phan_config.keys():
                return phan_config["phantom_serial_number"]
        # not found, return a default value
        default_value = "130-0093"
        mu.log("MR-BIASExperimentConfig::phantom_serial_number(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_field_strength_tesla(self):
        phan_config = super().get_phantom_experiment_config()
        if phan_config is not None:
            if "field_strength_tesla" in phan_config.keys():
                return phan_config["field_strength_tesla"]
        # not found, return a default value
        default_value = 3.0
        mu.log("MR-BIASExperimentConfig::field_strength_tesla(): not found in configuration file, "
               "using default value : %0.2f" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_temperature_celsius(self):
        phan_config = super().get_phantom_experiment_config()
        if phan_config is not None:
            if "temperature_celsius" in phan_config.keys():
                return phan_config["temperature_celsius"]
        # not found, return a default value
        default_value = 20.0
        mu.log("MR-BIASExperimentConfig::temperature_celsius(): not found in configuration file, "
               "using default value : %0.2f" % default_value, LogLevels.LOG_WARNING)
        return default_value


class MRIBiasCurveFitConfig(MRIBIASConfiguration):
    def __init__(self, config_filename):
        super().__init__(config_filename)

    def get_averaging(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "averaging" in cf_config.keys():
                return cf_config["averaging"]
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASCurveFitConfig::get_averaging(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_normalisation(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "normalisation" in cf_config.keys():
                return cf_config["normalisation"]
        # not found, return a default value
        default_value = "voxel_max"
        mu.log("MR-BIASCurveFitConfig::get_normalisation(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_exclude(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "exclude" in cf_config.keys():
                return cf_config["exclude"]
        # not found, return a default value
        default_value = "clipped"
        mu.log("MR-BIASCurveFitConfig::get_exclude(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_percent_clipped_threshold(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "percent_clipped_threshold" in cf_config.keys():
                return cf_config["percent_clipped_threshold"]
        # not found, return a default value
        default_value = 200 # no parital clipping
        mu.log("MR-BIASCurveFitConfig::get_percent_clipped_threshold(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value


    def get_t1_vir_models(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vir_options" in cf_config.keys():
                t1_opts = cf_config["t1_vir_options"]
                if "fitting_models" in t1_opts.keys():
                    return t1_opts["fitting_models"]
        # not found, return a default value
        default_value = ["3_param"]
        mu.log("MR-BIASCurveFitConfig::get_t1_vir_models(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vir_exclusion_list(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vir_options" in cf_config.keys():
                t1_opts = cf_config["t1_vir_options"]
                if "inversion_exclusion_list" in t1_opts.keys():
                    return t1_opts["inversion_exclusion_list"]
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASCurveFitConfig::get_t1_vir_exclusion_list(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vir_exclusion_label(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vir_options" in cf_config.keys():
                t1_opts = cf_config["t1_vir_options"]
                if "inversion_exclusion_label" in t1_opts.keys():
                    return t1_opts["inversion_exclusion_label"]
        # not found, return a default valuefcurve
        default_value = "user_IR_excld"
        mu.log("MR-BIASCurveFitConfig::get_t1_vir_exclusion_label(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vfa_models(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vfa_options" in cf_config.keys():
                t1_opts = cf_config["t1_vfa_options"]
                if "fitting_models" in t1_opts.keys():
                    return t1_opts["fitting_models"]
        # not found, return a default value
        default_value = ["2_param"]
        mu.log("MR-BIASCurveFitConfig::t1_vfa_options(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vfa_exclusion_list(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vfa_options" in cf_config.keys():
                t1_opts = cf_config["t1_vfa_options"]
                if "angle_exclusion_list" in t1_opts.keys():
                    return t1_opts["angle_exclusion_list"]
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASCurveFitConfig::get_t1_vfa_exclusion_list(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vfa_exclusion_label(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vfa_options" in cf_config.keys():
                t1_opts = cf_config["t1_vfa_options"]
                if "angle_exclusion_label" in t1_opts.keys():
                    return t1_opts["angle_exclusion_label"]
        # not found, return a default value
        default_value = "user_angle_excld"
        mu.log("MR-BIASCurveFitConfig::get_t1_vfa_exclusion_label(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vfa_2D_roi_setting(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vfa_options" in cf_config.keys():
                t1_opts = cf_config["t1_vfa_options"]
                if "use_2D_roi" in t1_opts.keys():
                    return t1_opts["use_2D_roi"]
        # not found, return a default value
        default_value = False
        mu.log("MR-BIASCurveFitConfig::get_t1_vfa_2D_roi_setting(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t1_vfa_2D_slice_offset(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t1_vfa_options" in cf_config.keys():
                t1_opts = cf_config["t1_vfa_options"]
                if "slice_offset_2D" in t1_opts.keys():
                    return t1_opts["slice_offset_2D"]
        # not found, return a default value
        default_value = 0
        mu.log("MR-BIASCurveFitConfig::get_t1_vfa_2D_slice_offset(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t2_mse_models(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t2_mse_options" in cf_config.keys():
                t2_opts = cf_config["t2_mse_options"]
                if "fitting_models" in t2_opts.keys():
                    return t2_opts["fitting_models"]
        # not found, return a default value
        default_value = ["3_param"]
        mu.log("MR-BIASCurveFitConfig::t2_mse_options(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value


    def get_t2_mse_exclusion_list(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t2_mse_options" in cf_config.keys():
                t2_opts = cf_config["t2_mse_options"]
                if "echo_exclusion_list" in t2_opts.keys():
                    return t2_opts["echo_exclusion_list"]
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASCurveFitConfig::get_t2_mse_exclusion_list(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t2_mse_exclusion_label(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t2_mse_options" in cf_config.keys():
                t2_opts = cf_config["t2_mse_options"]
                if "echo_exclusion_label" in t2_opts.keys():
                    return t2_opts["echo_exclusion_label"]
        # not found, return a default value
        default_value = "user_angle_excld"
        mu.log("MR-BIASCurveFitConfig::get_t2_mse_exclusion_label(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value


    def get_dw_models(self):
            cf_config = super().get_curve_fitting_config()
            if cf_config is not None:
                if "dw_options" in cf_config.keys():
                    dw_opts = cf_config["dw_options"]
                    if "fitting_models" in dw_opts.keys():
                        return dw_opts["fitting_models"]
            # not found, return a default value
            default_value = ["2_param"]
            mu.log("MR-BIASCurveFitConfig::dw_options(): not found in configuration file, "
                "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
            return default_value


    def get_save_voxel_data(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "save_voxel_data" in cf_config.keys():
                return cf_config["save_voxel_data"]
        # not found, return a default value
        default_value = False
        mu.log("MR-BIASCurveFitConfig::get_save_voxel_data(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value

    def get_save_parameter_maps(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "save_parameter_maps" in cf_config.keys():
                return cf_config["save_parameter_maps"]
        # not found, return a default value
        default_value = False
        mu.log("MR-BIASCurveFitConfig::get_save_parameter_maps(): not found in configuration file, "
               "using default value : %s" % default_value, LogLevels.LOG_WARNING)
        return default_value


if __name__ == "__main__":
    main()