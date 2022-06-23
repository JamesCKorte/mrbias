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
"""

import os
from enum import IntEnum
from collections import OrderedDict
from abc import ABC, abstractmethod

import SimpleITK as sitk
import yaml
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
from mrbias.misc_utils import LogLevels
from mrbias.misc_utils import ROI_IDX_LABEL_MAP, T1_ROI_LABEL_IDX_MAP, T2_ROI_LABEL_IDX_MAP
import mrbias.scan_session as scan_session
import mrbias.image_sets as image_sets


# for future expansion / different phantoms
class ROITypeOptions(IntEnum):
    SPHERE = 1

class RegistrationOptions(IntEnum):
    COREL_GRADIENTDESCENT = 1
    MSME_GRIDSEARCH = 2
    TWOSTAGE_MSMEGS_CORELGD = 3


def main():
    # setup the logger to write to file
    mu.initialise_logger("roi_detect.log", force_overwrite=True, write_to_screen=True)
    # setup a pdf to test the pdf reporting
    pdf = mu.PDFSettings()
    c = canvas.Canvas("roi_detect.pdf", landscape(pdf.page_size))

    # target images to test (skyra)
    dcm_dir_a = os.path.join(mu.reference_data_directory(), "mrbias_testset_B")
    ss = scan_session.ScanSessionSiemensSkyra(dcm_dir_a)
    test_geometric_images = ss.get_geometric_images()
    test_hard = test_geometric_images[0]
    test_easy = test_geometric_images[1]
    test_geo_vec = [test_hard, test_easy]
    case_name_vec = ["large miss-alignment", "small miss-alignment"]

    # get the T1 and T2 imagesets
    t1_vir_imagesets = ss.get_t1_vir_image_sets()
    t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
    t2_mse_imagesets = ss.get_t2_mse_image_sets()

    # set the test template
    TEST_TEMPLATE_DIR = os.path.join(mu.reference_template_directory(), "siemens_skyra_3p0T")

    for test_target_im, case_str in zip(test_geo_vec, case_name_vec):
        # create a roi detector
        roi_detector = ROIDetector(test_target_im, TEST_TEMPLATE_DIR)
        # detect the ROIs and return the masks on the target image
        roi_detector.detect()

        # output to pdf
        figure_title = "ROI DETECTOR : %s" % case_str
        roi_detector.write_pdf_summary_page(c, figure_title)

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

    # save the pdf report
    c.save()
    mu.log("------ FIN -------", LogLevels.LOG_INFO)
    plt.show()


class ROITemplate(object):
    def __init__(self, template_dir,
                 dcm_subdir="dicom",
                 t1_rois_file="default_T1_rois.yaml",
                 t2_rois_file="default_T2_rois.yaml"):
        dcm_dir = os.path.join(template_dir, dcm_subdir)
        t1_yaml_file = os.path.join(template_dir, t1_rois_file)
        t2_yaml_file = os.path.join(template_dir, t2_rois_file)
        if not os.path.isdir(dcm_dir):
            mu.log("ROITemplate::__init__(): invalid dicom dir : %s" % dcm_dir, LogLevels.LOG_WARNING)
        if not os.path.isfile(t1_yaml_file):
            mu.log("ROITemplate::__init__(): invalid t1 roi yaml file : %s" % t1_yaml_file, LogLevels.LOG_WARNING)
        if not os.path.isfile(t2_yaml_file):
            mu.log("ROITemplate::__init__(): invalid t2 roi yaml file : %s" % t2_yaml_file, LogLevels.LOG_WARNING)
        # load the image file
        mu.log("ROITemplate::init(): loading template geometry image from DCM dir: %s" %
               dcm_dir, LogLevels.LOG_INFO)
        self.image = mu.get_sitk_image_from_dicom_image_folder(dcm_dir)
        # parse the ROI yaml files
        self.t1_roi_dict = self.parse_t1_yaml(t1_yaml_file)
        self.t2_roi_dict = self.parse_t2_yaml(t2_yaml_file)

    def parse_t1_yaml(self, yaml_file):
        return self.__parse_roi_yaml_file(yaml_file, T1_ROI_LABEL_IDX_MAP)
    def parse_t2_yaml(self, yaml_file):
        return self.__parse_roi_yaml_file(yaml_file, T2_ROI_LABEL_IDX_MAP)
    def __parse_roi_yaml_file(self, yaml_file, roi_label_idx_map):
        roi_dict = OrderedDict()
        in_dict = yaml.full_load(open(yaml_file))
        for roi_label, roi_dx in roi_label_idx_map.items():
            if roi_label in in_dict.keys():
                # found in yaml file
                yaml_roi = in_dict[roi_label]
                if "roi_type" in yaml_roi.keys():
                    roi_type = yaml_roi["roi_type"]
                    if roi_type == "sphere":
                        # check all sphere fields are available and create Spherical ROI
                        for field in ['roi_radius_mm', 'ctr_vox_coords']:
                            if not (field in yaml_roi.keys()):
                                mu.log("ROITemplate::__parse_roi_yaml_file(): skipping ROI(%s) no expected field '%s' "
                                       "specified in yaml file : %s" % (roi_label, field, yaml_file), LogLevels.LOG_WARNING)
                        roi_radius_mm = yaml_roi["roi_radius_mm"]
                        ctr_vox_coords = yaml_roi["ctr_vox_coords"]
                        assert isinstance(roi_radius_mm, float), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "roi_radius_mm expected datatype is float (not %s)" % \
                                                                 type(roi_radius_mm)
                        assert isinstance(ctr_vox_coords, list), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "ctr_vox_coords expected datatype is list (not %s)" % \
                                                                 type(ctr_vox_coords)
                        roi_dict[roi_dx] = ROISphere(roi_label, roi_dx, ctr_vox_coords, roi_radius_mm)
                else:
                    mu.log("ROITemplate::__parse_roi_yaml_file(): skipping ROI(%s) no field 'roi_type' "
                           "specified in yaml file : %s" % (roi_label, yaml_file), LogLevels.LOG_WARNING)
            else:
                mu.log("ROITemplate::__parse_roi_yaml_file(): ROI(%s) not specified in yaml file : %s" %
                       (roi_label, yaml_file), LogLevels.LOG_WARNING)
        # return the ROI dictionary
        return roi_dict

    def get_T1_mask_image(self):
        return self.__get_mask_image(self.t1_roi_dict)
    def get_T2_mask_image(self):
        return self.__get_mask_image(self.t2_roi_dict)
    def __get_mask_image(self, roi_dict):
        """
        Create a template mask image of the same image grid as the template image
        Args:
            roi_dict (OrderedDict): with a roi_idx key and values of type ROI (only supports ROISphere)
        Returns:
            SimpleITK.Image: a template mask with 0 as background and ROIs with values as defined in points
        """
        # Create a numpy image array of zeros with the same size as the geometric image
        fixed_geo_arr = sitk.GetArrayFromImage(self.image)
        fixed_geo_spacing = np.array(self.image.GetSpacing())
        mask_arr = np.zeros_like(fixed_geo_arr)
        for roi_dx, roi in roi_dict.items():
            # rely on the concrete class to draw its own ROI on the mask image
            roi.draw(mask_arr, fixed_geo_spacing)
        # Convert the roi mask array to a simpleITK image with matched spatial properties to the
        # fixed geometric image
        masked_image = sitk.GetImageFromArray(mask_arr)
        masked_image.SetOrigin(self.image.GetOrigin())
        masked_image.SetDirection(self.image.GetDirection())
        masked_image.SetSpacing(self.image.GetSpacing())
        return masked_image

    def get_t1_slice_dx(self):
        slice_vec = []
        for roi in self.t1_roi_dict.values():
            slice_vec.append(roi.get_slice_dx())
        return int(np.median(np.array(slice_vec)))

    def get_t2_slice_dx(self):
        slice_vec = []
        for roi in self.t2_roi_dict.values():
            slice_vec.append(roi.get_slice_dx())
        return int(np.median(np.array(slice_vec)))

    def get_t1_roi_values(self):
        return list(T1_ROI_LABEL_IDX_MAP.values())
    def get_t2_roi_values(self):
        return list(T2_ROI_LABEL_IDX_MAP.values())



class ROI(ABC):
    def __init__(self, label, roi_index):
        self.label = label
        self.roi_index = roi_index
        assert self.roi_index in ROI_IDX_LABEL_MAP.keys(), "ROI::__init__: roi index is invalid, idx=%d" % self.roi_index

    """
    Draw the ROI into the passes mask array, marking the ROI with the global ROI index
    """
    @abstractmethod
    def draw(self, arr, spacing):
        return None

    @abstractmethod
    def get_slice_dx(self):
        return None

class ROISphere(ROI):
    def __init__(self, label, roi_index,
                 ctr_vox_coords, radius_mm):
        super().__init__(label, roi_index)
        self.ctr_vox_coords = ctr_vox_coords
        self.radius_mm = radius_mm
        mu.log("\t\tROISphere::__init__(): sphere (%d : %s) created!" % (roi_index, label), LogLevels.LOG_INFO)

    def draw(self, arr, spacing):
        # calculate how many voxels to achieve the radius
        radius_vox = self.radius_mm / spacing
        z, y, x = np.ogrid[:arr.shape[2],:arr.shape[1],:arr.shape[0]]
        # Assigns the masked pixels in the copy image array to corresponding pixel values
        # TODO: validate ordering of radius_vox [0,1,2] with an image which is less isotropic
        distance_from_centre = np.sqrt(((x - self.ctr_vox_coords[0]) / radius_vox[0]) ** 2 +
                                       ((y - self.ctr_vox_coords[1]) / radius_vox[1]) ** 2 +
                                       ((z - self.ctr_vox_coords[2]) / radius_vox[2]) ** 2)
        sphere_mask = distance_from_centre <= 1.0
        arr[sphere_mask] = self.roi_index

    def get_slice_dx(self):
        return self.ctr_vox_coords[2]


class ROIDetector(object):
    """
    A class for detecting regions of interest (ROIs) on a 3D T1 weighted image of the ISMRM/NIST qMRI System Phantom.

    The target image is registered to a template image which has associated template ROIs. The ROIs are then warped
    from the template image space onto the target image space.

    Attributes:
        target_geo_im (ImageSet.ImageGeometric): the target image on which we want to detect ROIs
        reg_method (RegistrationMethodAbstract): the registration method to align the target and template images
        fixed_geom_im (SimpleITK.Image): the template T1 weighted 3D MRI image
        transform (SimpleITK.Transform): the estimated transform between the target image and template image
    """

    def __init__(self,
                 target_geometry_image,
                 template_directory,
                 registration_method=RegistrationOptions.TWOSTAGE_MSMEGS_CORELGD):
        """
        Class constructor stores the target image, registration method choice and loads the template image.

        Args:
            target_geometry_image (ImageSet.ImageGeometric): the target image on which we want to detect ROIs
            template_directory (directory): a template directory with a DICOM folder and ROI yaml files
            registration_method (RegistrationOptions): a flag representing the registration method to use
        """
        assert isinstance(target_geometry_image, image_sets.ImageGeometric), \
            "ROIDetector::init(): taget_geometry_image is expected as a SimpleITK image (not %s)" % \
            type(target_geometry_image)
        self.target_geo_im = target_geometry_image
        self.reg_method = registration_method
        self.roi_template = ROITemplate(template_directory)
        self.fixed_geom_im = self.roi_template.image
        assert isinstance(self.fixed_geom_im, sitk.Image), \
            "ROIDetector::init(): self.fixed_geom_im (template image) is expected as a SimpleITK image (not %s)" % \
            type(self.fixed_geom_im)
        self.transform = None

    def detect(self):
        """ Detect the ROIs on the target image, by registering the target image to template image."""
        # create relevant registration class
        # rego = None
        # if self.reg_method == RegistrationOptions.COREL_GRADIENTDESCENT:
        #     rego = RegistrationCorrelationGradientDescent(self.fixed_geom_im, self.target_geo_im)
        # elif self.reg_method == RegistrationOptions.MSME_GRIDSEARCH:
        #     rego = RegistrationMSMEGridSearch(self.fixed_geom_im, self.target_geo_im)
        # assert rego is not None, "ROIDetector::detect(): invalid registration method, %s" % str(self.reg_method)
        rego = RegistrationMethodAbstract.generate_registration_instance(self.reg_method,
                                                                         self.fixed_geom_im,
                                                                         self.target_geo_im.get_image())

        mu.log("ROIDetector::detect():",
               LogLevels.LOG_INFO)
        mu.log("\t Iteration |   Metric   |   Optimizer Position",
               LogLevels.LOG_INFO)
        transform, metric = rego.register()
        # store the transform for warping ROIs
        self.transform = transform
        # store the resulting masks on the geometric image
        self.target_geo_im.set_T1_mask(self.get_detected_T1_mask())
        self.target_geo_im.set_T2_mask(self.get_detected_T2_mask())

    def get_detected_T1_mask(self):
        """
        Returns:
            SimpleITK.Image: T1 mask in target image space, with 0 for background and 1-14 for detected ROIs
        """
        mu.log("ROIDetector::get_detected_T1_mask()", LogLevels.LOG_INFO)
        fixed_t1_mask_im = self.get_fixed_T1_mask()
        return self.__get_registered_mask(fixed_t1_mask_im, self.target_geo_im.get_image())

    def get_detected_T2_mask(self):
        """
        Returns:
            SimpleITK.Image: T2 mask  in target image space, with 0 for background and 15-28 for detected ROIs
        """
        mu.log("ROIDetector::get_detected_T2_mask()", LogLevels.LOG_INFO)
        fixed_t2_mask_im = self.get_fixed_T2_mask()
        return self.__get_registered_mask(fixed_t2_mask_im, self.target_geo_im.get_image())

    def get_fixed_T1_mask(self):
        """
        Returns:
            SimpleITK.Image: T1 mask in template image space, with 0 for background and 1-14 for detected ROIs
        """
        mu.log("ROIDetector::get_fixed_T1_mask()", LogLevels.LOG_INFO)
        return self.roi_template.get_T1_mask_image()

    def get_fixed_T2_mask(self):
        """
        Returns:
            SimpleITK.Image: T2 mask  in template image space, with 0 for background and 15-28 for detected ROIs
        """
        mu.log("ROIDetector::get_fixed_T2_mask()", LogLevels.LOG_INFO)
        return self.roi_template.get_T2_mask_image()

    def visualise_fixed_T1_rois(self, ax=None):
        geo_arr = sitk.GetArrayFromImage(self.fixed_geom_im)
        roi_arr = sitk.GetArrayFromImage(self.get_fixed_T1_mask())
        t1_slice_dx = self.roi_template.get_t1_slice_dx()
        t1_roi_values = self.roi_template.get_t1_roi_values()
        self.__visualise_rois(geo_arr, roi_arr, t1_slice_dx, t1_roi_values, ax,
                              title="T1 (template)")

    def visualise_fixed_T2_rois(self, ax=None):
        geo_arr = sitk.GetArrayFromImage(self.fixed_geom_im)
        roi_arr = sitk.GetArrayFromImage(self.get_fixed_T2_mask())
        t2_slice_dx = self.roi_template.get_t2_slice_dx()
        t2_roi_values = self.roi_template.get_t2_roi_values()
        self.__visualise_rois(geo_arr, roi_arr, t2_slice_dx, t2_roi_values, ax,
                              title="T2 (template)")

    def visualise_detected_T1_rois(self, ax=None):
        self.__visualise_transformed_rois(self.get_detected_T1_mask(),
                                          self.roi_template.get_t1_slice_dx(),
                                          self.roi_template.get_t1_roi_values(), ax,
                                          title="T1 (detected)")

    def visualise_detected_T2_rois(self, ax=None):
        self.__visualise_transformed_rois(self.get_detected_T2_mask(),
                                          self.roi_template.get_t2_slice_dx(),
                                          self.roi_template.get_t2_roi_values(), ax,
                                          title="T2 (detected)")

    def visualise_T1_registration(self, pre_reg_ax=None, post_reg_ax=None, invert_moving=True):
        self.__visualise_registration(self.roi_template.get_t1_slice_dx(),
                                      pre_reg_ax=pre_reg_ax, post_reg_ax=post_reg_ax,
                                      invert_moving=invert_moving,
                                      title="T1")

    def visualise_T2_registration(self, pre_reg_ax=None, post_reg_ax=None, invert_moving=True):
        self.__visualise_registration(self.roi_template.get_t2_slice_dx(),
                                      pre_reg_ax=pre_reg_ax, post_reg_ax=post_reg_ax,
                                      invert_moving=invert_moving,
                                      title="T2")

    # same output as log but to pdf
    def write_pdf_summary_page(self, c, sup_title="ROI Detection: Summary"):
        table_width = 170
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font

        # draw the summary figure
        # -----------------------------------------------------------
        # setup figure
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
        if sup_title is not None:
            f.suptitle(sup_title)
        f.set_size_inches(14, 6)
        # draw the template rois on the template image
        self.visualise_fixed_T1_rois(ax=ax1)
        self.visualise_fixed_T2_rois(ax=ax5)
        # draw the registration (pre/post)
        self.visualise_T1_registration(invert_moving=False, pre_reg_ax=ax2, post_reg_ax=ax3)
        self.visualise_T2_registration(invert_moving=False, pre_reg_ax=ax6, post_reg_ax=ax7)
        # visualise the transfromed ROI masks on the target image
        self.visualise_detected_T1_rois(ax=ax4)
        self.visualise_detected_T2_rois(ax=ax8)

        # draw it on the pdf
        pil_f = mu.mplcanvas_to_pil(f)
        width, height = pil_f.size
        height_3d, width_3d = pdf.page_width * (height / width), pdf.page_width
        c.drawImage(ImageReader(pil_f),
                    0,
                    pdf.page_height - pdf.top_margin - height_3d - pdf.line_width,
                    width_3d,
                    height_3d)
        plt.close(f)

        c.showPage()  # new page

    def __visualise_registration(self, slice_dx, pre_reg_ax=None, post_reg_ax=None, invert_moving=True, title=None):
        """
        Visualise the image registration with a checkerboard of the fixed and moving images. This function creates
        two checkerboard comparisons; the first with the unregistered target image and the second with the registered
        target image.

        Args:
            slice_dx (int): index of slice to display
            pre_reg_ax (matplotlib.axes): axes to plot the target image prior to registration
            post_reg_ax (matplotlib.axes): axes to plot the target image after registration
            invert_moving (bool): invert the image intensity of the moving image
            title (string): a title for the subplots
        """
        # fake a target geo image to be on same spatial grid as fixed_geo_im for checkerboard filter
        target_geo_im = sitk.GetImageFromArray(sitk.GetArrayFromImage(self.target_geo_im.get_image()))
        target_geo_im.SetOrigin(self.fixed_geom_im.GetOrigin())
        target_geo_im.SetDirection(self.fixed_geom_im.GetDirection())
        target_geo_im.SetSpacing(self.fixed_geom_im.GetSpacing())

        # define intensity limits for visualisation
        t1_im_slice = sitk.GetArrayFromImage(target_geo_im)[slice_dx, :, :]
        vmin = np.mean(t1_im_slice) - 1.0 * np.std(t1_im_slice)
        vmax = np.mean(t1_im_slice) + 2.0 * np.std(t1_im_slice)
        inversion_max = np.mean(t1_im_slice) + 2.5 * np.std(t1_im_slice)

        # pre-registration
        if invert_moving:
            target_geo_im = sitk.InvertIntensity(target_geo_im, maximum=inversion_max)
        target_geo_im_match_type = sitk.Cast(target_geo_im, sitk.sitkUInt16)
        checker_im = sitk.CheckerBoard(self.fixed_geom_im, target_geo_im_match_type, [4, 4, 1])
        checker_arr = sitk.GetArrayFromImage(checker_im)

        # post-registration
        rotated_fixed_im = self.__get_rotated_sampling_im()
        # resample the target image and mask - onto the deformed fixed space
        # (so we can use the reference ROI slice for visualisation)
        target_im = sitk.Resample(self.target_geo_im.get_image(),
                                  rotated_fixed_im,
                                  sitk.Euler3DTransform(), sitk.sitkLinear)
        if invert_moving:
            target_im = sitk.InvertIntensity(target_im, maximum=inversion_max)

        target_im_match_type = sitk.Cast(target_im, sitk.sitkUInt16)
        checker_im_reg = sitk.CheckerBoard(rotated_fixed_im, target_im_match_type, [4, 4, 1])
        checker_arr_reg = sitk.GetArrayFromImage(checker_im_reg)
        if (pre_reg_ax is None) or (post_reg_ax is None):
            f, (pre_reg_ax, post_reg_ax) = plt.subplots(1, 2)
        im_slice = checker_arr[slice_dx, :, :]
        pre_reg_ax.imshow(im_slice, cmap='gray', vmin=vmin, vmax=vmax)
        im_slice = checker_arr_reg[slice_dx, :, :]
        post_reg_ax.imshow(im_slice, cmap='gray', vmin=vmin, vmax=vmax)
        for ax in [pre_reg_ax, post_reg_ax]:
            ax.axis('off')
        if title is not None:
            pre_reg_ax.set_title("%s un-registered" % title)
            post_reg_ax.set_title("%s registered" % title)
        #plt.pause(0.01)


    def __get_registered_mask(self, src_mask, target_im):
        """
        Warp and resample a template mask onto the target image.

        Args:
            src_mask (SimpleITK.Image): a template mask to warp and resample
            target_im (SimpleITK.Image): a target image to specify the resampling grid

        Returns:
            SimpleITK.Image: the template mask warped and resampled into the target image space
        """
        if self.transform is None:
            mu.log("\tROIDetector::__get_registered_mask() : no transform found, need to run detect() function",
                   LogLevels.LOG_WARNING)
            return None
        return sitk.Resample(src_mask, target_im, sitk.Transform.GetInverse(self.transform), sitk.sitkNearestNeighbor)


    def __get_rotated_sampling_im(self):
        """
        Transform the template image into the target image space. This is useful for visualisation, was the template
        landmarks such as a reference T1 or T2 slice can then be sampled in the estimated/registered target space.

        Returns:
            SimpleITK.Image: The template image transformed (but not resampled) into the target image space
        """
        if self.transform is None:
            mu.log("ROIDetector::__get_rotated_sampling_im() : no transform found, need to run detect() function",
                   LogLevels.LOG_WARNING)
            return None
        # transform the reference geometry into the target space
        inv_transform = self.transform  # sitk.Transform.GetInverse(self.transform)
        # deep copy the reference image
        rotated_geom_im = sitk.GetImageFromArray(sitk.GetArrayFromImage(self.fixed_geom_im))
        rotated_geom_im.SetSpacing(self.fixed_geom_im.GetSpacing())
        rotated_geom_im.SetOrigin(self.fixed_geom_im.GetOrigin())
        rotated_geom_im.SetDirection(self.fixed_geom_im.GetDirection())
        # transform the origin
        o_new = inv_transform.TransformPoint(rotated_geom_im.GetOrigin())
        # transform the direction matrix
        d = rotated_geom_im.GetDirection()
        d_0 = [d[0], d[3], d[6]]
        d_1 = [d[1], d[4], d[7]]
        d_2 = [d[2], d[5], d[8]]
        d_0_rot = inv_transform.TransformVector(d_0, (0.0, 0.0, 0.0))
        d_1_rot = inv_transform.TransformVector(d_1, (0.0, 0.0, 0.0))
        d_2_rot = inv_transform.TransformVector(d_2, (0.0, 0.0, 0.0))
        d_new = np.zeros_like(d)
        d_new[0] = d_0_rot[0]
        d_new[1] = d_1_rot[0]
        d_new[2] = d_2_rot[0]
        d_new[3] = d_0_rot[1]
        d_new[4] = d_1_rot[1]
        d_new[5] = d_2_rot[1]
        d_new[6] = d_0_rot[2]
        d_new[7] = d_1_rot[2]
        d_new[8] = d_2_rot[2]
        # debug (remove)
        # print("origin: ", rotated_geom_im.GetOrigin())
        # print("direction: ", rotated_geom_im.GetDirection())
        # print("origin (new):", o_new)
        # print("direction (new): ", d_new)
        # re-orient and locate the image for resampling
        rotated_geom_im.SetOrigin(o_new)
        rotated_geom_im.SetDirection(d_new)
        return rotated_geom_im

    def __visualise_transformed_rois(self, mask_im,
                                     slice_dx, roi_list,
                                     ax=None,
                                     title=None):
        if self.transform is None:
            mu.log("ROIDetector::visualise_detected_T1_rois() : no transform found, need to run detect() function",
                   LogLevels.LOG_WARNING)
            return None
        rotated_geom_im = self.__get_rotated_sampling_im()
        # resample the target image and mask - onto the deformed fixed space
        # (so we can use the reference ROI slice for visualisation)
        geo_arr = sitk.GetArrayFromImage(sitk.Resample(self.target_geo_im.get_image(), rotated_geom_im,
                                                       sitk.Euler3DTransform(), sitk.sitkLinear))
        roi_arr = sitk.GetArrayFromImage(sitk.Resample(mask_im, rotated_geom_im,
                                                       sitk.Euler3DTransform(), sitk.sitkNearestNeighbor))
        self.__visualise_rois(geo_arr, roi_arr, slice_dx, roi_list, ax, title)


    def __visualise_rois(self, im_arr, roi_arr,
                         slice_dx, roi_list,
                         ax=None,
                         title=None):
        """
        Show a greyscale image and overlay coloured ROIs.

        Args:
            im_arr (numpy.array): a 3D image array
            roi_arr (numpy.array): a 3D roi array (0 as background, non-zero as regions of interest)
            slice_dx (int): an index of the 2D slice / image to display (0th dimension of the image and roi arrays)
            roi_list (list): a list of ROI indexes in the ROI array
            ax (matplotlib.axes): the axes to draw the image and ROI overlay
            title (string): a title to label the axes
        """
        if ax is None:
            f, ax = plt.subplots(1, 1)
        roi_slice = roi_arr[slice_dx, :, :]
        im_slice = im_arr[slice_dx, :, :]
        ax.imshow(im_slice, cmap='gray',
                  vmin=np.mean(im_slice)-1.0*np.std(im_slice),
                  vmax=np.mean(im_slice)+2.0*np.std(im_slice))
        i = ax.imshow(np.ma.masked_where(roi_slice == 0, roi_slice),
                      cmap='nipy_spectral', vmin=np.min(roi_list)-1, vmax=np.max(roi_list)+1,
                      interpolation='none',
                      alpha=0.7)
        ax.axis('off')
        ticks = list(range(np.min(roi_list), np.max(roi_list)+1))
        ticklabels = [ROI_IDX_LABEL_MAP[x] for x in ticks]
        cb = plt.colorbar(mappable=i, ax=ax,
                     ticks=ticks)
        cb.set_ticklabels(ticklabels=ticklabels)
        if title is not None:
            ax.set_title(title)
        #plt.pause(0.01)




class RegistrationMethodAbstract(ABC):
    def __init__(self, fixed_image, moving_image):
        self.fixed_im = fixed_image
        self.moving_im = moving_image

    @abstractmethod
    def register(self):
        return None, None

    @staticmethod
    def generate_registration_instance(reg_method, fixed_geom_im, target_geo_im):
        rego = None
        if reg_method == RegistrationOptions.COREL_GRADIENTDESCENT:
            rego = RegistrationCorrelationGradientDescent(fixed_geom_im, target_geo_im)
        elif reg_method == RegistrationOptions.MSME_GRIDSEARCH:
            rego = RegistrationMSMEGridSearch(fixed_geom_im, target_geo_im)
        elif reg_method == RegistrationOptions.TWOSTAGE_MSMEGS_CORELGD:
            rego = RegistrationTwoStage(fixed_geom_im, target_geo_im,
                                        RegistrationOptions.MSME_GRIDSEARCH,
                                        RegistrationOptions.COREL_GRADIENTDESCENT)
        assert rego is not None, "RegistrationMethodAbstract::generate_registration_instance(): " \
                                 "invalid registration method, %s" % str(reg_method)
        return rego

    def print_iteration_info(self, method):
        mu.log("\t\t{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                                    method.GetMetricValue(),
                                                    method.GetOptimizerPosition()),
               LogLevels.LOG_INFO)



class RegistrationTwoStage(RegistrationMethodAbstract):
    def __init__(self, fixed_image, moving_image,
                 stage_a_registration_method=RegistrationOptions.MSME_GRIDSEARCH,
                 stage_b_registration_method=RegistrationOptions.COREL_GRADIENTDESCENT):
        super().__init__(fixed_image, moving_image)
        mu.log("RegistrationTwoStage::init()", LogLevels.LOG_INFO)
        self.stage_a_registration_method = stage_a_registration_method
        self.stage_b_registration_method = stage_b_registration_method
        self.stage_a_rego = RegistrationMethodAbstract.generate_registration_instance(stage_a_registration_method,
                                                                                      fixed_image, moving_image)
        self.stage_b_rego = RegistrationMethodAbstract.generate_registration_instance(stage_b_registration_method,
                                                                                      fixed_image, moving_image)

    def register(self):
        # stage A registration
        trans_a, metric_a = self.stage_a_rego.register()
        trans_image_a = sitk.Resample(self.moving_im, self.fixed_im, trans_a, sitk.sitkLinear)
        # stage B registration
        self.stage_b_rego = RegistrationMethodAbstract.generate_registration_instance(self.stage_b_registration_method,
                                                                                      self.fixed_im, trans_image_a)
        trans_b, metric_b = self.stage_b_rego.register()
        # compose the final transform
        combined_transform = None
        if hasattr(sitk, "CompositeTransform"):
            combined_transform = sitk.CompositeTransform(trans_a)
        elif hasattr(sitk, "Transform"):
            combined_transform = sitk.Transform(trans_a)
        else:
            mu.log("RegistrationTwoStage::register(): unable to locate simpleITK multi-stage transform", LogLevels.LOG_ERROR)
            assert False
        combined_transform.AddTransform(trans_b)
        return combined_transform, metric_b



class RegistrationCorrelationGradientDescent(RegistrationMethodAbstract):
    def __init__(self, fixed_image, moving_image):
        super().__init__(fixed_image, moving_image)
        mu.log("RegistrationCorrelationGradientDescent::init()", LogLevels.LOG_INFO)

    def register(self):
        moving = sitk.Cast(self.moving_im, sitk.sitkFloat32)
        fixed = sitk.Cast(self.fixed_im, sitk.sitkFloat32)
        # setup the registration method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingPercentage(0.01)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                   numberOfIterations=1000,
                                                   minStep=0.000001)
        R.SetOptimizerScalesFromPhysicalShift()
        # centre as initial transform
        initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        # hook up a logger
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.print_iteration_info(R))
        # register
        final_transform = R.Execute(fixed, moving)
        return final_transform, R.GetMetricValue()


class RegistrationMSMEGridSearch(RegistrationMethodAbstract):
    def __init__(self, fixed_image, moving_image):
        super().__init__(fixed_image, moving_image)
        mu.log("RegistrationMSMEGridSearch::init()", LogLevels.LOG_INFO)

    def register(self):
        moving = sitk.Cast(self.moving_im, sitk.sitkFloat32)
        fixed = sitk.Cast(self.fixed_im, sitk.sitkFloat32)
        # setup the registration method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetMetricSamplingPercentage(0.001)
        R.SetMetricSamplingStrategy(R.REGULAR)
        # Number of samples for each rotational axis, 360 divided by this number gives you the angle 'incremented'
        sample_per_axis = 24
        # Note: Order of parameters is ( x-rotation, y-rotation, z-rotation, x, y, z)
        R.SetOptimizerAsExhaustive([sample_per_axis // 2, sample_per_axis // 2, sample_per_axis // 4,
                                    0, 0, 0])
        R.SetOptimizerScales(
            [2.0 * np.pi / sample_per_axis, 2.0 * np.pi / sample_per_axis, 2.0 * np.pi / sample_per_axis,
             1.0, 1.0, 1.0])
        # centre as initial transform
        initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        # hook up a logger (too much output and won't show convergance as it is a grid search)
        # R.AddCommand(sitk.sitkIterationEvent, lambda: self.print_iteration_info(R))
        # register
        final_transform = R.Execute(fixed, moving)
        # output the best transform of the grid search
        mu.log("\t\t{0:3} = {1:10.5f} : {2}".format(R.GetOptimizerIteration(),
                                                    R.GetMetricValue(),
                                                    final_transform),
               LogLevels.LOG_INFO)
        return final_transform, R.GetMetricValue()







if __name__ == "__main__":
    main()
