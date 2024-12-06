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
  21-September-2024  :               (James Korte) : Refactoring   MR-BIAS v1.0
"""
import os
import copy
from enum import IntEnum
from collections import OrderedDict
from abc import ABC, abstractmethod

import SimpleITK as sitk
import pandas as pd
import scipy.ndimage
import scipy.signal as scisig # to find local minima in metric grid search
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
from mrbias.misc_utils import LogLevels, OrientationOptions
from mrbias.misc_utils import ROI_IDX_LABEL_MAP, T1_ROI_LABEL_IDX_MAP, T2_ROI_LABEL_IDX_MAP, DW_ROI_LABEL_IDX_MAP
import mrbias.scan_session as scan_session
import mrbias.image_sets as image_sets


ROI_DETECT_SEED = 123456789 # to make the SimpleITK random sampling reproducable (sitk default uses system clock)

class DetectionOptions(IntEnum):
    NONE = 0
    COREL_GRADIENTDESCENT = 1
    MSE_GRIDSEARCH = 2
    TWOSTAGE_MSEGS_CORELGD = 3
    MMI_GRADIENTDESCENT = 4
    COREL_AXIGRID_NBEST_GRADIENTDESCENT = 5
    SHAPE_DIFFUSION_NIST = 6


"""
target_geo_im (ImageSet.ImageGeometric): the target image on which we want to detect ROIs
fixed_geom_im (SimpleITK.Image): the template T1 weighted 3D MRI image
"""
class DetectionMethodAbstract(ABC):
    def __init__(self, target_geo_im, fixed_geom_im, moving_im, roi_template):
        self.target_geo_im = target_geo_im
        self.fixed_im = fixed_geom_im
        self.moving_im = moving_im
        self.roi_template = roi_template

    @abstractmethod
    def detect(self):
        return None
    @abstractmethod
    def get_detected_T1_mask(self):
        return None, None # mask, prop_dict
    @abstractmethod
    def get_detected_T2_mask(self):
        return None, None # mask, prop_dict
    @abstractmethod
    def get_detected_DW_mask(self):
        return None, None # mask, prop_dict
    @abstractmethod
    def get_image_transform(self):
        return None
    @abstractmethod
    def get_image_transform_matrix(self, fixed_to_moving=True):
        return None
    @abstractmethod
    def write_pdf_summary_page(self, c, sup_title=None):
        return None

    def get_fixed_T1_mask(self):
        """
        Returns:
            SimpleITK.Image: T1 mask in template image space, with 0 for background and 1-14 for detected ROIs
        """
        mu.log("DetectionMethodAbstract::get_fixed_T1_mask()", LogLevels.LOG_INFO)
        return self.roi_template.get_T1_mask_image()

    def get_fixed_T2_mask(self):
        """
        Returns:
            SimpleITK.Image: T2 mask  in template image space, with 0 for background and 15-28 for detected ROIs
        """
        mu.log("DetectionMethodAbstract::get_fixed_T2_mask()", LogLevels.LOG_INFO)
        return self.roi_template.get_T2_mask_image()

    def get_fixed_DW_mask(self):
        """
        Returns:
            SimpleITK.Image: DW mask  in template image space, with 0 for background and 29-41 for detected ROIs
        """
        mu.log("DetectionMethodAbstract::get_fixed_DW_mask()", LogLevels.LOG_INFO)
        return self.roi_template.get_DW_mask_image()

    def visualise_fixed_T1_rois(self, ax=None):
        t1_fixed_im_mask, prop_im_dict = self.get_fixed_T1_mask()
        t1_slice_dx = self.roi_template.get_t1_slice_dx()
        t1_roi_values = list(T1_ROI_LABEL_IDX_MAP.values())
        self.__visualise_rois(self.fixed_im, t1_fixed_im_mask, t1_slice_dx, t1_roi_values, ax=ax,
                              title="T1 (template)")

    def visualise_fixed_T2_rois(self, ax=None):
        t2_fixed_im_mask, prop_im_dict = self.get_fixed_T2_mask()
        t2_slice_dx = self.roi_template.get_t2_slice_dx()
        t2_roi_values = list(T2_ROI_LABEL_IDX_MAP.values())
        self.__visualise_rois(self.fixed_im, t2_fixed_im_mask, t2_slice_dx, t2_roi_values, ax=ax,
                              title="T2 (template)")

    def visualise_fixed_DW_rois(self, ax=None, slice_orient=OrientationOptions.AXI):
        dw_fixed_im_mask, prop_im_dict = self.get_fixed_DW_mask()
        dw_slice_dx = self.roi_template.get_dw_slice_dx(slice_orient)
        dw_roi_values = list(DW_ROI_LABEL_IDX_MAP.values())
        self.__visualise_rois(self.fixed_im, dw_fixed_im_mask, dw_slice_dx, dw_roi_values, slice_orient, ax=ax,
                              title="DW (template)")

    def visualise_detected_T1_rois(self, ax=None):
        t1_roi_values = list(T1_ROI_LABEL_IDX_MAP.values())
        t1_mask, t1_prop_dict = self.get_detected_T1_mask()
        self.__visualise_transformed_rois(t1_mask,
                                          self.roi_template.get_t1_slice_dx(),
                                          t1_roi_values, ax=ax,
                                          title="T1 (detected)")

    def visualise_detected_T2_rois(self, ax=None):
        t2_roi_values = list(T2_ROI_LABEL_IDX_MAP.values())
        t2_mask, t2_prop_dict = self.get_detected_T2_mask()
        self.__visualise_transformed_rois(t2_mask,
                                          self.roi_template.get_t2_slice_dx(),
                                          t2_roi_values, ax=ax,
                                          title="T2 (detected)")

    def visualise_detected_DW_rois(self, ax=None, slice_orient=OrientationOptions.AXI, in_template_space=True):
        dw_roi_values = list(DW_ROI_LABEL_IDX_MAP.values())
        dw_mask, dw_prop_dict = self.get_detected_DW_mask()
        if in_template_space:
            self.__visualise_transformed_rois(dw_mask,
                                              self.roi_template.get_dw_slice_dx(slice_orient),
                                              dw_roi_values, slice_orient,
                                              ax=ax,
                                              title="DW (detected)\n[template space]")
        else:
            im_tx = self.get_image_transform()
            if im_tx is None:
                mu.log(
                    "DetectionMethodAbstract::visualise_detected_DW_rois() : no transform found, need to run detect() function",
                    LogLevels.LOG_WARNING)
                return None
            # calculate the centroid in the moving image space (from the template)
            fixed_ctr_dxs = self.roi_template.get_dw_centroid_dx()
            fixed_ctr_dxs = [int(fixed_ctr_dxs[2]), int(fixed_ctr_dxs[1]), int(fixed_ctr_dxs[0])]
            ctr_mm = np.array(self.fixed_im.TransformIndexToPhysicalPoint(fixed_ctr_dxs))
            ctr_mm = np.array([ctr_mm[0], ctr_mm[1], ctr_mm[2], 1.0])
            tx_matrix = self.get_image_transform_matrix(fixed_to_moving=True)
            ctr_mov_mm = tx_matrix.dot(ctr_mm)[0:3]
            moving_ctr_dxs = np.array(self.moving_im.TransformPhysicalPointToIndex(ctr_mov_mm))
            dw_slice_dx = None
            if slice_orient == OrientationOptions.AXI:
                dw_slice_dx = moving_ctr_dxs[2]
            if slice_orient == OrientationOptions.COR:
                dw_slice_dx = moving_ctr_dxs[1]
            if slice_orient == OrientationOptions.SAG:
                dw_slice_dx = moving_ctr_dxs[0]
            self.__visualise_rois(self.moving_im, dw_mask, dw_slice_dx, dw_roi_values, slice_orient, ax=ax,
                                  title="DW (detected)\n[target space]")


    def __visualise_transformed_rois(self, mask_im,
                                     slice_dx, roi_list,
                                     slice_orient=OrientationOptions.AXI,
                                     ax=None,
                                     title=None):
        im_tx = self.get_image_transform()
        if im_tx is None:
            mu.log("DetectionMethodAbstract::visualise_detected_T1_rois() : no transform found, need to run detect() function",
                   LogLevels.LOG_WARNING)
            return None
        rotated_geom_im = self._get_rotated_sampling_im()
        # resample the target image and mask - onto the deformed fixed space
        # (so we can use the reference ROI slice for visualisation)
        geo_im = sitk.Resample(self.target_geo_im.get_image(), rotated_geom_im,
                               sitk.Euler3DTransform(), sitk.sitkLinear)
        roi_im = sitk.Resample(mask_im, rotated_geom_im,
                               sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        self.__visualise_rois(geo_im, roi_im, slice_dx, roi_list, slice_orient, ax=ax, title=title)

    def __visualise_rois(self, im, roi_map,
                         slice_dx, roi_list,
                         slice_orient=OrientationOptions.AXI,
                         ax=None,
                         title=None,
                         roi_outline_only=True):
        """
        Show a greyscale image and overlay coloured ROIs.

        Args:
            im_arr (SimpleITK.Image): a 3D image
            roi_arr (SimpleITK.Image): a 3D roi map (0 as background, non-zero as regions of interest)
            slice_dx (int): an index of the 2D slice / image to display (0th dimension of the image and roi arrays)
            roi_list (list): a list of ROI indexes in the ROI array
            ax (matplotlib.axes): the axes to draw the image and ROI overlay
            title (string): a title to label the axes
        """
        # get arrays from images
        im_arr = sitk.GetArrayFromImage(im)
        roi_arr = sitk.GetArrayFromImage(roi_map)
        # get image spacing
        im_spacing = np.flip(np.array(im.GetSpacing()))
        # pull out the relevant slice based on orientation
        roi_slice, im_slice, extent = None, None, [0, 0, 0, 0]
        if slice_orient == OrientationOptions.AXI:
            roi_slice = roi_arr[slice_dx, :, :]
            im_slice = im_arr[slice_dx, :, :]
            extent = [0, im_arr.shape[2]*im_spacing[2], 0, im_arr.shape[1]*im_spacing[1]]
        elif slice_orient == OrientationOptions.COR:
            roi_slice = roi_arr[:, slice_dx, :]
            im_slice = im_arr[:, slice_dx, :]
            extent = [0, im_arr.shape[2]*im_spacing[2], 0, im_arr.shape[0]*im_spacing[0]]
        elif slice_orient == OrientationOptions.SAG:
            roi_slice = roi_arr[:, :, slice_dx]
            im_slice = im_arr[:, :, slice_dx]
            extent = [0, im_arr.shape[1]*im_spacing[1], 0, im_arr.shape[0]*im_spacing[0]]
        assert roi_slice is not None, "DetectionMethodAbstract::__visualise_rois() : " \
                                      "expected slice_orient parameter of AXI, COR, or SAG [not %s]" % slice_orient
        # reduce the ROI slice to an outline only
        if roi_outline_only:
            binary_slice = roi_slice > 0
            inner_slice_mask = scipy.ndimage.binary_erosion(binary_slice, iterations=2)
            roi_slice[inner_slice_mask] = 0

        # plot
        if ax is None:
            f, ax = plt.subplots(1, 1)
        ax.imshow(im_slice, cmap='gray',
                  vmin=np.mean(im_slice)-1.0*np.std(im_slice),
                  vmax=np.mean(im_slice)+2.0*np.std(im_slice),
                  extent=extent)
        i = ax.imshow(np.ma.masked_where(roi_slice == 0, roi_slice),
                      cmap='nipy_spectral', vmin=np.min(roi_list)-1, vmax=np.max(roi_list)+1,
                      interpolation='none',
                      alpha=0.7,
                      extent=extent)
        ax.axis('off')
        ticks = list(range(np.min(roi_list), np.max(roi_list)+1))
        ticklabels = [ROI_IDX_LABEL_MAP[x] for x in ticks]
        cb = plt.colorbar(mappable=i, ax=ax,
                     ticks=ticks)
        cb.set_ticklabels(ticklabels=ticklabels)
        if title is not None:
            ax.set_title(title)

    def _get_rotated_sampling_im(self):
        """
        Transform the template image into the target image space. This is useful for visualisation, was the template
        landmarks such as a reference T1 or T2 slice can then be sampled in the estimated/registered target space.

        Returns:
            SimpleITK.Image: The template image transformed (but not resampled) into the target image space
        """
        im_tx = self.get_image_transform()
        if im_tx is None:
            mu.log("ROIDetector::_get_rotated_sampling_im() : no transform found, need to run detect() function",
                   LogLevels.LOG_WARNING)
            return None
        # transform the reference geometry into the target space
        # deep copy the reference image
        rotated_geom_im = sitk.GetImageFromArray(sitk.GetArrayFromImage(self.fixed_im))
        rotated_geom_im.SetSpacing(self.fixed_im.GetSpacing())
        rotated_geom_im.SetOrigin(self.fixed_im.GetOrigin())
        rotated_geom_im.SetDirection(self.fixed_im.GetDirection())
        # transform the origin
        o_new = im_tx.TransformPoint(rotated_geom_im.GetOrigin())
        # transform the direction matrix
        d = rotated_geom_im.GetDirection()
        d_0 = [d[0], d[3], d[6]]
        d_1 = [d[1], d[4], d[7]]
        d_2 = [d[2], d[5], d[8]]
        d_0_rot = im_tx.TransformVector(d_0, (0.0, 0.0, 0.0))
        d_1_rot = im_tx.TransformVector(d_1, (0.0, 0.0, 0.0))
        d_2_rot = im_tx.TransformVector(d_2, (0.0, 0.0, 0.0))
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

class ShapeDetectionMethodAbstract(DetectionMethodAbstract):
    def __init__(self, target_geo_im, fixed_im, roi_template):
        super().__init__(target_geo_im, fixed_im, target_geo_im.get_image(), roi_template)


class RegistrationMethodAbstract(DetectionMethodAbstract):
    def __init__(self, target_geo_im, fixed_im, roi_template):
        super().__init__(target_geo_im, fixed_im, target_geo_im.get_image(), roi_template)
        # parameters to store
        self.transform = None

        # flags for iteration logs/plotters
        self.first_iteration = True
        self.fig, self.axes = None, None

    @abstractmethod
    def register(self):
        return None, None

    def detect(self):
        mu.log("\t Iteration |   Metric   |   Optimizer Position",
               LogLevels.LOG_INFO)
        # register and store the transform for warping images and ROIs
        self.transform, metric = self.register()

    def get_image_transform(self):
        return self.transform

    def get_image_transform_matrix(self, fixed_to_moving=True):
        if fixed_to_moving:
            return mu.get_homog_matrix_from_transform(self.get_image_transform().GetInverse())
        else:
            return mu.get_homog_matrix_from_transform(self.get_image_transform())

    def get_detected_T1_mask(self):
        """
        Returns:
            SimpleITK.Image: T1 mask in target image space, with 0 for background and 1-14 for detected ROIs
        """
        mu.log("RegistrationMethodAbstract::get_detected_T1_mask()", LogLevels.LOG_INFO)
        target_im = self.target_geo_im.get_image()
        fixed_t1_mask_im, prop_im_dict = self.get_fixed_T1_mask()
        detected_t1_mask_im = self.__get_registered_mask(fixed_t1_mask_im, target_im)
        detected_prop_im_dict = OrderedDict()
        for p, prop_im in prop_im_dict.items():
            detected_prop_im_dict[p] = self.__get_registered_mask(prop_im, target_im)
        return detected_t1_mask_im, detected_prop_im_dict

    def get_detected_T2_mask(self):
        """
        Returns:
            SimpleITK.Image: T2 mask  in target image space, with 0 for background and 15-28 for detected ROIs
        """
        mu.log("RegistrationMethodAbstract::get_detected_T2_mask()", LogLevels.LOG_INFO)
        target_im = self.target_geo_im.get_image()
        fixed_t2_mask_im, prop_im_dict = self.get_fixed_T2_mask()
        detected_t2_mask_im = self.__get_registered_mask(fixed_t2_mask_im, target_im)
        detected_prop_im_dict = OrderedDict()
        for p, prop_im in prop_im_dict.items():
            detected_prop_im_dict[p] = self.__get_registered_mask(prop_im, target_im)
        return detected_t2_mask_im, detected_prop_im_dict

    def get_detected_DW_mask(self):
        """
        Returns:
            SimpleITK.Image: DW mask  in target image space, with 0 for background and 29-41 for detected ROIs
        """
        mu.log("RegistrationMethodAbstract::get_detected_DW_mask()", LogLevels.LOG_INFO)
        target_im = self.target_geo_im.get_image()
        fixed_dw_mask_im, prop_im_dict = self.get_fixed_DW_mask()
        detected_dw_mask_im = self.__get_registered_mask(fixed_dw_mask_im, target_im)
        detected_prop_im_dict = OrderedDict()
        for p, prop_im in prop_im_dict.items():
            detected_prop_im_dict[p] = self.__get_registered_mask(prop_im, target_im)
        return detected_dw_mask_im, detected_prop_im_dict

    # same output as log but to pdf
    def write_pdf_summary_page(self, c, sup_title=None):
        table_width = 170
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font

        if sup_title is None:
            sup_title = "ROI Detection: Summary <%s>" % self.target_geo_im.get_label()

        # draw the summary figure
        # -----------------------------------------------------------
        # setup figure
        f = None
        if self.roi_template.get_dw_roi_dict():
            f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
            f.suptitle(sup_title)
            f.set_size_inches(14, 6)
            # draw the template rois on the template image
            self.visualise_fixed_DW_rois(ax=ax1)
            self.visualise_fixed_DW_rois(ax=ax5, slice_orient=OrientationOptions.SAG)
            # draw the registration (pre/post)
            self.visualise_DW_registration(invert_moving=False, pre_reg_ax=ax2, post_reg_ax=ax3)
            self.visualise_DW_registration(invert_moving=False, pre_reg_ax=ax6, post_reg_ax=ax7,
                                           slice_orient=OrientationOptions.SAG)
            # visualise the transfromed ROI masks on the target image
            self.visualise_detected_DW_rois(ax=ax4)
            self.visualise_detected_DW_rois(ax=ax8, slice_orient=OrientationOptions.SAG)

        if self.roi_template.get_t1_roi_dict():
            f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
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

        if f is not None:
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

    def visualise_DW_registration(self,
                                  pre_reg_ax=None, post_reg_ax=None,
                                  invert_moving=True, slice_orient=OrientationOptions.AXI):
        self.__visualise_registration(self.roi_template.get_dw_slice_dx(slice_orient),
                                      slice_orient=slice_orient,
                                      pre_reg_ax=pre_reg_ax, post_reg_ax=post_reg_ax,
                                      invert_moving=invert_moving,
                                      title="DW")

    def print_iteration_info(self, method):
        mu.log("\t\t{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                                    method.GetMetricValue(),
                                                    method.GetOptimizerPosition()),
               LogLevels.LOG_INFO)

    def plot_iteration_info(self, method):
        # create plot in first iteration
        if self.first_iteration:
            self.fig, self.axes = plt.subplots(2, 2)
            self.first_iteration = False
        # plot it
        rx, ry, rz, tx, ty, tz = method.GetOptimizerPosition()
        trans = sitk.Euler3DTransform()
        trans.SetTranslation([tx, ty, tz])
        trans.SetRotation(rx, ry, rz)

        registered_im = sitk.Resample(self.moving_im, self.fixed_im, trans, sitk.sitkLinear)
        checker_im = sitk.CheckerBoard(self.fixed_im, registered_im, [2, 2, 2])

        checker_arr = sitk.GetArrayFromImage(checker_im)
        self.axes[0][0].imshow(checker_arr[:, :, int(checker_arr.shape[2]/2)])
        self.axes[1][0].imshow(checker_arr[:, int(checker_arr.shape[1]/2), :])
        self.axes[1][1].imshow(checker_arr[int(checker_arr.shape[0]/2), :, :])
        plt.pause(10)

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
            mu.log("\tRegistrationMethodAbstract::__get_registered_mask() : no transform found, need to run detect() function",
                   LogLevels.LOG_WARNING)
            return None
        return sitk.Resample(src_mask, target_im, sitk.Transform.GetInverse(self.transform), sitk.sitkNearestNeighbor)

    def __visualise_registration(self, slice_dx, slice_orient=OrientationOptions.AXI,
                                 pre_reg_ax=None, post_reg_ax=None, invert_moving=True, title=None):
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
        if (pre_reg_ax is None) or (post_reg_ax is None):
            f, (pre_reg_ax, post_reg_ax) = plt.subplots(1, 2)

        # pre-registration
        # resample the target image and mask - onto the deformed fixed space
        # (so we can use the reference ROI slice for visualisation)
        target_pre_im = sitk.Resample(self.moving_im,
                                      self.fixed_im,
                                      sitk.Euler3DTransform(), sitk.sitkLinear)
        if invert_moving:
            target_pre_im = sitk.InvertIntensity(target_pre_im, maximum=inversion_max)

        fixed_im_pixel_type = self.fixed_im.GetPixelID()
        target_pre_im_match_type = sitk.Cast(target_pre_im, fixed_im_pixel_type)
        checker_im = sitk.CheckerBoard(self.fixed_im, target_pre_im_match_type, [4, 4, 4])

        # pull out the relevant slice based on orientation
        def get_slice_and_extent(im, dx, orient):
            im_slice, extent = None, [0, 0, 0, 0]
            im_arr = sitk.GetArrayFromImage(im)
            im_spacing = np.flip(np.array(im.GetSpacing()))
            if orient == OrientationOptions.AXI:
                im_slice = im_arr[dx, :, :]
                extent = [0, im_arr.shape[2] * im_spacing[2], 0, im_arr.shape[1] * im_spacing[1]]
            elif orient == OrientationOptions.COR:
                im_slice = im_arr[:, dx, :]
                extent = [0, im_arr.shape[2] * im_spacing[2], 0, im_arr.shape[0] * im_spacing[0]]
            elif orient == OrientationOptions.SAG:
                im_slice = im_arr[:, :, dx]
                extent = [0, im_arr.shape[1] * im_spacing[1], 0, im_arr.shape[0] * im_spacing[0]]
            assert im_slice is not None, "ROIDetector::__visualise_registration() : " \
                                         "expected slice_orient parameter of AXI, COR, or SAG [not %s]" % slice_orient
            return im_slice, extent

        im_slice, extent = get_slice_and_extent(checker_im, slice_dx, slice_orient)
        vmin = np.mean(im_slice) - 1.0 * np.std(im_slice)
        vmax = np.mean(im_slice) + 2.0 * np.std(im_slice)
        pre_reg_ax.imshow(im_slice, cmap='gray', vmin=vmin, vmax=vmax, extent=extent)

        # post-registration
        rotated_fixed_im = self._get_rotated_sampling_im()
        # resample the target image and mask - onto the deformed fixed space
        # (so we can use the reference ROI slice for visualisation)
        target_im = sitk.Resample(self.moving_im,
                                  rotated_fixed_im,
                                  sitk.Euler3DTransform(), sitk.sitkLinear)
        if invert_moving:
            target_im = sitk.InvertIntensity(target_im, maximum=inversion_max)

        rotated_fixed_im_pixel_type = rotated_fixed_im.GetPixelID()
        target_im_match_type = sitk.Cast(target_im, rotated_fixed_im_pixel_type)
        checker_im_reg = sitk.CheckerBoard(rotated_fixed_im, target_im_match_type, [4, 4, 4])
        im_slice, extent = get_slice_and_extent(checker_im_reg, slice_dx, slice_orient)
        post_reg_ax.imshow(im_slice, cmap='gray', vmin=vmin, vmax=vmax, extent=extent)
        for ax in [pre_reg_ax, post_reg_ax]:
            ax.axis('off')
        if title is not None:
            pre_reg_ax.set_title("%s un-registered" % title)
            post_reg_ax.set_title("%s registered" % title)








if __name__ == "__main__":
    main()
