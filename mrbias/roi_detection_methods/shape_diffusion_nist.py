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
from collections import OrderedDict

import SimpleITK as sitk
import pandas as pd
import numpy as np
import scipy.signal as scisig
from scipy import ndimage
from scipy import spatial as scispat
from scipy import interpolate as scipolate
from skimage import feature as skim_feat
from skimage import filters as skim_fltr
from skimage import measure as skim_meas
from skimage import transform as skim_trans
from skimage import draw as skim_draw
from skimage import color as skim_col
from skimage import util as skim_util
from scipy.cluster.vq import kmeans


import matplotlib.pyplot as plt
import matplotlib as mpl

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
from mrbias.misc_utils import LogLevels, OrientationOptions, ROI_IDX_LABEL_MAP, DW_ROI_LABEL_IDX_MAP
from mrbias.roi_detection_methods.detection_methods import ShapeDetectionMethodAbstract
from mrbias.roi_detection_methods.roi_template import ROIImage, ROICylinder, ROISphere


class ShapeDiffusionNIST(ShapeDetectionMethodAbstract):
    def __init__(self, target_geo_im, fixed_im, roi_template,
                 flip_cap_dir=False,
                 debug_vis=False,
                 fine_tune_rois=False):
        super().__init__(target_geo_im, fixed_im, roi_template)
        self.moving_roi_image = None
        # image transforms
        self.tx0_fix = None
        self.tx0_mov = None
        self.tx1_mov = None
        self.tx_fix_2_mov = None
        self.tx_mov_2_fix = None
        # bottle detection parameters
        self.edge_dect_sigma_mm = 1.5  # for canny edge detector
        self.edge_dect_thresholds = [0.7, 1.0]  # for canny edge detector
        self.bottle_radii_mm = [14.0, 18.0]  # for hough transform (circle detect)
        self.cap_radii_mm = [8.0, 10.0]  # for hough transform (circle detect)
        self.phantom_radii_mm = [85, 110] # for phantom detection
        self.hough_n_circles = 20  # for peak detection of circles
        self.threshold_euclid_mm = 4.0
        self.peak_detect_mod = 0.4
        self.inner_ring_only = True
        self.flip_cap_dir = flip_cap_dir
        self.fine_tune_rois = fine_tune_rois
        # expected bottle size parameters
        self.bottle_height_mm = 50.0
        self.bottlecap_height_mm = 16.0
        self.bottle_radius_mm = 15.0
        self.inner_circle_rad_mm = 35.0
        self.outer_circle_rad_mm = 60.0
        self.band_mm = 5.0
        # detection analysis data for visualisation
        self.fixed_bot_locs_0 = None
        self.moving_bot_locs_0 = None
        self.fixed_bot_locs_1 = None
        self.moving_bot_locs_1 = None
        self.fixed_bot_mid_mm = None
        self.moving_bot_mid_mm = None
        self.fixed_cap_mid_mm = None
        self.moving_cap_mid_mm = None
        self.fixed_cap_dir_mod = None
        self.moving_cap_dir_mod = None
        # longitudinal profiles
        self.bot_profile_fix = None
        self.bot_mm_vec_fix = None
        self.cap_profile_fix = None
        self.cap_mm_vec_fix = None
        self.bot_profile_mov = None
        self.bot_mm_vec_mov = None
        self.cap_profile_mov = None
        self.cap_mm_vec_mov = None
        # phantom centre
        self.cntr_dx_fix = None
        self.cntr_dx_mov = None
        # ctr, inner ring, outer ring
        self.axi_ctr_dxs_fix = None
        self.axi_inner_dxs_fix = None
        self.axi_outer_dxs_fix = None
        self.axi_ctr_dxs_mov = None
        self.axi_inner_dxs_mov = None
        self.axi_outer_dxs_mov = None
        # storage for fine-tuning visualisation
        self.ft_centroid0_dict = None  # initial centroid from detection
        self.ft_centroid1_dict = None  # centroid following fine-tuning
        self.ft_bbox_dict = None  # bounding box for crop region
        self.ft_circle_dict = None  # detected circle centroids
        self.ft_curve_fit_dict = None
        # other flags
        self.debug_vis = debug_vis

    def detect(self):
        # STEP 1: Detect diffusion bottles and centre of array of bottles (on both template and target image)
        # ----------------------------------------------------------------------------------
        # detect bottle landmarks on fixed image
        r_val = self.detect_diffusion_bottles(self.fixed_im)
        axi_cntr_dxs_fix_0, axi_cntr_dxs_fix, \
            bot_mid_fix_mm, bot_profile_fix, bot_mm_vec_fix,\
            cap_mid_fix_mm, cap_profile_fix, cap_mm_vec_fix, cap_mod_fix, \
            axi_ctr_dxs_fix, axi_inner_dxs_fix, axi_outer_dxs_fix,\
            bot_cntr_dx_fix, botcap_cntr_dx_fix, cntr_dx_fix = r_val
        # detect bottle landmarks on moving image
        r_val = self.detect_diffusion_bottles(self.moving_im, flip_cap_dir=self.flip_cap_dir)
        axi_cntr_dxs_mov_0, axi_cntr_dxs_mov, \
            bot_mid_mov_mm, bot_profile_mov, bot_mm_vec_mov,\
            cap_mid_mov_mm, cap_profile_mov, cap_mm_vec_mov, cap_mod_mov, \
            axi_ctr_dxs_mov, axi_inner_dxs_mov, axi_outer_dxs_mov,\
            bot_cntr_dx_mov, botcap_cntr_dx_mov, cntr_dx_mov = r_val
        # store the detected bottle location parameters
        self.fixed_bot_locs_0 = axi_cntr_dxs_fix_0
        self.fixed_bot_locs_1 = axi_cntr_dxs_fix
        self.moving_bot_locs_0 = axi_cntr_dxs_mov_0
        self.moving_bot_locs_1 = axi_cntr_dxs_mov
        # store the bottle longitudinal location params
        self.fixed_bot_mid_mm = bot_mid_fix_mm
        self.fixed_cap_mid_mm = cap_mid_fix_mm
        self.fixed_cap_dir_mod = cap_mod_fix
        self.moving_cap_mid_mm = cap_mid_mov_mm
        self.moving_bot_mid_mm = bot_mid_mov_mm
        self.moving_cap_dir_mod = cap_mod_mov
        # longitudinal profiles
        self.bot_profile_fix = bot_profile_fix
        self.bot_mm_vec_fix = bot_mm_vec_fix
        self.cap_profile_fix = cap_profile_fix
        self.cap_mm_vec_fix = cap_mm_vec_fix
        self.bot_profile_mov = bot_profile_mov
        self.bot_mm_vec_mov = bot_mm_vec_mov
        self.cap_profile_mov = cap_profile_mov
        self.cap_mm_vec_mov = cap_mm_vec_mov
        # phantom centre
        self.cntr_dx_fix = cntr_dx_fix
        self.cntr_dx_mov = cntr_dx_mov
        # store the final locaiton params (ctr, inner ring, outer ring)
        self.axi_ctr_dxs_fix = axi_ctr_dxs_fix
        self.axi_inner_dxs_fix = axi_inner_dxs_fix
        self.axi_outer_dxs_fix = axi_outer_dxs_fix
        self.axi_ctr_dxs_mov = axi_ctr_dxs_mov
        self.axi_inner_dxs_mov = axi_inner_dxs_mov
        self.axi_outer_dxs_mov = axi_outer_dxs_mov


        # STEP 2: Align the detected target bottle array with the detected template bottle array
        # ----------------------------------------------------------------------------------
        # align the bottles, including a flip if bottles are oriented with different cap directions
        # use the image intensity in the bottles and the distance between centroids to find the best in-plane rotation
        align_dat = self.__align_bottle_array(cntr_dx_fix, axi_cntr_dxs_fix, axi_inner_dxs_fix, bot_cntr_dx_fix,
                                              botcap_cntr_dx_fix,
                                              cntr_dx_mov, axi_cntr_dxs_mov, axi_inner_dxs_mov, bot_cntr_dx_mov,
                                              botcap_cntr_dx_mov)
        im0_fix, im0_mov, im1_mov, tx0_fix, tx0_mov, tx1_mov = align_dat
        # store the transforms
        self.tx0_fix = tx0_fix
        self.tx0_mov = tx0_mov
        self.tx1_mov = tx1_mov
        # calculate transforms from moving to fixed (and vice-versa)
        tx_mov_combined = sitk.Transform()
        tx_mov_combined.AddTransform(tx0_mov)
        tx_mov_combined.AddTransform(tx1_mov)
        tx_combined = sitk.Transform()
        tx_combined.AddTransform(tx0_fix)
        tx_combined.AddTransform(tx_mov_combined.GetInverse())
        self.tx_fix_2_mov = tx_combined
        self.tx_mov_2_fix = self.tx_fix_2_mov.GetInverse()

        # STEP 3: Associate template ROI labels with the detected bottles on the target image
        # ----------------------------------------------------------------------------------
        r_val = self.__link_template_roi_labels_to_target_image(im0_fix, axi_cntr_dxs_fix, bot_cntr_dx_fix,
                                                                im0_mov, axi_cntr_dxs_mov, bot_cntr_dx_mov,
                                                                tx0_mov, tx1_mov,
                                                                euclid_threshold_mm=20.0)
        fixed_coords_and_label_list, moving_coords_and_label_list = r_val

        # STEP 4: Map template ROIs to the target image to construct a ROI mask on the target image
        # ------------------------------------------------------------------------------------------
        r_val = self.__construct_moving_roi_image(fixed_coords_and_label_list, moving_coords_and_label_list,
                                                  cap_mod_fix, cap_mod_mov)
        self.moving_roi_image = r_val




    def get_detected_T1_mask(self):
        """
        Returns:
           SimpleITK.Image: T1 mask in target image space, with 0 for background and 1-14 for detected ROIs
        """
        target_im = self.target_geo_im.get_image() # this is the same as self.moving_im
        blank_mask_im = mu.make_sitk_zeros_image_like(target_im)
        blank_prop_dict = OrderedDict()
        return blank_mask_im, blank_prop_dict

    def get_detected_T2_mask(self):
        """
        Returns:
            SimpleITK.Image: T2 mask  in target image space, with 0 for background and 15-28 for detected ROIs
        """
        target_im = self.target_geo_im.get_image() # this is the same as self.moving_im
        blank_mask_im = mu.make_sitk_zeros_image_like(target_im)
        blank_prop_dict = OrderedDict()
        return blank_mask_im, blank_prop_dict

    def get_detected_DW_mask(self):
        """
        Returns:
           SimpleITK.Image: DW mask  in target image space, with 0 for background and 29-41 for detected ROIs
        """
        mu.log("ShapeDiffusionNIST::get_detected_DW_mask()", LogLevels.LOG_INFO)
        return self.moving_roi_image.get_mask_image() # mask, prop_dict

    def get_image_transform(self):
        return self.tx_mov_2_fix

    def get_image_transform_matrix(self, fixed_to_moving=True):
        tx0_fix = mu.get_homog_matrix_from_transform(self.tx0_fix)
        tx0_mov = mu.get_homog_matrix_from_transform(self.tx0_mov)
        tx1_mov = mu.get_homog_matrix_from_transform(self.tx1_mov)
        tx01_mov = tx0_mov.dot(tx1_mov)
        tx_fix_2_mov = tx0_fix.dot(np.linalg.inv(tx01_mov))
        if fixed_to_moving:
            return tx_fix_2_mov
        else:
            return np.linalg.inv(tx_fix_2_mov)


    def write_pdf_summary_page(self, c, sup_title=None):
        self.write_pdf_shape_detection_page(c, sup_title=sup_title)
        if self.fine_tune_rois:
            self.write_pdf_fine_tuning_page(c, sup_title=sup_title)
        self.write_pdf_result_page(c, sup_title=sup_title)

    def write_pdf_shape_detection_page(self, c, sup_title=None):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font

        if sup_title is None:
            sup_title = "ROI Detection: Summary <%s>" % self.target_geo_im.get_label()

        # draw the summary figure
        # -----------------------------------------------------------
        # setup figure
        f = None
        if self.get_image_transform() is not None:
            f, axes = plt.subplots(2, 3, gridspec_kw=dict(width_ratios=[3, 2, 3]))
            f.suptitle(sup_title)
            f.set_size_inches(12, 8)

            # plot the candiate centres on the fixed and moving images
            for d_vec in zip([self.fixed_im, self.moving_im],
                             [self.fixed_bot_locs_0, self.moving_bot_locs_0],
                             [self.fixed_bot_locs_1, self.moving_bot_locs_1],
                             [self.fixed_bot_mid_mm, self.moving_bot_mid_mm],
                             [self.fixed_cap_mid_mm, self.moving_cap_mid_mm],
                             [self.fixed_cap_dir_mod, self.moving_cap_dir_mod],
                             [[self.axi_ctr_dxs_fix, self.axi_inner_dxs_fix, self.axi_outer_dxs_fix],
                              [self.axi_ctr_dxs_mov, self.axi_inner_dxs_mov, self.axi_outer_dxs_mov]],
                             [self.cntr_dx_fix, self.cntr_dx_mov],
                             [(axes[0][0], axes[0][1]), (axes[1][0], axes[1][1])]):
                im, bot_locs_0, bot_locs_1, bot_mid_mm, cap_mid_mm, cap_dir_mod, ring_vec, cntr_dx, sum_axes = d_vec
                spc_arr = np.array(im.GetSpacing())
                in_plane_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
                # plot candidate centres
                im_slc, extent = mu.get_slice_and_extent(im, OrientationOptions.AXI)
                sum_axes[0].imshow(im_slc, extent=extent, cmap="gray", origin='lower')
                sum_axes[0].scatter(bot_locs_0[:, 0] * in_plane_spacing_mm, bot_locs_0[:, 1] * in_plane_spacing_mm,
                                    marker='o', color='r')
                # plot the detected bottle location in the slice direction
                im_slc, extent = mu.get_slice_and_extent(im, OrientationOptions.SAG)
                rot_extent = [extent[2], extent[3], extent[0], extent[1]]
                sum_axes[1].imshow(np.rot90(im_slc), extent=rot_extent, cmap="gray", origin='upper')
                sum_axes[1].vlines([bot_mid_mm,
                                    bot_mid_mm - self.bottle_height_mm / 2.0,
                                    bot_mid_mm + self.bottle_height_mm / 2.0],
                                   ymin=extent[0], ymax=extent[1],
                                   colors=['b', 'b', 'b'], linestyles=[':', '-', '-'])
                sum_axes[1].vlines([cap_mid_mm,
                                    cap_mid_mm + cap_dir_mod * self.bottlecap_height_mm / 2.0],
                                   ymin=extent[0], ymax=extent[1],
                                   colors=['r', 'r'], linestyles=[':', '-'])

                # plot refined candidate centres
                sum_axes[0].scatter(bot_locs_1[:, 0] * in_plane_spacing_mm, bot_locs_1[:, 1] * in_plane_spacing_mm,
                                    marker='x',
                                    color='g')
                for x, y in zip(bot_locs_1[:, 0], bot_locs_1[:, 1]):
                    circ = mpl.patches.Circle((x * in_plane_spacing_mm, y * in_plane_spacing_mm),
                                              self.bottle_radius_mm,
                                              edgecolor='g', fill=False)
                    sum_axes[0].add_patch(circ)

                # plot detected centres, inner and outer ring circles
                cntr_locs, inner_locs, outer_locs = ring_vec
                for locs, col in zip([cntr_locs, inner_locs, outer_locs],
                                     ['y', 'b', 'y']):
                    if locs.shape[0]: # check not empty
                        sum_axes[0].scatter(locs[:, 0] * in_plane_spacing_mm, locs[:, 1] * in_plane_spacing_mm,
                                            marker='+', color=col)
                        for x, y in zip(locs[:, 0], locs[:, 1]):
                            circ = mpl.patches.Circle((x * in_plane_spacing_mm, y * in_plane_spacing_mm),
                                                      self.bottle_radius_mm - 2,
                                                      edgecolor=col, fill=False)
                            sum_axes[0].add_patch(circ)
                # draw the expected rings
                for ring_rad_mm, col in zip([self.inner_circle_rad_mm, self.outer_circle_rad_mm],
                                            ['b', 'y']):
                    for line_style, rad, col2 in zip([":", ":"],
                                                     [ring_rad_mm - self.band_mm, ring_rad_mm + self.band_mm],
                                                     [col, col]):
                        circ = mpl.patches.Circle((cntr_dx[0] * in_plane_spacing_mm, cntr_dx[1] * in_plane_spacing_mm),
                                                  rad,
                                                  edgecolor=col2, fill=False, linestyle=line_style)
                        sum_axes[0].add_patch(circ)

            ### BOTTLE LONGITUDINAL DETECTION / HISTORGRAM
            for d_vec in zip([self.fixed_im, self.moving_im],
                             [self.bot_profile_fix, self.bot_profile_mov],
                             [self.bot_mm_vec_fix, self.bot_mm_vec_mov],
                             [self.cap_profile_fix, self.cap_profile_mov],
                             [self.cap_mm_vec_fix, self.cap_mm_vec_mov],
                             [self.fixed_bot_mid_mm, self.moving_bot_mid_mm],
                             [self.fixed_cap_mid_mm, self.moving_cap_mid_mm],
                             [axes[0][2], axes[1][2]]):
                im, b_profile, b_mm_vec, c_profile, c_mm_vec, b_mid_mm, c_mid_mm, ax = d_vec
                spc_arr = np.array(im.GetSpacing())
                thru_plane_spacing_mm = spc_arr[2]
                # plot raw data
                ax.plot(b_mm_vec * thru_plane_spacing_mm - thru_plane_spacing_mm / 2.0, b_profile, "bo")
                ax.plot(c_mm_vec * thru_plane_spacing_mm - thru_plane_spacing_mm / 2.0, c_profile, "ro")
                ax.bar(b_mm_vec * thru_plane_spacing_mm - thru_plane_spacing_mm / 2.0, b_profile, align='center',
                       width=thru_plane_spacing_mm,
                       color='b', linewidth=0, alpha=0.25)
                ax.bar(c_mm_vec * thru_plane_spacing_mm - thru_plane_spacing_mm / 2.0, c_profile, align='center',
                       width=thru_plane_spacing_mm,
                       color='r', linewidth=0, alpha=0.25)
                # plot the final fit
                x_mm = b_mm_vec * thru_plane_spacing_mm - thru_plane_spacing_mm / 2.0
                bot_top_level0 = np.max(b_profile)
                cap_top_level0 = np.max(c_profile)
                ax.plot(x_mm, tophat(x_mm, 0.0, bot_top_level0, b_mid_mm, self.bottle_height_mm), 'b-', linewidth=2.0)
                ax.plot(x_mm, tophat(x_mm, 0.0, cap_top_level0, c_mid_mm, self.bottlecap_height_mm), 'r-', linewidth=2.0)



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

    def write_pdf_fine_tuning_page(self, c, sup_title=None):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font

        if sup_title is None:
            sup_title = "ROI Detection: Summary <%s>" % self.target_geo_im.get_label()

        # draw the summary figure
        # -----------------------------------------------------------
        # setup figure
        f = None
        if self.get_image_transform() is not None:
            f, axes = plt.subplots(3, 13)
            f.set_tight_layout(True)
            f.suptitle(sup_title)
            f.set_size_inches(12, 8)

            fixed_roi_dict = self.roi_template.get_dw_roi_dict()
            bottle_arr = sitk.GetArrayFromImage(self.moving_im)
            spc_arr = np.array(self.moving_im.GetSpacing())
            in_plane_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
            thru_plane_spacing_mm = spc_arr[2]

            for r_dx, ((roi_label, fixed_roi), ax_axi, ax_cor, ax_sag) in enumerate(zip(fixed_roi_dict.items(),
                                                                                        axes[0],
                                                                                        axes[1],
                                                                                        axes[2])):
                ax_axi.set_title(fixed_roi.label)
                cx, cy, cz = self.ft_centroid0_dict[fixed_roi.label]
                cx1, cy1, cz1 = self.ft_centroid1_dict[fixed_roi.label]
                bot_half_height_vox, bot_radius_vox = self.ft_bbox_dict[fixed_roi.label]
                circ_dx_x, circ_dx_y, circ_dx_z = self.ft_circle_dict[fixed_roi.label]
                curve_dx_x, curve_dx_y, curve_dx_z = self.ft_curve_fit_dict[fixed_roi.label]

                padd_y, padd_x = None, None
                if (cy - bot_radius_vox) < 0:
                    padd_y = -(cy - bot_radius_vox)
                if (cx - bot_radius_vox) < 0:
                    padd_x = -(cx - bot_radius_vox)
                bottle_crop_arr = bottle_arr[
                                  cz - bot_half_height_vox:cz + bot_half_height_vox,
                                  np.max([0, cy - bot_radius_vox]):     cy + bot_radius_vox,
                                  np.max([0, cx - bot_radius_vox]):     cx + bot_radius_vox]
                if padd_y is not None:
                    bottle_crop_arr = np.pad(bottle_crop_arr, ((0, 0), (padd_y, 0), (0, 0)),
                                             'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                if padd_x is not None:
                    bottle_crop_arr = np.pad(bottle_crop_arr, ((0, 0), (0, 0), (padd_x, 0)),
                                             'constant', constant_values=((0, 0), (0, 0), (0, 0)))
                bottle_crop_im = sitk.GetImageFromArray(bottle_crop_arr)
                bottle_crop_im.SetSpacing(spc_arr)
                if ax_axi is not None:
                    im_slc, extent = mu.get_slice_and_extent(bottle_crop_im, OrientationOptions.AXI)
                    ax_axi.imshow(im_slc, extent=extent, cmap="gray", origin='lower')
                    ax_axi.axhline(bot_radius_vox * in_plane_spacing_mm, color='r')
                    ax_axi.axhline(cz1 * in_plane_spacing_mm, color='r', linestyle=':')
                    ax_axi.axvline(bot_radius_vox * in_plane_spacing_mm, color='b')
                    ax_axi.axvline(cy1 * in_plane_spacing_mm, color='b', linestyle=":")
                if ax_cor is not None:
                    im_slc, extent = mu.get_slice_and_extent(bottle_crop_im, OrientationOptions.COR)
                    ax_cor.imshow(im_slc, extent=extent, cmap="gray", origin='lower')
                    ax_cor.axvline(bot_radius_vox * in_plane_spacing_mm, color='b')
                    ax_cor.plot(cy1 * np.ones_like(circ_dx_x) * in_plane_spacing_mm,
                                circ_dx_x * thru_plane_spacing_mm, 'b:')
                    ax_cor.scatter(circ_dx_y * in_plane_spacing_mm,
                                   circ_dx_x * thru_plane_spacing_mm, color='g')
                    ax_cor.plot(curve_dx_y * in_plane_spacing_mm, curve_dx_x * thru_plane_spacing_mm, 'g')
                if ax_sag is not None:
                    im_slc, extent = mu.get_slice_and_extent(bottle_crop_im, OrientationOptions.SAG)
                    ax_sag.imshow(im_slc, extent=extent, cmap="gray", origin='lower')
                    ax_sag.axvline(bot_radius_vox * in_plane_spacing_mm, color='r')
                    ax_sag.plot(cz1 * np.ones_like(circ_dx_x) * in_plane_spacing_mm,
                                circ_dx_x * thru_plane_spacing_mm, 'r:')
                    ax_sag.scatter(circ_dx_z * in_plane_spacing_mm,
                                   circ_dx_x * thru_plane_spacing_mm, color='g')
                    ax_sag.plot(curve_dx_z * in_plane_spacing_mm, curve_dx_x * thru_plane_spacing_mm, 'g')
                for ax in [ax_axi, ax_cor, ax_sag]:
                    if ax is not None:
                        ax.set_xticks([])
                        ax.set_yticks([])

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


    def write_pdf_result_page(self, c, sup_title=None):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font

        if sup_title is None:
            sup_title = "ROI Detection: Summary <%s>" % self.target_geo_im.get_label()

        # draw the summary figure
        # -----------------------------------------------------------
        # setup figure
        f = None
        if self.get_image_transform() is not None:
            f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
            f.suptitle(sup_title)
            f.set_size_inches(12, 8)
            self.visualise_fixed_DW_rois(ax1)
            self.visualise_fixed_DW_rois(ax4, slice_orient=OrientationOptions.COR)
            self.visualise_fixed_DW_rois(ax7, slice_orient=OrientationOptions.SAG)
            self.visualise_detected_DW_rois(ax2)
            self.visualise_detected_DW_rois(ax5, slice_orient=OrientationOptions.COR)
            self.visualise_detected_DW_rois(ax8, slice_orient=OrientationOptions.SAG)
            self.visualise_detected_DW_rois(ax3, in_template_space=False)
            self.visualise_detected_DW_rois(ax6, slice_orient=OrientationOptions.COR, in_template_space=False)
            self.visualise_detected_DW_rois(ax9, slice_orient=OrientationOptions.SAG, in_template_space=False)

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


    # bottle dimensions:
    #    diameter = 30 mm
    #    height   = 50 mm
    # bottle cap dimensions:
    #    diameter = 18 mm
    #    height   = 16 mm
    def detect_diffusion_bottles(self, im, flip_cap_dir=False):
        # extract image data and spatial parameters
        im_arr = sitk.GetArrayFromImage(im)
        spc_arr = np.array(im.GetSpacing())
        # assume spacing is isotropic in 2D plane
        in_plane_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
        thru_plane_spacing_mm = spc_arr[2]
        # other
        n_slices = im.GetSize()[2]

        # STEP A: detect circles matching the bottle and the bottle-cap expected radius
        # ----------------------------------------------------------------------------------
        bot_centre_map, bottle_centre_arr, cap_centre_arr = self.__detect_bottle_and_cap_circles(im_arr,
                                                                                                 in_plane_spacing_mm)

        # STEP B: Use the detected circles to identify candidate bottle locations (in-plane)
        # ----------------------------------------------------------------------------------
        circle_centre_blur_sigma_mm = [30.0, 5.0, 5.0]
        bot_locs_0 = self.__detect_candidate_bottle_locations(bot_centre_map,
                                                              in_plane_spacing_mm, thru_plane_spacing_mm,
                                                              circle_centre_blur_sigma_mm)

        # STEP C: 	Estimate the bottle array location in the through plane direction
        #           using the candidate bottle locations
        # ----------------------------------------------------------------------------------
        # determine the through slice extent of the bottles and the caps:
        #   - 1. find detected circle centres which are close to candidate bottle centres
        #   - 2. locate bottle in through plane direction
        #           by fitting an expected square profile to the bottle and cap projected data
        bottle_height_mm = self.bottle_height_mm
        bottlecap_height_mm = self.bottlecap_height_mm
        r_vec = self.__estimate_longitudinal_bottle_location(bot_locs_0,
                                                             bottle_centre_arr, cap_centre_arr,
                                                             in_plane_spacing_mm,
                                                             thru_plane_spacing_mm,
                                                             bottle_height_mm,
                                                             bottlecap_height_mm,
                                                             n_slices,
                                                             flip_cap_dir)
        bot_mid_mm, bot_profile, bot_mm_vec, \
            cap_mid_mm, cap_profile, cap_mm_vec, cap_dir_mod = r_vec

        bot_mid_slice_dx = int(np.round(bot_mid_mm / thru_plane_spacing_mm))
        bot_half_height_vox = int(np.round(bottle_height_mm / thru_plane_spacing_mm / 2.0))
        botcap_mid_slice_dx = int(np.round(cap_mid_mm / thru_plane_spacing_mm))

        # STEP D: 	Use the detected bottle location to estimate the phantom location
        # ----------------------------------------------------------------------------------
        centre_of_mass = self.__estimate_phantom_location(im_arr, bot_mid_slice_dx, bot_half_height_vox)

        # STEP E:  Use the detected bottle location and estimated phantom location to refine candidate bottle locations (in-plane).
        #          This allows the rejection of detected circles outside of the phantom and above/bellow the bottles.
        # ----------------------------------------------------------------------------------
        bot_locs_1, bot_centre_smooth_map = self.__refine_candidate_bottle_locations(bot_centre_map, centre_of_mass,
                                                                                     bot_mid_slice_dx, bot_half_height_vox,
                                                                                     im, in_plane_spacing_mm)

        # STEP F: Detect the centre of the array of bottles based on the refined candidate bottle positions
        #         and estimated phantom location
        # ----------------------------------------------------------------------------------
        # detect the centre based on detected objects and the first crack at centre based on image intensity
        inner_circle_rad_mm = self.inner_circle_rad_mm
        outer_circle_rad_mm = self.outer_circle_rad_mm
        cntr_dx = self.__detect_centre_of_bottle_array(bot_locs_1, centre_of_mass,
                                                       im, in_plane_spacing_mm,
                                                       inner_circle_rad_mm, outer_circle_rad_mm)

        # STEP G: Use the detected centre of bottle array to identify candidate bottles in the inner ring
        # ----------------------------------------------------------------------------------
        # use known distances from the centre to re-filter cadidate centroids
        band_mm = self.band_mm
        cntr_locs, inner_locs, outer_locs = self.__categorise_centroids(bot_locs_1, cntr_dx,
                                                                        bot_centre_smooth_map, in_plane_spacing_mm,
                                                                        inner_circle_rad_mm,
                                                                        outer_circle_rad_mm,
                                                                        band_mm)

        # return the key data
        return bot_locs_0, bot_locs_1, \
            bot_mid_mm, bot_profile, bot_mm_vec, \
            cap_mid_mm, cap_profile, cap_mm_vec, cap_dir_mod,\
            cntr_locs, inner_locs, outer_locs, \
            int(bot_mid_slice_dx), int(botcap_mid_slice_dx), cntr_dx

    def __align_bottle_array(self,
                             cntr_dx_fix, axi_cntr_dxs_fix, axi_inner_dxs_fix,
                             bot_cntr_dx_fix, botcap_cntr_dx_fix,
                             cntr_dx_mov, axi_cntr_dxs_mov, axi_inner_dxs_mov,
                             bot_cntr_dx_mov,  botcap_cntr_dx_mov):
        # STEP A: Get both images into a shared coordinate space for comparison/angular alignment
        # ----------------------------------------------------------------------------------
        # transform the fixed image so that it's origin aligns with the central vial
        # get the location of the centre (fixed)
        r_val = self.__get_images_and_cntrds_into_shared_space(cntr_dx_fix, bot_cntr_dx_fix, botcap_cntr_dx_fix,
                                                               cntr_dx_mov, axi_cntr_dxs_mov, axi_inner_dxs_mov,
                                                               bot_cntr_dx_mov, botcap_cntr_dx_mov)
        im0_fix, tx0_fix, im0_mov, tx0_mov, centre_dx_list, inner_centre_dx_list = r_val

        # plot the centered to see if it worked
        axes_rot = None
        if self.debug_vis:
            f, axes = plt.subplots(2, 2)
            f.suptitle("ShapeDiffusionNIST::__align_bottle_array()")
            registered_im = sitk.Resample(im0_mov, im0_fix)
            checker_im = sitk.CheckerBoard(sitk.Normalize(im0_fix), sitk.Normalize(registered_im), [2, 2, 2])
            for ax_dx, (orient, ax) in enumerate(zip([OrientationOptions.AXI,
                                                      OrientationOptions.COR,
                                                      OrientationOptions.SAG],
                                                     [axes[0][0], axes[1][0], axes[1][1]])):
                im_slc, extent = mu.get_slice_and_extent(checker_im, orient)
                origin = 'upper'
                if extent == OrientationOptions.AXI:
                    origin = 'lower'
                ax.imshow(im_slc, extent=extent, cmap="gray", origin=origin)
            # check the centres on the moving image are still correct
            f, axes_rot = plt.subplots(1, 3)
            # fixed
            im_slc, extent = mu.get_slice_and_extent(im0_fix, OrientationOptions.AXI)
            axes_rot[0].imshow(im_slc, extent=extent, cmap="gray", origin='lower')
            spc_arr = np.array(im0_fix.GetSpacing())
            in_plance_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
            axes_rot[0].scatter(np.array(axi_cntr_dxs_fix)[:, 0] * in_plance_spacing_mm,
                                np.array(axi_cntr_dxs_fix)[:, 1] * in_plance_spacing_mm, marker='o', color='r')
            # moving
            im_slc, extent = mu.get_slice_and_extent(im0_mov, OrientationOptions.AXI)
            axes_rot[1].imshow(im_slc, extent=extent, cmap="gray", origin='lower')
            spc_arr = np.array(im0_mov.GetSpacing())
            in_plance_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
            axes_rot[1].scatter(np.array(centre_dx_list)[:, 0] * in_plance_spacing_mm,
                                np.array(centre_dx_list)[:, 1] * in_plance_spacing_mm, marker='o', color='r')

        # STEP B: Find the optimal rotation of the moving data based on image intensity similarity
        #         and detected centroid proximity against the fixed/template image
        # ----------------------------------------------------------------------------------
        angle1, tx1_mov = self.__find_optimal_angle(im0_fix, axi_cntr_dxs_fix, axi_inner_dxs_fix, bot_cntr_dx_fix,
                                                    im0_mov, centre_dx_list, inner_centre_dx_list)

        # STEP C: Rotate the moving image by the detected angle
        # ----------------------------------------------------------------------------------
        tx_combined = sitk.Transform()
        tx_combined.AddTransform(tx0_mov)
        tx_combined.AddTransform(tx1_mov)
        im1_mov = sitk.Resample(self.moving_im, im0_mov, tx_combined, sitk.sitkLinear)

        if self.debug_vis:
            # plot it
            im1_mov_arr = sitk.GetArrayFromImage(im1_mov)
            axes_rot[2].imshow(im1_mov_arr[im1_mov_arr.shape[0] // 2, :, :], cmap="gray", origin="lower")

            # plot the centered to see if it worked
            f, axes = plt.subplots(2, 2)
            f.suptitle("ShapeDiffusionNIST::__align_bottle_array()")
            registered_im = sitk.Resample(im1_mov, im0_fix)
            checker_im = sitk.CheckerBoard(sitk.Normalize(im0_fix), sitk.Normalize(registered_im), [2, 2, 2])
            for ax_dx, (orient, ax) in enumerate(zip([OrientationOptions.AXI,
                                                      OrientationOptions.COR,
                                                      OrientationOptions.SAG],
                                                     [axes[0][0], axes[1][0], axes[1][1]])):
                im_slc, extent = mu.get_slice_and_extent(checker_im, orient)
                origin = 'upper'
                if extent == OrientationOptions.AXI:
                    origin = 'lower'
                ax.imshow(im_slc, extent=extent, cmap="gray", origin=origin)

        return im0_fix, im0_mov, im1_mov, tx0_fix, tx0_mov, tx1_mov

    # FUNCTIONS FOR BOTTLE DETECTION : START
    def __detect_bottle_and_cap_circles(self, im_arr, in_plane_spacing_mm):
        edge_dect_sigma_mm = self.edge_dect_sigma_mm
        edge_dect_thresholds = self.edge_dect_thresholds
        bottle_radii_mm = self.bottle_radii_mm
        cap_radii_mm = self.cap_radii_mm
        hough_n_circles= self.hough_n_circles

        cap_centre_arr = []
        bottle_centre_arr = []
        bot_centre_map = np.zeros_like(im_arr, dtype=float)
        for slice_dx in range(im_arr.shape[0]):
            #print("detect_diffusion_bottles(): detect circles in slice %d/%d" % (slice_dx, im_arr.shape[0]))
            slc = im_arr[slice_dx, :, :]

            # edge detect in preparation for circle detection
            sigma_vox = edge_dect_sigma_mm / in_plane_spacing_mm
            slc_edges = skim_feat.canny(slc, sigma=sigma_vox, low_threshold=edge_dect_thresholds[0],
                                        high_threshold=edge_dect_thresholds[1])
            # look for circles of bottle size and cap size
            bot_accums, bot_cx, bot_cy, bot_radii = self.__circle_search(slc_edges, bottle_radii_mm,
                                                                         in_plane_spacing_mm, hough_n_circles,
                                                                         enforce_min_distance=True)
            cap_accums, cap_cx, cap_cy, cap_radii = self.__circle_search(slc_edges, cap_radii_mm,
                                                                         in_plane_spacing_mm, hough_n_circles,
                                                                         enforce_min_distance=True)
            # add the centres to a 3D map
            bot_centre_map[slice_dx, bot_cx, bot_cy] = bot_accums
            # store the centroid data
            for cx, cy, a in zip(bot_cx, bot_cy, bot_accums):
                bottle_centre_arr.append([slice_dx, cx, cy, a])
            for cx, cy, a in zip(cap_cx, cap_cy, cap_accums):
                cap_centre_arr.append([slice_dx, cx, cy, a])
        return bot_centre_map, bottle_centre_arr, cap_centre_arr

    def __detect_candidate_bottle_locations(self, bot_centre_map,
                                            in_plane_spacing_mm, thru_plane_spacing_mm,
                                            bottle_blur_sigma_mm):
        # gaussian blur with a through plane width similar to the bottle height
        # this should create blobs to detect for likely bottle centroids
        sigma_mm = np.array(bottle_blur_sigma_mm)
        sigma_vox = sigma_mm / np.array([thru_plane_spacing_mm, in_plane_spacing_mm, in_plane_spacing_mm])
        bot_centre_smooth_map = skim_fltr.gaussian(bot_centre_map, sigma=sigma_vox)
        # flatten the bottle centre map through plane to identify candidate bottle centres in-plane
        bot_2d_map = np.sum(bot_centre_smooth_map, axis=0)
        bottle_radius_voxel_min = int(
            np.round(self.bottle_radii_mm[0] / in_plane_spacing_mm))  # minimum distance is smaller bottle radius
        return skim_feat.peak_local_max(bot_2d_map, min_distance=bottle_radius_voxel_min,
                                        threshold_abs=self.peak_detect_mod * np.max(bot_2d_map))

    # use hough transform to find circles within a radius range
    def __circle_search(self, edge_im, radii_mm, in_plane_spacing_mm, n_candidate_circles,
                        enforce_min_distance=False):
        radius_min_mm = radii_mm[0]  # for the bottle
        radius_max_mm = radii_mm[1]  # for the bottle
        radius_voxel_min = int(np.round(radius_min_mm / in_plane_spacing_mm))
        radius_voxel_max = int(np.round(radius_max_mm / in_plane_spacing_mm))
        hough_radii = np.arange(radius_voxel_min, radius_voxel_max + 1, 1)
        hough_res = skim_trans.hough_circle(edge_im, hough_radii)
        if enforce_min_distance:
            accums, cx, cy, radii = skim_trans.hough_circle_peaks(hough_res, hough_radii,
                                                                  total_num_peaks=n_candidate_circles,
                                                                  min_xdistance=radius_voxel_max,
                                                                  min_ydistance=radius_voxel_max)
        else:
            accums, cx, cy, radii = skim_trans.hough_circle_peaks(hough_res, hough_radii,
                                                                  total_num_peaks=n_candidate_circles)
        return accums, cx, cy, radii

    def __estimate_longitudinal_bottle_location(self, bot_locs,
                                                bottle_centre_arr, cap_centre_arr,
                                                in_plane_spacing_mm, thru_plane_spacing_mm,
                                                bottle_height_mm, bottlecap_height_mm,
                                                n_slices, flip_cap_dir):
        # C1. find detected circle centres which are close to candidate bottle centres
        bot_point_arr = []
        cap_point_arr = []
        for roi_dx, (c_x, c_y) in enumerate(bot_locs):
            # loop over the slices and note any points within a certain distance
            for z, x, y, match in bottle_centre_arr:
                dist = np.sqrt(
                    np.power((c_x - x) * in_plane_spacing_mm, 2) + np.power((c_y - y) * in_plane_spacing_mm, 2))
                if dist < self.threshold_euclid_mm:
                    bot_point_arr.append([roi_dx, z, x, y])
            # loop over the slices and note any points within a certain distance
            for z, x, y, match in cap_centre_arr:
                dist = np.sqrt(
                    np.power((c_x - x) * in_plane_spacing_mm, 2) + np.power((c_y - y) * in_plane_spacing_mm, 2))
                if dist < self.threshold_euclid_mm:
                    cap_point_arr.append([roi_dx, z, x, y])
        df_id_bottle = pd.DataFrame(bot_point_arr, columns=["ROI", "Slice", "x", "y"])
        df_id_cap = pd.DataFrame(cap_point_arr, columns=["ROI", "Slice", "x", "y"])
        #   - C2. locate bottle in through plane direction
        #           by fitting an expected square profile to the bottle and cap projected data
        # pull out the histogram signal for fitting the bottle position
        b_profile, b_bin_edges = np.histogram(df_id_bottle.Slice.values, bins=np.arange(n_slices))
        c_profile, c_bin_edges = np.histogram(df_id_cap.Slice.values, bins=np.arange(n_slices))
        bot_mid_mm, cap_mid_mm, cap_dir_mod = self.__locate_bottle_and_cap(b_profile, b_bin_edges,
                                                                           c_profile, c_bin_edges,
                                                                           bottle_height_mm, bottlecap_height_mm,
                                                                           thru_plane_spacing_mm,
                                                                           flip_cap_dir)
        return bot_mid_mm, b_profile, b_bin_edges[1:],\
            cap_mid_mm, c_profile, c_bin_edges[1:], cap_dir_mod

    def __locate_bottle_and_cap(self, b_profile, b_bin_edges, c_profile, c_bin_edges,
                                bottle_height_mm, bottlecap_height_mm,
                                thru_plane_spacing_mm, flip_cap_dir=False):

        # OBJECTIVE : Fit a single top hat function to data (use individually for bottle and then bottle cap)
        def objective(x0, x_mm, y, hat_level, hat_width):
            return np.sum(np.abs(tophat(x_mm, 0.0, hat_level, x0, hat_width) - y))

        # OBJECTIVE II : Do a joint fit of top hat functions for the bottle and the bottle cap data
        def objective_II(x0, x_mm, y_bot, y_cap, bot_width, cap_width, dir_mod):
            bottle_metric = np.sum(np.abs(tophat(x_mm, 0.0, np.max(y_bot), x0, bot_width) - y_bot))
            cap_metric = np.sum(np.abs(
                tophat(x_mm, 0.0, np.max(y_cap), x0 + dir_mod * (bot_width / 2. + cap_width / 2.), cap_width) - y_cap))
            return bottle_metric + cap_metric

        # FIT the bottle position
        x_mm = b_bin_edges[1:] * thru_plane_spacing_mm - thru_plane_spacing_mm / 2.0
        bot_top_level0 = np.max(b_profile)
        mid0 = np.mean(x_mm)
        mid0_vec = np.linspace(mid0 - 20., mid0 + 20, 60)
        obj_vec = []
        for m in mid0_vec:
            obj_vec.append(objective(m, x_mm, b_profile, bot_top_level0, bottle_height_mm))
        bottle_mid_mm = mid0_vec[np.argmin(obj_vec)]

        # FIT the bottlecap position
        mid0_vec = np.linspace(mid0 - 50., mid0 + 50, 120)
        cap_top_level0 = np.max(c_profile)
        obj_vec = []
        for m in mid0_vec:
            obj_vec.append(objective(m, x_mm, c_profile, cap_top_level0, bottlecap_height_mm))
        cap_mid_mm = mid0_vec[np.argmin(obj_vec)]

        # we can now identify which side the cap is on, and use this to create a more accurate joint
        dir_mod = 1
        if cap_mid_mm < bottle_mid_mm:
            dir_mod = -1
        if flip_cap_dir:
            dir_mod = -dir_mod
        # FIT both the bottle and the bottlecap together (based on the inital individial fits)
        mid0_vec = np.linspace(mid0 - 20., bottle_mid_mm + 20, 80)
        obj_vec = []
        for m in mid0_vec:
            obj_vec.append(objective_II(m, x_mm, b_profile, c_profile, bottle_height_mm, bottlecap_height_mm, dir_mod))
        bot_II_mid_mm = mid0_vec[np.argmin(obj_vec)]
        cap_II_mid_mm = bot_II_mid_mm + dir_mod * (bottle_height_mm / 2.0 + bottlecap_height_mm / 2.0)
        return bot_II_mid_mm, cap_II_mid_mm, dir_mod

    def __estimate_phantom_location(self, im_arr, bot_mid_slice_dx, bot_half_height_vox):
        mean_bot_slc = np.mean(
            im_arr[bot_mid_slice_dx - bot_half_height_vox:bot_mid_slice_dx + bot_half_height_vox, :, :],
            axis=0)
        threshold_value = skim_fltr.threshold_otsu(mean_bot_slc)
        labeled_foreground = (mean_bot_slc > threshold_value).astype(int)
        properties = skim_meas.regionprops(labeled_foreground, mean_bot_slc)
        center_of_mass = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        return center_of_mass

    def __refine_candidate_bottle_locations(self, bot_centre_map, center_of_mass,
                                            bot_mid_slice_dx, bot_half_height_vox,
                                            im, in_plane_spacing_mm):
        # flatten the bottle centre map through plane to identify candidate bottle centres in-plane
        bot_2d_map = np.sum(
            bot_centre_map[bot_mid_slice_dx - bot_half_height_vox:bot_mid_slice_dx + bot_half_height_vox, :, :],
            axis=0)
        # crop the candidate region to the estimated phantom extent
        phan_radius_min_mm = self.phantom_radii_mm[0]  # for the whole phantom
        phan_radius_voxel_min = int(np.round(phan_radius_min_mm / in_plane_spacing_mm))
        im_arr = sitk.GetArrayFromImage(im)
        mean_bot_slc = np.mean(
            im_arr[bot_mid_slice_dx - bot_half_height_vox:bot_mid_slice_dx + bot_half_height_vox, :, :],
            axis=0)
        circ_dx = skim_draw.disk(center=center_of_mass, radius=phan_radius_voxel_min, shape=mean_bot_slc.shape)
        sigma_2d_mm = [3.0, 3.0]
        sigma_vox = np.array(sigma_2d_mm) / np.array([in_plane_spacing_mm, in_plane_spacing_mm])
        bot_2d_map_crop = np.zeros_like(bot_2d_map)
        bot_2d_map_crop[circ_dx] = bot_2d_map[circ_dx]
        bot_centre_smooth_map = skim_fltr.gaussian(bot_2d_map_crop, sigma=sigma_vox)
        bottle_radius_voxel_min = int(
            np.round(self.bottle_radii_mm[0] / in_plane_spacing_mm))  # minimum distance is smaller bottle radius
        max_locs = skim_feat.peak_local_max(bot_centre_smooth_map, min_distance=bottle_radius_voxel_min)
        # filter to the top 13 peaks
        bot_locs_1 = self.__select_n_best_peaks(max_locs, bot_centre_smooth_map)
        # DEBUG
        if self.debug_vis:
            f, (ax1, ax2) = plt.subplots(1, 2)
            f.suptitle("ShapeDiffusionNIST::__refine_candidate_bottle_locations()")
            ax1.imshow(mean_bot_slc, cmap="gray", origin='lower')
            ax2.imshow(bot_centre_smooth_map, cmap='jet', origin='lower')
            ax2.scatter(max_locs[:, 1], max_locs[:, 0], marker='.', color='k')
            ax2.scatter(bot_locs_1[:, 1], bot_locs_1[:, 0], marker='x', color='r')
            plt.pause(0.01)
        return bot_locs_1, bot_centre_smooth_map

    def __detect_centre_of_bottle_array(self, centre_dx_list, centre_mass_dx,
                                        im, in_plane_spacing_mm,
                                        inner_circle_rad_mm, outer_circle_rad_mm):
        # other parameters
        threshold_euclid_mm = 10.0
        # create 2D array of centres
        centre_im = np.zeros([im.GetSize()[0], im.GetSize()[1]], dtype=float)
        for x_dx, y_dx in centre_dx_list:
            centre_im[x_dx, y_dx] = 1.0
        # search for circles which intersect the inner ring of diffusion test tubes
        radii_mm = [inner_circle_rad_mm - 5, inner_circle_rad_mm + 5]
        r_vals = self.__circle_search(centre_im, radii_mm, in_plane_spacing_mm,
                                      n_candidate_circles=50,   enforce_min_distance=False)
        in_accums, in_cx, in_cy, in_radii = r_vals
        # search for circles which intersect the out ring of diffusion test tubes
        radii_mm = [outer_circle_rad_mm - 5, outer_circle_rad_mm + 5]
        r_val = self.__circle_search(centre_im, radii_mm, in_plane_spacing_mm,
                                     n_candidate_circles=50,  enforce_min_distance=False)
        out_accums, out_cx, out_cy, out_radii = r_val
        # accumulate all the circle centre locations (inner, and outer) and detect the most likely center point
        circ_accum_map = np.zeros_like(centre_im)
        for cx, cy, accums in zip([in_cx, out_cx],
                                  [in_cy, out_cy],
                                  [in_accums, out_accums]):
            for x, y, a in zip(cx, cy, accums):
                circ_accum_map[x, y] += a
        # smooth new map and detect peaks
        sigma_mm = np.array([5.0, 5.0])
        sigma_vox = sigma_mm / np.array([in_plane_spacing_mm, in_plane_spacing_mm])
        circ_accum_map = skim_fltr.gaussian(circ_accum_map, sigma=sigma_vox)
        # Rather than picking the highest accumulation as the centre
        # - detect top n_peaks
        max_locs = skim_feat.peak_local_max(circ_accum_map,
                                            threshold_abs=0.5 * np.max(circ_accum_map))
        # - combine with the passed centre of mass to select best candidate
        dist_vec = []
        accum_vec = []
        for x, y in zip(max_locs[:, 0], max_locs[:, 1]):
            dist_mm = np.sqrt(np.power((centre_mass_dx[0] - x) * in_plane_spacing_mm, 2) + np.power(
                (centre_mass_dx[1] - y) * in_plane_spacing_mm, 2))
            dist_vec.append(dist_mm)
            accum_vec.append(circ_accum_map[x, y])
        dist_vec = np.array(dist_vec)
        accum_vec = np.array(accum_vec)
        # normalise distance and accumulatino for a joint cost functions
        min_dist_mm = np.min(dist_vec)
        dist_vec = 1.0 - (dist_vec - min_dist_mm) / np.max(dist_vec - min_dist_mm)  # [0, 1] where 1 is best
        accum_vec = accum_vec / np.max(accum_vec)  # [0, 1] where 1 is best
        combined_vec = dist_vec + accum_vec
        y0, x0 = max_locs[np.argmax(combined_vec), :]
        #print("detect_centre_of_bottles: minimum distance to pre-detected centre: %0.3f mm" % min_dist_mm)

        # DEBUG
        ax2 = None
        if self.debug_vis:
            f, (ax1, ax2) = plt.subplots(1, 2)
            f.suptitle("ShapeDiffusionNIST::__detect_centre_of_bottle_array()")
            im_arr = sitk.GetArrayFromImage(im)
            ax1.imshow(circ_accum_map, cmap='jet', origin='lower')
            ax2.imshow(im_arr[im_arr.shape[0] // 2, :, :], cmap='gray', origin='lower')
            ax1.scatter(x0, y0, marker='x', color='k')
            ax2.scatter(x0, y0, marker='x', color='r')

        # if it is nearby an actual previously detected ROI centre, then use that instead of the one here
        for x, y in centre_dx_list:
            dist = np.sqrt(np.power((x0 - x) * in_plane_spacing_mm, 2) + np.power((y0 - y) * in_plane_spacing_mm, 2))
            if dist < threshold_euclid_mm:
                y0, x0 = y, x
                #print("detect_centre_of_bottles:  distance to pre-located centre: %0.3f mm" % dist)

        # DEBUG
        if self.debug_vis:
            ax2.scatter(x0, y0, marker='x', color='g')
            plt.pause(0.01)

        return int(x0), int(y0)

    def __categorise_centroids(self, bot_locs, cntr_dx,
                               bot_centre_smooth_map, in_plane_spacing_mm,
                               inner_circle_rad_mm, outer_circle_rad_mm,
                               band_mm):
        ctr_locs, inner_locs, outer_locs = [], [], []
        for x, y in zip(bot_locs[:, 0], bot_locs[:, 1]):
            dist_mm = np.sqrt(np.power((x - cntr_dx[0]) * in_plane_spacing_mm, 2) +
                              np.power((y - cntr_dx[1]) * in_plane_spacing_mm, 2))
            d_check_0 = dist_mm < band_mm  # centre vial
            d_check_in = ((inner_circle_rad_mm - band_mm) < dist_mm) and (
                    dist_mm < (inner_circle_rad_mm + band_mm))  # inner ring
            d_check_out = ((outer_circle_rad_mm - band_mm) < dist_mm) and (
                    dist_mm < (outer_circle_rad_mm + band_mm))  # outer ring
            if d_check_0:
                ctr_locs.append([x, y])
            if d_check_in:
                inner_locs.append([x, y])
            if d_check_out:
                outer_locs.append([x, y])
        # filter to the best 1 centre
        ctr_locs = np.array(ctr_locs)
        ctr_locs = self.__select_n_best_peaks(ctr_locs, bot_centre_smooth_map, n_best=1)
        # filter to the best 6 inner
        inner_locs = np.array(inner_locs)
        inner_locs = self.__select_n_best_peaks(inner_locs, bot_centre_smooth_map, n_best=6)
        # filter to the best 6 outer
        outer_locs = np.array(outer_locs)
        outer_locs = self.__select_n_best_peaks(outer_locs, bot_centre_smooth_map, n_best=6)
        return ctr_locs, inner_locs, outer_locs

    # function to select the best peaks based on the accumlation map
    def __select_n_best_peaks(self, locs, accum_map, n_best=13):
        peak_quality_list = []
        for peak_dx in range(locs.shape[0]):
            x_dx, y_dx = locs[peak_dx, :]
            peak_quality_list.append([np.sum(accum_map[x_dx - 2:x_dx + 2, y_dx - 2:y_dx + 2]), peak_dx])
        # sort based on peak quality
        peak_quality_list.sort(key=lambda row: row[0], reverse=True)
        max_locs_list = []
        for peak_quality, peak_dx in peak_quality_list[0:n_best]:
            x_dx, y_dx = locs[peak_dx, :]
            max_locs_list.append([x_dx, y_dx])
        return np.array(max_locs_list)
    # FUNCTIONS FOR BOTTLE DETECTION: END


    # FUNCTIONS FOR BOTTLE ARRAY ALIGNMENT: START
    def __centre_image_on_point(self, point_dx, im):
        o_mm = np.array(im.TransformIndexToPhysicalPoint(point_dx))
        # create a centred copy
        im0 = mu.sitk_deepcopy(im)
        im0.SetOrigin(np.array(im.GetOrigin()) - o_mm)
        # create a transform between original space and centred space
        tx0 = sitk.Euler3DTransform()
        tx0.SetTranslation(o_mm)
        # return
        return im0, tx0

    def __get_images_and_cntrds_into_shared_space(self,
                                                  cntr_dx_fix,   bot_cntr_dx_fix, botcap_cntr_dx_fix,
                                                  cntr_dx_mov, axi_cntr_dxs_mov, axi_inner_dxs_mov,
                                                  bot_cntr_dx_mov, botcap_cntr_dx_mov):
        im_fix = self.fixed_im
        im_mov = self.moving_im
        o_dx_fix = [cntr_dx_fix[0], cntr_dx_fix[1], bot_cntr_dx_fix]
        im0_fix, tx0_fix = self.__centre_image_on_point(o_dx_fix, im_fix)
        # get the location of the centre (moving)
        o_dx_mov = [cntr_dx_mov[0], cntr_dx_mov[1], bot_cntr_dx_mov]
        im0_mov, tx0_mov = self.__centre_image_on_point(o_dx_mov, im_mov)
        # flip the moving image if cap alignment suggests it is required
        centre_dx_list = copy.deepcopy(axi_cntr_dxs_mov)
        inner_centre_dx_list = copy.deepcopy(axi_inner_dxs_mov)
        # convert to 3D points
        centre_dx_list = [[x, y, bot_cntr_dx_mov] for x, y in centre_dx_list]
        inner_centre_dx_list = [[x, y, bot_cntr_dx_mov] for x, y in inner_centre_dx_list]
        # if bottle cap orientation is different then flip the moving data
        if not ((bot_cntr_dx_fix < botcap_cntr_dx_fix) == (bot_cntr_dx_mov < botcap_cntr_dx_mov)):
            def tranform_dx_list(c_dx_list, im, tx_matrix):
                new_centre_dx_list = []
                for b_cntr_dx in c_dx_list:
                    b_dx = [int(b_cntr_dx[0]), int(b_cntr_dx[1]), int(b_cntr_dx[2])]
                    b_mm = np.array(im.TransformIndexToPhysicalPoint(b_dx))
                    b_mm = np.array([b_mm[0], b_mm[1], b_mm[2], 1.0])
                    b0_mm = tx_matrix.dot(b_mm)[0:3]
                    new_centre_dx_list.append(np.array(im.TransformPhysicalPointToIndex(b0_mm)))
                return new_centre_dx_list
            # rotate the image
            tx0_mov.SetRotation(0.0, np.deg2rad(180.0), 0.0)
            im0_mov = sitk.Resample(im_mov, im0_mov, tx0_mov, sitk.sitkLinear)
            # transform/rotate each point dx in the list
            tx0_matrix = mu.get_homog_matrix_from_transform(tx0_mov.GetInverse())
            centre_dx_list = tranform_dx_list(centre_dx_list, im_mov, tx0_matrix)
            inner_centre_dx_list = tranform_dx_list(inner_centre_dx_list, im_mov, tx0_matrix)
        return im0_fix, tx0_fix, im0_mov, tx0_mov, centre_dx_list, inner_centre_dx_list

    def __find_optimal_angle(self,
                             im_fix, axi_cntr_dxs_fix, axi_inner_dxs_fix, bot_cntr_dx_fix,
                             im_mov, axi_cntr_dxs_mov, axi_inner_dxs_mov):
        # establish the reference intensities and testtube centres from fixed image
        intensity_centre_dxs = axi_cntr_dxs_fix
        distance_centre_dx = axi_cntr_dxs_fix
        distance_centre_mov_dx = axi_cntr_dxs_mov
        if self.inner_ring_only:
            intensity_centre_dxs = axi_inner_dxs_fix
            distance_centre_dx = axi_inner_dxs_fix
            distance_centre_mov_dx = axi_inner_dxs_mov

        intensities_fix = self.__extract_intensities(im_fix, intensity_centre_dxs, bot_cntr_dx_fix)
        # for point proximity
        fix_centre_dx_list = copy.deepcopy(distance_centre_dx)
        # convert to 3D points
        fix_centre_dx_list = [[x, y, bot_cntr_dx_fix] for x, y in fix_centre_dx_list]
        fix_centre_mm_list = []
        for b_cntr_dx in fix_centre_dx_list:
            b_dx = [int(b_cntr_dx[0]), int(b_cntr_dx[1]), int(b_cntr_dx[2])]
            b_mm = np.array(im_fix.TransformIndexToPhysicalPoint(b_dx))
            fix_centre_mm_list.append(b_mm)
        fix_tree = scispat.KDTree(np.array([[x, y] for x, y, z in fix_centre_mm_list]))

        # rotate the moving image and calculate differences in intensity and point distances
        intensity_metrics = []
        distance_metrics = []
        rot_angles = np.linspace(0.0, 359.0, 360)
        for rot_angle in rot_angles:
            # rotate the moving image
            tx_rot = sitk.Euler3DTransform()
            tx_rot.SetRotation(0.0, 0.0, np.deg2rad(rot_angle))
            im1_mov = sitk.Resample(im_mov, im_fix, tx_rot, sitk.sitkLinear)
            # pull out the intensities
            intensities_mov = self.__extract_intensities(im1_mov, intensity_centre_dxs, bot_cntr_dx_fix)
            # also rotate the points and calculate the point distances
            tx_rot_matrix = mu.get_homog_matrix_from_transform(tx_rot.GetInverse())
            rot_centre_mm_list = []
            for b_cntr_dx in distance_centre_mov_dx:
                b_dx = [int(b_cntr_dx[0]), int(b_cntr_dx[1]), int(b_cntr_dx[2])]
                b_mm = np.array(im_mov.TransformIndexToPhysicalPoint(b_dx))
                b_mm = np.array([b_mm[0], b_mm[1], b_mm[2], 1.0])
                b_rot_mm = tx_rot_matrix.dot(b_mm)[0:3]
                rot_centre_mm_list.append(b_rot_mm)
            # use a kdtree to efficiently find the closes points
            dist, indx = fix_tree.query(np.array([[x, y] for x, y, z in rot_centre_mm_list]))
            # judge accuracy of distance and intensity match
            intensity_metrics.append(np.linalg.norm(np.array(intensities_mov) - np.array(intensities_fix)))
            distance_metrics.append(np.linalg.norm(dist))
        # normalise metrics
        intensity_metrics = (intensity_metrics - np.min(intensity_metrics)) / np.max(
            (intensity_metrics - np.min(intensity_metrics)))
        distance_metrics = (distance_metrics - np.min(distance_metrics)) / np.max(
            (distance_metrics - np.min(distance_metrics)))
        combine_metrics = intensity_metrics + distance_metrics

        # pull out the best performing angle for rotation
        min_dx = np.argmin(combine_metrics)
        angle1 = rot_angles[min_dx]
        #print("Rotate by... ", angle1)
        tx1_mov = sitk.Euler3DTransform()
        tx1_mov.SetRotation(0.0, 0.0, np.deg2rad(angle1))

        if self.debug_vis:
            f, ax1 = plt.subplots(1, 1)
            f.suptitle("ShapeDiffusionNIST::__find_optimal_angle()")
            ax1.plot(rot_angles, distance_metrics, 'b')
            ax1.plot(rot_angles, intensity_metrics, 'r')
            ax1.plot(rot_angles, combine_metrics, 'k')
            ax1.plot(rot_angles[min_dx], combine_metrics[min_dx], 'ko')

        return angle1, tx1_mov

    def __extract_intensities(self, im, points,
                              c_slice,
                              bottle_rad_mm=10.0,           # slightly smaller than actual bottle to avoid edges and minor missalignment issues
                              bottle_half_height_mm=20.0):  # slightly smaller than actual bottle to avoid edges and minor missalignment issues
        # spatial details
        spc_arr = np.array(im.GetSpacing())
        in_plance_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
        thru_plane_spacing_mm = spc_arr[2]
        # region for extraction
        bottle_half_height_vox = int(np.round(bottle_half_height_mm / thru_plane_spacing_mm))
        bottle_rad_vox = int(np.round(bottle_rad_mm / in_plance_spacing_mm))
        # get the image data
        im_arr = sitk.GetArrayFromImage(im)
        # extract cyclindrical regions around points
        intensities = []
        for x, y in points:
            yy, xx = skim_draw.disk((y, x), radius=bottle_rad_vox,
                                    shape=(im_arr.shape[1], im_arr.shape[2]))
            intensities.append(
                np.mean(im_arr[c_slice - bottle_half_height_vox:c_slice + bottle_half_height_vox, yy, xx]))
        return intensities
    # FUNCTIONS FOR BOTTLE ARRAY ALIGNMENT: END


    # FUNCTIONS FOR ROI LABEL MAPPING: START
    def __link_template_roi_labels_to_target_image(self,
                                                   im0_fix, axi_cntr_dxs_fix, bot_cntr_dx_fix,
                                                   im0_mov, axi_cntr_dxs_mov, bot_cntr_dx_mov,
                                                   tx0_mov, tx1_mov,
                                                   euclid_threshold_mm=20.0):
        # STEP A: Associate template ROI labels with the detected bottles on the template/fixed image
        # ----------------------------------------------------------------------------------
        fixed_coords_and_label_list = self.__map_template_rois_to_fixed_centroids(axi_cntr_dxs_fix, bot_cntr_dx_fix)

        # STEP B: match the ROI labels in the template to the detected bottles in the moving image
        # ----------------------------------------------------------------------------------
        moving_coords_and_label_list = self.__map_template_rois_to_moving_centroids(im0_fix, fixed_coords_and_label_list,
                                                                                    im0_mov, axi_cntr_dxs_mov,
                                                                                    bot_cntr_dx_mov,
                                                                                    tx0_mov, tx1_mov,
                                                                                    euclid_threshold_mm)
        if self.debug_vis:
            # draw fixed and moving central slice with detected ROI labels
            f, axes = plt.subplots(1, 2)
            f.suptitle("ShapeDiffusionNIST::__link_template_roi_labels_to_target_image()")
            # FIXED
            spc_arr = np.array(im0_fix.GetSpacing())
            in_plane_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
            im_slc, extent = mu.get_slice_and_extent(im0_fix, OrientationOptions.AXI)
            axes[0].imshow(im_slc, extent=extent, cmap="gray", origin='lower')
            for x, y, z, label in fixed_coords_and_label_list:
                axes[0].annotate(label[7:],
                                 xy=(x * in_plane_spacing_mm, y * in_plane_spacing_mm),
                                 xytext=(x * in_plane_spacing_mm - 2, y * in_plane_spacing_mm + 2),
                                 color='r')
                axes[0].scatter(x * in_plane_spacing_mm, y * in_plane_spacing_mm, marker='.', color='r')
            # MOVING
            spc_arr = np.array(self.moving_im.GetSpacing())
            in_plane_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
            im_slc, extent = mu.get_slice_and_extent(self.moving_im, OrientationOptions.AXI)
            axes[1].imshow(im_slc, extent=extent, cmap="gray", origin='lower')
            for x, y, z, label in moving_coords_and_label_list:
                axes[1].annotate(label[7:],
                                 xy=(x * in_plane_spacing_mm, y * in_plane_spacing_mm),
                                 xytext=(x * in_plane_spacing_mm - 2, y * in_plane_spacing_mm + 2),
                                 color='r')
                axes[1].scatter(x * in_plane_spacing_mm, y * in_plane_spacing_mm, marker='.', color='r')
        return fixed_coords_and_label_list, moving_coords_and_label_list

    def __map_template_rois_to_fixed_centroids(self, axi_cntr_dxs_fix, bot_cntr_dx_fix):
        # match the ROI labels from the template file to the bottles
        roi_data = OrderedDict()
        for label, roi in self.roi_template.get_dw_roi_dict().items():
            roi_data[ROI_IDX_LABEL_MAP[label]] = (roi.ctr_vox_coords,
                                                  roi.radius_mm,
                                                  roi.height_mm)
        # loop over each detected bottle in the template and match to an ROI label (based on x,y coordinates)
        fixed_coords_and_label_list = []
        for cx, cy in axi_cntr_dxs_fix:
            label_dist_list = []
            for label, (ctr_coord, radius, height) in roi_data.items():
                cx0, cy0, cz0 = ctr_coord
                label_dist_list.append([label,
                                        np.sqrt(np.power(cx - cx0, 2.0) + np.power(cy - cy0, 2.0))])
            # get the closest
            label_dist_list.sort(key=lambda row: row[1])
            closest_bottle_label = label_dist_list[0][0]
            fixed_coords_and_label_list.append([cx, cy, bot_cntr_dx_fix, closest_bottle_label])  # x, y, z, label
        return fixed_coords_and_label_list

    # match the ROI labels in the template to the detected bottles in the moving image
    def __map_template_rois_to_moving_centroids(self,
                                                im0_fix, fixed_coords_and_label_list,
                                                im0_mov, axi_cntr_dxs_mov, bot_cntr_dx_mov,
                                                tx0_mov, tx1_mov,
                                                euclid_threshold_mm=20.0):
        # extract image data
        spc_arr = np.array(im0_mov.GetSpacing())
        in_plane_spacing_mov_mm = (spc_arr[0] + spc_arr[1]) / 2.

        # rotate the template points into the moving image
        tx0_matrix = mu.get_homog_matrix_from_transform(tx0_mov)
        tx1_matrix = mu.get_homog_matrix_from_transform(tx1_mov)
        # match centroids between moving and fixed images based on distance
        moving_coords_and_label_list = []
        for x, y, z, label in fixed_coords_and_label_list:
            b_dx = [int(x), int(y), int(z)]
            b_mm = np.array(im0_fix.TransformIndexToPhysicalPoint(b_dx))
            b_mm = np.array([b_mm[0], b_mm[1], b_mm[2], 1.0])
            b0_mov_mm = tx1_matrix.dot(b_mm)[0:3]
            b0_mov_mm = np.array([b0_mov_mm[0], b0_mov_mm[1], b0_mov_mm[2], 1.0])
            b_mov_mm = tx0_matrix.dot(b0_mov_mm)[0:3]
            cx0, cy0, cz0 = np.array(self.moving_im.TransformPhysicalPointToIndex(b_mov_mm))
            # loop over bottles detected in moving image and look for a matching position
            label_dist_list = []
            for bot_dx, (cx, cy) in enumerate(axi_cntr_dxs_mov):
                # measure distance
                label_dist_list.append([bot_dx,
                                        np.sqrt(np.power(cx - cx0, 2.0) + np.power(cy - cy0, 2.0))])
            # get the closest
            label_dist_list.sort(key=lambda row: row[1])
            closest_bottle_dx, closest_bottle_mm = label_dist_list[0][0], label_dist_list[0][
                                                                              1] * in_plane_spacing_mov_mm
            if (closest_bottle_mm < euclid_threshold_mm):
                #print("\t\tUsing detected bottle position for ", label)
                cx, cy = axi_cntr_dxs_mov[closest_bottle_dx]
                moving_coords_and_label_list.append([cx, cy, bot_cntr_dx_mov, label])
            else:
                #print("\t\tWARNING: Using TEMPLATE bottle position for ", label)
                moving_coords_and_label_list.append([cx0, cy0, bot_cntr_dx_mov, label])
        return moving_coords_and_label_list

    def __construct_moving_roi_image(self,
                                     fixed_coords_and_label_list, moving_coords_and_label_list,
                                     cap_mod_fix, cap_mod_mov):
        fixed_roi_dict = self.roi_template.get_dw_roi_dict()
        # get the image spacing for offset calcs
        spc_arr = np.array(self.fixed_im.GetSpacing())
        in_plane_spacing_fix_mm = (spc_arr[0] + spc_arr[1]) / 2.
        thru_plane_spacing_fix_mm = spc_arr[2]
        spc_arr = np.array(self.moving_im.GetSpacing())
        in_plane_spacing_mov_mm = (spc_arr[0] + spc_arr[1]) / 2.
        thru_plane_spacing_mov_mm = spc_arr[2]
        # check if moving bottle array is pointing in same direction as fixed
        is_same_orientation = (cap_mod_fix == cap_mod_mov)
        # loop over template ROIs and determine offset between template centroids and detected bottle centroids
        fixed_offset_dict_mm = OrderedDict()
        for roi_label, roi in fixed_roi_dict.items():
            x_dx, y_dx, z_dx = roi.get_centroid_dx()
            for cx, cy, cz, label in fixed_coords_and_label_list:
                if roi_label == DW_ROI_LABEL_IDX_MAP[label]:
                    dir_mod = 1.0
                    if not is_same_orientation:
                        # flip the offset to match the 180 deg rotation
                        dir_mod = -1.0
                    fixed_offset_dict_mm[roi_label] = ((x_dx-cx)*in_plane_spacing_fix_mm,
                                                       (y_dx-cy)*in_plane_spacing_fix_mm,
                                                       dir_mod*(z_dx-cz)*thru_plane_spacing_fix_mm)
        # create an ROIImage instance for the moving image
        moving_roi_dict = OrderedDict()
        # get transform for between fixed to moving rotations only
        tx_rot_matrix = mu.get_homog_matrix_from_transform(self.tx1_mov)
        if self.fine_tune_rois:
            # setup storage of variables needed for fine-tuning visualisation
            self.ft_centroid0_dict = OrderedDict()  # initial centroid from detection
            self.ft_centroid1_dict = OrderedDict()  # centroid following fine-tuning
            self.ft_bbox_dict = OrderedDict()  # bounding box for crop region
            self.ft_circle_dict = OrderedDict()  # detected circle centroids
            self.ft_curve_fit_dict = OrderedDict()  # curve fit to detected circles

        for r_dx, (roi_label, fixed_roi) in enumerate(fixed_roi_dict.items()):
            for cx, cy, cz, label in moving_coords_and_label_list:
                mov_label = DW_ROI_LABEL_IDX_MAP[label]
                if roi_label == mov_label:
                    # get the offset
                    dx_fix_mm, dy_fix_mm, dz_mov_mm = fixed_offset_dict_mm[roi_label]
                    # convert the x, y components from the fixed to the moving frame
                    b_mm = [dx_fix_mm, dy_fix_mm, 0.0, 1.0]
                    dx_mov_mm, dy_mov_mm = tx_rot_matrix.dot(b_mm)[0:2]
                    # change from mm to an index shift for ROI creation
                    dx = int(np.round(dx_mov_mm/in_plane_spacing_mov_mm))
                    dy = int(np.round(dy_mov_mm/in_plane_spacing_mov_mm))
                    dz = int(np.round(dz_mov_mm/thru_plane_spacing_mov_mm))
                    # create the new ROI as detected on the moving image with an approriate offset
                    moving_roi = None
                    assert isinstance(fixed_roi, ROICylinder), "ShapeDiffusionNIST::__construct_moving_roi_image(): only supports templates with ROICylinder objects (not %s)" \
                                                               "please check your template selection or use another ROI detection method." % type(fixed_roi)
                    if self.fine_tune_rois:
                        # perform another local ellipse detection to correct for spatial distortion
                        moving_roi = self.__fine_tune_roi_on_moving_image(cx, cy, cz,
                                                                          dx, dy, dz, fixed_roi)
                    else:
                        if isinstance(fixed_roi, ROICylinder):
                            # label, roi_index, ctr_vox_coords, radius_mm, height_mm
                            moving_roi = ROICylinder(fixed_roi.label, fixed_roi.roi_index,
                                                     [cx+dx, cy+dy, cz+dz],
                                                     fixed_roi.radius_mm,
                                                     fixed_roi.height_mm)
                    # add it to a dictionary
                    moving_roi_dict[roi_label] = moving_roi

        # make the ROIImage to return
        return ROIImage(self.moving_im, moving_roi_dict)

    def __fine_tune_roi_on_moving_image(self,
                                        cx, cy, cz,
                                        dx, dy, dz, fixed_roi,
                                        padd_inplane_mm=10.0, padd_thru_plane_mm=0.):
        # get spacing & assume spacing is isotropic in 2D plane
        spc_arr = np.array(self.moving_im.GetSpacing())
        in_plane_spacing_mm = (spc_arr[0] + spc_arr[1]) / 2.
        thru_plane_spacing_mm = spc_arr[2]
        # crop the image to the bottle based on the ROI centroid
        bot_half_height_vox = int(np.round((self.bottle_height_mm+2*padd_thru_plane_mm)/thru_plane_spacing_mm/2.0))
        bot_radius_vox = int(np.round((self.bottle_radius_mm+padd_inplane_mm)/in_plane_spacing_mm))
        bottle_arr = sitk.GetArrayFromImage(self.moving_im)
        padd_y, padd_x = None, None
        if (cy-bot_radius_vox) < 0:
            padd_y = -(cy-bot_radius_vox)
        if (cx-bot_radius_vox) < 0:
            padd_x = -(cx-bot_radius_vox)
        bottle_crop_arr = bottle_arr[
                          cz-bot_half_height_vox:cz+bot_half_height_vox,
                          np.max([0, cy-bot_radius_vox]):     cy+bot_radius_vox,
                          np.max([0, cx-bot_radius_vox]):     cx+bot_radius_vox]
        if padd_y is not None:
            bottle_crop_arr = np.pad(bottle_crop_arr, ((0, 0), (padd_y, 0), (0, 0)),
                                     'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        if padd_x is not None:
            bottle_crop_arr = np.pad(bottle_crop_arr, ((0, 0), (0, 0), (padd_x, 0)),
                                     'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        # look through the slices and fit an ellipse to each slice
        bot_centre_map = np.zeros_like(bottle_crop_arr, dtype=float)
        n_slices = bottle_crop_arr.shape[0]
        centroids = []
        for slice_dx in range(n_slices):
            bot_slc_arr = bottle_crop_arr[slice_dx, :, :]
            # edge detect in preparation for circle detection
            sigma_vox = self.edge_dect_sigma_mm / in_plane_spacing_mm
            slc_edges = skim_feat.canny(bot_slc_arr, sigma=sigma_vox,
                                        low_threshold=0.9,
                                        high_threshold=1.0)
            # hough transform
            bottle_radii_mm = [14., 15.]
            bot_accums, bot_cx, bot_cy, bot_radii = self.__circle_search(slc_edges, bottle_radii_mm,
                                                                         in_plane_spacing_mm, n_candidate_circles=1,
                                                                         enforce_min_distance=False)
            bot_centre_map[slice_dx, bot_cy, bot_cx] = bot_accums
            bot_accums, bot_cx, bot_cy, bot_radii = zip(*sorted(list(zip(bot_accums, bot_cx, bot_cy, bot_radii)), key=lambda x: x[0]))

            centroids.append([slice_dx, bot_cx[-1], bot_cy[-1]])


        centroids = np.array(centroids)
        cx_vec = centroids[:, 0]*thru_plane_spacing_mm
        cy_vec = centroids[:, 1]*in_plane_spacing_mm
        cz_vec = centroids[:, 2]*in_plane_spacing_mm

        w = np.ones(cx_vec.shape)
        w[0:2] = 0.05
        # w[2:4] = 0.2
        # w[-4:-2] = 0.2
        w[-1:] = 0.05
        p_coeff_y = np.polyfit(cx_vec, cy_vec, 1, w=w)
        p_coeff_z = np.polyfit(cx_vec, cz_vec, 1, w=w)
        #     ax1.plot(cx_vec, cy_vec, 'xk')
        #     ax1.plot(cx_vec, np.polyval(p_coeff_y, cx_vec), 'k')
        #     ax2.plot(cx_vec, cz_vec, 'xk')
        #     ax2.plot(cx_vec, np.polyval(p_coeff_z, cx_vec), 'k')
        cy_cor = np.polyval(p_coeff_y, cx_vec)
        cz_cor = np.polyval(p_coeff_z, cx_vec)

        # try smoothing rather than fitting
        win_len = int(np.round(len(cx_vec)*2./3.))
        if (win_len % 2) == 0:
            win_len = win_len + 1 # make it odd
        cy_cor = scisig.savgol_filter(cy_vec, window_length=win_len,
                                      polyorder=1)
        cz_cor = scisig.savgol_filter(cz_vec, window_length=win_len,
                                      polyorder=1)

        def reject_outliers(data, m=2):
            if np.std(data) < 1e-6:
                return data
            return data[abs(data - np.median(data)) < m * np.std(data)]

        cy_vec_clean = reject_outliers(cy_vec)
        cy_lin = np.mean(cy_vec_clean) * np.ones_like(cy_vec)

        cz_vec_clean = reject_outliers(cz_vec)
        cz_lin = np.mean(cz_vec_clean) * np.ones_like(cz_vec)

        # bounding box parameters
        self.ft_bbox_dict[fixed_roi.label] = [bot_half_height_vox, bot_radius_vox]
        # initial cylinder centroid
        self.ft_centroid0_dict[fixed_roi.label] = [cx, cy, cz]
        # fine-tuned cylinder centroid
        cy1 = int(np.round(cy_lin[0]/in_plane_spacing_mm))
        cz1 = int(np.round(cz_lin[0]/in_plane_spacing_mm))
        self.ft_centroid1_dict[fixed_roi.label] = [cx, cy1, cz1]
        # detected circles
        self.ft_circle_dict[fixed_roi.label] = [centroids[:, 0], centroids[:, 1], centroids[:, 2]]  # detected circle centroids
        self.ft_curve_fit_dict[fixed_roi.label] = [cx_vec/thru_plane_spacing_mm,
                                                   cy_cor/in_plane_spacing_mm,
                                                   cz_cor/in_plane_spacing_mm] # curve fit to the detected circles

        # return a cylinder matched to the new centroid
        return ROICylinder(fixed_roi.label, fixed_roi.roi_index,
                           [cx+(cy1-bot_radius_vox)+dx,
                            cy+(cz1-bot_radius_vox)+dy,
                            cz+dz],
                           fixed_roi.radius_mm,
                           fixed_roi.height_mm)



    # FUNCTIONS FOR ROI LABEL MAPPING: END

# tophat  from https://stackoverflow.com/questions/49878701/scipy-curve-fit-cannot-fit-a-tophat-function
def tophat(x, base_level, hat_level, hat_mid, hat_width):
    return np.where((hat_mid - hat_width / 2. < x) & (x < hat_mid + hat_width / 2.), hat_level, base_level)

