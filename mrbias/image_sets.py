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

from abc import ABC, abstractmethod
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from mrbias.misc_utils import ROI_IDX_LABEL_MAP, T1_ROI_LABEL_IDX_MAP, T2_ROI_LABEL_IDX_MAP, DW_ROI_LABEL_IDX_MAP


def check_image_same_grid(im_a, im_b):
    spacing_check = (im_a.GetSpacing() == im_b.GetSpacing())
    origin_check = (im_a.GetOrigin() == im_b.GetOrigin())
    direction_check = (im_a.GetDirection() == im_b.GetDirection())
    mu.log("image_sets::check_image_same_grid() [Spacing:%s, Origin:%s, Direct:%s]" %
           (spacing_check, origin_check, direction_check), LogLevels.LOG_INFO)
    return spacing_check and origin_check and direction_check

# a class for a derived parameter map (i.e. an inline ADC map)
class ParameterMapAbstract(ABC):
    def __init__(self,
                 map_label,
                 sitk_im,
                 parameter_name,
                 parameter_units,
                 series_instance_UID,
                 reference_imageset_label,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope=None, rescale_intercept=None,
                 scale_slope=None, scale_intercept=None,
                 date_acquired=None, time_acquired=None):
        self.label = map_label
        self.sitk_im = sitk_im
        # Parameter map details
        self.parameter_name = parameter_name
        self.parameter_units = parameter_units
        # Image details
        self.series_instance_UID = series_instance_UID
        # The reference imageset lable (from which it was derived)
        self.reference_imageset_label = reference_imageset_label
        # Time/date
        self.date = date_acquired
        self.time = time_acquired
        # Image Scaling Details
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
        self.scale_slope = scale_slope
        self.scale_intercept = scale_intercept
        self.bits_allocated = bits_allocated
        self.bits_stored = bits_stored
    def resample_to_match_image(self, sitk_im, interpol=sitk.sitkLinear):
        self.sitk_im = sitk.Resample(self.sitk_im, sitk_im,
                                     sitk.Euler3DTransform(), interpol)
    def __str__(self):
        return "%s (image=[size%s%s], SeriesUID=%s)" % (self.label, str(sitk.GetArrayFromImage(self.sitk_im).shape),
                                                        type(self.sitk_im), self.series_instance_UID)

class ADCMap(ParameterMapAbstract):
    def __init__(self,
                 map_label,
                 sitk_im,
                 series_instance_UID,
                 reference_imageset_label,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope=None, rescale_intercept=None,
                 scale_slope=None, scale_intercept=None,
                 date_acquired=None, time_acquired=None):
        super().__init__(map_label,
                         sitk_im,
                         parameter_name="ADC",
                         parameter_units="s/um^2",
                         series_instance_UID=series_instance_UID,
                         reference_imageset_label=reference_imageset_label,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         rescale_slope=rescale_slope, rescale_intercept=rescale_intercept,
                         scale_slope=scale_slope, scale_intercept=scale_intercept,
                         date_acquired=date_acquired, time_acquired=time_acquired)


# a class to bundle together multiple images
# their associated variable of interest (from DICOM tags)
# and label names and units
# also linked to a geometry image
class ImageSetAbstract(ABC):
    def __init__(self,
                 set_label,
                 sitk_im_list,
                 measurement_variable_list,
                 measurement_variable_name,
                 measurement_variable_units,
                 repetition_time_list,
                 geometry_image=None,
                 series_instance_UIDs=None, series_numbers=None,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope_list=None, rescale_intercept_list=None,
                 scale_slope_list=None, scale_intercept_list=None,
                 scanner_make=None, scanner_model=None, scanner_serial_number=None, scanner_field_strength=None,
                 date_acquired=None, time_acquired=None,
                 image_filepaths_list=None):
        self.label = set_label
        self.image_list = sitk_im_list
        self.image_filepaths_list = image_filepaths_list
        self.meas_var_list = measurement_variable_list
        self.meas_var_name = measurement_variable_name
        self.meas_var_units = measurement_variable_units
        self.repetition_time_list = repetition_time_list
        self.geometry_image = geometry_image
        self.series_instance_UIDs = series_instance_UIDs
        self.series_numbers = series_numbers
        self.roi_image = None
        self.roi_pmap_dict = None
        self.derived_pmap_list = None # this is for derived parameter maps which are calculated on the MRI scanner such as ADC maps
        # test if all the images in the set are in the same physical space
        assert len(self.image_list) > 0, "ImageSetAbstract[%s]: has an empty image list" % self.label
        ref_im = self.image_list[0]
        all_same_grid = True
        for meas_dx, im in enumerate(self.image_list):
            on_same_grid = check_image_same_grid(ref_im, im)
            if not on_same_grid:
                mu.log("ImageSetAbstract[%s]: image with %s=%s is on a different spatial grid"
                       % (self.label, self.meas_var_name, self.meas_var_list[meas_dx]), LogLevels.LOG_WARNING)
            all_same_grid = all_same_grid and on_same_grid
        assert all_same_grid,  "ImageSetAbstract[%s]: not all images in the set are in the same physical space!" % self.label
        # MRI scanner details
        self.scanner_make = scanner_make
        self.scanner_model = scanner_model
        self.scanner_serial_number = scanner_serial_number
        self.scanner_field_strength = scanner_field_strength
        # Time/date
        self.date = date_acquired
        self.time = time_acquired
        # Image Scaling Details
        self.rescale_slope_list     = rescale_slope_list
        self.rescale_intercept_list = rescale_intercept_list
        self.scale_slope_list       = scale_slope_list
        self.scale_intercept_list   = scale_intercept_list
        self.bits_allocated = bits_allocated
        self.bits_stored    = bits_stored

    def __str__(self):
        r_str = "ImageSetAbstract [%s : %s]" % (self.label, type(self))
        for im, x, suid in zip(self.image_list, self.meas_var_list, self.series_instance_UIDs):
            r_str = "%s\n\t\t\t\t\t%s=%s %s : (image=[size=%s%s], SeriesUID=%s)" % (r_str,
                                                                           self.meas_var_name, x, self.meas_var_units,
                                                                           str(sitk.GetArrayFromImage(im).shape), type(im), suid)
        # and the reference geometry
        if self.geometry_image is not None:
            r_str = "%s\n\t\t\tReferenced Geometry: \n\t\t\t\t\t%s" % (r_str, str(self.geometry_image))
        else:
            r_str = "%s\n\t\t\tReferenced Geometry: \n\t\t\t\t\tnone" % (r_str)

        # and any linked parameter maps
        if self.derived_pmap_list is not None:
            r_str = "%s\n\t\t\tDerived parameter maps:" % (r_str)
            for pmap in self.derived_pmap_list:
                r_str = "%s\n\t\t\t\t\t%s" % (r_str, str(pmap))
        return r_str

    def get_ROI_data(self):
        roi_dict = OrderedDict()
        if self.roi_image is None:
            mu.log("ImageSetAbstract::get_ROI_data() - no mask available", LogLevels.LOG_WARNING)
            return roi_dict
        # loop over rois in the mask and extract voxel data
        mask_arr = sitk.GetArrayFromImage(self.roi_image)
        mu.log("ImageSetAbstract:: get_ROI_data(): MASK values: [%d, %d]" %
               (np.min(mask_arr.flatten()), np.max(mask_arr.flatten())), LogLevels.LOG_INFO)
        mask_vals = np.unique(mask_arr.flatten())
        mask_vals = np.delete(mask_vals, np.where(mask_vals == 0)) # remove background
        n_meas = (len(self.meas_var_list))
        n_images = (len(self.image_list))
        assert n_meas == n_images, "ImageSetAbstract::get_ROI_data() - number of measurements (%d) and " \
                                   "images (%d) does not match" % (n_meas, n_images)
        for mask_val in mask_vals:
            if mask_val in mu.ROI_IDX_LABEL_MAP.keys():
                # Gets a boolean mask by comparing the mask np array with a given ROI number
                roi_boolean_mask = (mask_arr == mask_val)
                roi_mask_xyz = np.nonzero(roi_boolean_mask)
                # get the data type
                assert len(self.image_list) > 0, "ImageSetAbstract::get_ROI_data(): image_list has no images"
                image_array = sitk.GetArrayFromImage(self.image_list[0])
                dtype = type(image_array.flatten()[0])
                # iterate over the measurements/images and extract ROI voxel data
                voxel_data = np.zeros((n_meas, np.count_nonzero(roi_boolean_mask)), dtype=dtype)
                for meas_dx, image in enumerate(self.image_list):
                    image_array = sitk.GetArrayFromImage(image)
                    roi_im_data = image_array[roi_boolean_mask]
                    voxel_data[meas_dx] = roi_im_data.flatten()
                # if ROI parameter maps exist then pull out the pmap for this ROI
                roi_mask_pmap_dict = None
                if self.roi_pmap_dict is not None:
                    roi_mask_pmap_dict = OrderedDict()
                    for pmap_name, pmap_im in self.roi_pmap_dict.items():
                        pmap_arr = sitk.GetArrayFromImage(pmap_im)
                        roi_mask_pmap_dict[pmap_name] = pmap_arr[roi_boolean_mask]
                # if derived parameter maps (i.e. ADC maps) exist, then pull out the map for this ROI
                derived_map_dict = None
                if self.derived_pmap_list is not None:
                    derived_map_dict = OrderedDict()
                    for pmap in self.derived_pmap_list:
                        pmap_arr = sitk.GetArrayFromImage(pmap.sitk_im)
                        derived_map_dict[pmap.label] = (pmap_arr[roi_boolean_mask],
                                                        pmap.parameter_name,
                                                        pmap.parameter_units)


                # instantiate a ROI object and append to return structure
                im_set_roi = ImageSetROI(label=mu.ROI_IDX_LABEL_MAP[mask_val],
                                         voxel_data_array=voxel_data.transpose(),
                                         voxel_data_xyz=roi_mask_xyz,
                                         measurement_variable_vector=self.meas_var_list,
                                         measurement_variable_name=self.meas_var_name,
                                         measurement_variable_units=self.meas_var_units,
                                         voxel_pmap_dict=roi_mask_pmap_dict,
                                         derived_map_dict=derived_map_dict,
                                         rescale_slope_list=self.rescale_slope_list,
                                         rescale_intercept_list=self.rescale_intercept_list,
                                         scale_slope_list=self.scale_slope_list,
                                         scale_intercept_list=self.scale_intercept_list,
                                         bits_allocated=self.bits_allocated,
                                         bits_stored=self.bits_stored,
                                         scanner_make=self.scanner_make)
                roi_dict[mask_val] = im_set_roi
            else:
                mu.log("ImageSetAbstract::get_ROI_data() - skipping unknown roi_idx (%s)" % mask_val,
                       LogLevels.LOG_WARNING)
        return roi_dict


    @abstractmethod
    def update_ROI_mask(self):
        return None

    @abstractmethod
    def write_roi_pdf_page(self, c, sup_title=None, mask_override_sitk_im=None):
        return None

    def get_set_label(self):
        return self.label

    def get_images(self):
        return self.image_list

    def get_geometry_image(self):
        return self.geometry_image

    def get_roi_image(self):
        return self.roi_image

    def set_roi_image(self, roi_sitk_image):
        mu.log("ImageSetAbstract::set_roi_image(): check ROI mask matched the ImageSet spatial grid: %s" %
               check_image_same_grid(roi_sitk_image, self.image_list[0]), LogLevels.LOG_INFO)
        self.roi_image = roi_sitk_image

    def set_roi_parameter_map_dict(self, roi_pmap_dict):
        for p, pmap in roi_pmap_dict.items():
            mu.log("ImageSetAbstract::set_roi_parameter_map_dict(): check ROI pmap[%s] matched the ImageSet spatial grid: %s" %
                   (p, check_image_same_grid(pmap, self.image_list[0])), LogLevels.LOG_INFO)
        self.roi_pmap_dict = roi_pmap_dict

    def resample_derived_maps(self):
        if self.derived_pmap_list is not None:
            resampled_derived_maps_dict = OrderedDict()
            for pmap in self.derived_pmap_list:
                pmap.resample_to_match_image(self.image_list[0])

    def get_measurement_variables(self):
        return self.meas_var_list

    def get_label(self):
        return self.meas_var_name

    def get_label_units(self):
        return self.meas_var_units

    def add_derived_parameter_map(self, pmap):
        assert issubclass(type(pmap), ParameterMapAbstract), "ImageSetAbstract::add_derived_parameter_map(): expects parmeter map of type ParameterMapAbstract not %s" % str(type(pmap))
        if self.derived_pmap_list is None:
            self.derived_pmap_list = [pmap]
        else:
            self.derived_pmap_list.append(pmap)

    def _write_roi_pdf_page(self, c, sup_title=None,
                            mask_override_sitk_im=None,
                            all_roi_values=None,
                            parameter_map_name=None,
                            parameter_map=None):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font
        if sup_title is None:
            sup_title = "IMAGESET [%s]" % (self.get_set_label())
        if parameter_map_name is not None:
            sup_title = "%s (pmap = %s)" % (sup_title, parameter_map_name)
        # get some data from the image set
        image_list = self.get_images()
        base_im = image_list[0]
        im_spacing = base_im.GetSpacing()
        base_arr = sitk.GetArrayFromImage(base_im)
        if mask_override_sitk_im is None:
            mask_im = self.get_roi_image()
        else:
            mask_im = mask_override_sitk_im
        mask_arr = sitk.GetArrayFromImage(mask_im)

        roi_vals = np.unique(mask_arr.flatten())[1:]

        # draw the ROI location summary figure
        # -----------------------------------------------------------
        # setup figure
        n_rois = len(roi_vals)
        n_rows = 2
        rois_per_row = int(np.ceil(n_rois / float(n_rows)))
        # f, axes_arr = plt.subplots(n_rows*2, rois_per_row)
        f = plt.figure()  # constrained_layout=True)
        f.suptitle(sup_title)
        f.set_size_inches(14, 8)
        # setup the layout
        gs0 = gridspec.GridSpec(2, 2, figure=f,
                                bottom=0.1, top=0.85, left=0.1, right=0.9,
                                width_ratios=[1, 2], hspace=0.5)

        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[:, 0])
        if n_rois:
            gs01 = gridspec.GridSpecFromSubplotSpec(2, rois_per_row, subplot_spec=gs0[0, 1],
                                                    hspace=0.01, wspace=0.01)
            gs02 = gridspec.GridSpecFromSubplotSpec(2, rois_per_row, subplot_spec=gs0[1, 1],
                                                    hspace=0.01, wspace=0.01)
        # make the main image axes
        ax_glob = f.add_subplot(gs00[:, :])
        # and for the zoom axes
        if n_rois:
            ax_row_1a = [f.add_subplot(gs01[0, x]) for x in range(rois_per_row)]
            ax_row_1b = [f.add_subplot(gs01[1, x]) for x in range(rois_per_row)]
            ax_row_2a = [f.add_subplot(gs02[0, x]) for x in range(rois_per_row)]
            ax_row_2b = [f.add_subplot(gs02[1, x]) for x in range(rois_per_row)]

        # generate the color settings based on mask values
        cs = mu.ColourSettings(roi_index_list=roi_vals)

        # draw the main slice for reference
        c_x, c_y, c_z = np.array(mask_arr.shape)//2
        if n_rois:
            roi_xyz = np.nonzero(mask_arr == roi_vals[0])
            c_x, c_y, c_z = np.median(roi_xyz[0]).astype(int), \
                            np.median(roi_xyz[1]).astype(int), \
                            np.median(roi_xyz[2]).astype(int)
        roi_slice = mask_arr[c_x, :, :]
        im_slice = base_arr[c_x, :, :]
        i_gray = ax_glob.imshow(im_slice, cmap='gray')
        g_vmin, g_vmax = i_gray.get_clim()
        i = ax_glob.imshow(np.ma.masked_where(roi_slice == 0, roi_slice),
                           cmap=cs.get_cmap(), vmin=cs.get_vmin(all_roi_values), vmax=cs.get_vmax(all_roi_values),
                           interpolation='none',
                           alpha=cs.get_cmap_alpha())
        ax_glob.axis('off')
        ticks = list(range(cs.get_vmin(all_roi_values)+np.uint16(1), cs.get_vmax(all_roi_values)))
        ticklabels = [ROI_IDX_LABEL_MAP[x] for x in ticks]
        cb = plt.colorbar(mappable=i, ax=ax_glob,
                          ticks=ticks)
        cb.set_ticklabels(ticklabels=ticklabels)


        # loop over the axes and plot the individual roi regions
        if n_rois:
            roi_dx = 0
            for ax_row_dx, axes_row in zip([0, 2],
                                           [ax_row_1a, ax_row_2a]):
                for ax_dx in range(rois_per_row):
                    ax = axes_row[ax_dx]
                    # if its a 3D image try again
                    sag_ax = None
                    if ax_row_dx == 0:
                        sag_ax = ax_row_1b[ax_dx]
                    if ax_row_dx == 2:
                        sag_ax = ax_row_2b[ax_dx]

                    if (roi_dx < n_rois):
                        roi_xyz = np.nonzero(mask_arr == roi_vals[roi_dx])
                        c_x, c_y, c_z = np.median(roi_xyz[0]).astype(int), \
                                        np.median(roi_xyz[1]).astype(int), \
                                        np.median(roi_xyz[2]).astype(int)
                        c_x_min, c_y_min, c_z_min = np.min(roi_xyz[0]).astype(int), \
                                                    np.min(roi_xyz[1]).astype(int), \
                                                    np.min(roi_xyz[2]).astype(int)
                        c_x_max, c_y_max, c_z_max = np.max(roi_xyz[0]).astype(int), \
                                                    np.max(roi_xyz[1]).astype(int), \
                                                    np.max(roi_xyz[2]).astype(int)
                        padd_yz = int(np.round(10. / im_spacing[0]))
                        padd_x = int(np.round(10. / im_spacing[2]))
                        extent_x_a = c_x_min - padd_x if (c_x_min - padd_x) >= 0 else 0
                        extent_x_b = c_x_max + padd_x if (c_x_max + padd_x) <= base_arr.shape[0] else base_arr.shape[0]
                        extent_y_a = c_y_min - padd_yz if (c_y_min - padd_yz) >= 0 else 0
                        extent_y_b = c_y_max + padd_yz if (c_y_max + padd_yz) <= base_arr.shape[1] else base_arr.shape[1]
                        extent_z_a = c_z_min - padd_yz if (c_z_min - padd_yz) >= 0 else 0
                        extent_z_b = c_z_max + padd_yz if (c_z_max + padd_yz) <= base_arr.shape[2] else base_arr.shape[2]

                        zoom_arr = base_arr[c_x, extent_y_a:extent_y_b, extent_z_a:extent_z_b]
                        zoom_extent = [0, zoom_arr.shape[0] * im_spacing[1],
                                       0, zoom_arr.shape[1] * im_spacing[0]]
                        roi_slice = mask_arr[c_x, extent_y_a:extent_y_b, extent_z_a:extent_z_b]
                        # prepare the overlay data to be used (ROI label, or a parameter map)
                        overlay_slice, vmin, vmax = None, None, None
                        if parameter_map is not None:
                            pmap_arr = sitk.GetArrayFromImage(parameter_map)
                            overlay_slice = pmap_arr[c_x, extent_y_a:extent_y_b, extent_z_a:extent_z_b]
                            if parameter_map_name == "height_dist_mm":
                                vmin = -20.0
                                vmax =  20.0
                                olay_cmap = "RdYlGn"
                                olay_cmap_alpha = 0.7
                            else:
                                vmin = 0.
                                vmax = 15.0
                                olay_cmap = "nipy_spectral"
                                olay_cmap_alpha = 0.7
                        else:
                            overlay_slice = roi_slice
                            vmin = cs.get_vmin(all_roi_values)
                            vmax = cs.get_vmax(all_roi_values)
                            olay_cmap = cs.get_cmap()
                            olay_cmap_alpha = cs.get_cmap_alpha()
                        ax.imshow(zoom_arr,
                                  extent=zoom_extent,
                                  cmap='gray', vmin=g_vmin, vmax=g_vmax)
                        roi_i = ax.imshow(np.ma.masked_where(roi_slice == 0, overlay_slice),
                                          extent=zoom_extent,
                                          cmap=olay_cmap, vmin=vmin, vmax=vmax,
                                          interpolation='none',
                                          alpha=olay_cmap_alpha)
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                        ax.set_title("%s\n(n=%d)" % (ROI_IDX_LABEL_MAP[roi_vals[roi_dx]], np.count_nonzero(mask_arr == roi_vals[roi_dx])))
                        if ax_dx == 0:
                            ax.set_ylabel("axial")

                        if sag_ax is not None:
                            if (c_x_max - c_x_min) > 1:
                                zoom_arr = base_arr[extent_x_a:extent_x_b, c_y, extent_z_a:extent_z_b]
                                zoom_extent = [0, zoom_arr.shape[1] * im_spacing[0],
                                               0, zoom_arr.shape[0] * im_spacing[2]]
                                sag_ax.imshow(zoom_arr,
                                              extent=zoom_extent,
                                              cmap='gray', vmin=g_vmin, vmax=g_vmax)
                                roi_slice = mask_arr[extent_x_a:extent_x_b, c_y, extent_z_a:extent_z_b]
                                # prepare the overlay data to be used (ROI label, or a parameter map)
                                overlay_slice = None
                                if parameter_map is not None:
                                    pmap_arr = sitk.GetArrayFromImage(parameter_map)
                                    overlay_slice = pmap_arr[extent_x_a:extent_x_b, c_y, extent_z_a:extent_z_b]
                                else:
                                    overlay_slice = roi_slice
                                sag_ax.imshow(np.ma.masked_where(roi_slice == 0, overlay_slice),
                                              extent=zoom_extent,
                                              cmap=olay_cmap, vmin=vmin, vmax=vmax,
                                              interpolation='none',
                                              alpha=olay_cmap_alpha)
                                if ax_dx == 0:
                                    sag_ax.set_ylabel("sagittal")

                                sag_ax.set_xticks([])
                                sag_ax.set_xticklabels([])
                                sag_ax.set_yticks([])
                                sag_ax.set_yticklabels([])
                            else:
                                # hide any unused axes
                                sag_ax.axis('off')

                        roi_dx = roi_dx + 1
                    else:
                        # hide any unused axes
                        ax.axis('off')
                        if sag_ax is not None:
                            sag_ax.axis('off')
            # add colorbar for parameter map
            if parameter_map is not None:
                cb = plt.colorbar(mappable=roi_i, ax=ax_row_1b, location='bottom')
                cb.set_label(parameter_map_name)

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
        # -------------------------------------------------------------
        c.showPage()  # new page


class ImageSetROI(object):
    def __init__(self, label,
                 voxel_data_array, voxel_data_xyz, measurement_variable_vector,
                 measurement_variable_name, measurement_variable_units,
                 voxel_pmap_dict = None, derived_map_dict=None,
                 rescale_slope_list=None, rescale_intercept_list=None,
                 scale_slope_list=None, scale_intercept_list=None,
                 bits_allocated=None, bits_stored=None,
                 scanner_make=None
                 ):
        mu.log("\t\t\tImageSetROI::__init__(): creating %s with %d voxels %s (each with %d measurements)" %
               (label, voxel_data_array.shape[0], type(voxel_data_array.flatten()[0]), voxel_data_array.shape[1]),
               LogLevels.LOG_INFO)
        self.label = label
        self.voxel_data_array = voxel_data_array
        self.voxel_data_xyz = voxel_data_xyz
        self.meas_var_vector = measurement_variable_vector
        self.meas_var_name = measurement_variable_name
        self.meas_var_units = measurement_variable_units
        # for storing additional voxel information
        self.voxel_pmap_dict = voxel_pmap_dict
        # for storing derived maps (i.e. ADC map)
        # - format is [key] -> (data tuple)
        #   [map_set_label] -> [voxel_data, map_name, map_units]
        self.derived_map_dict = derived_map_dict
        # data acquisition and scaling information
        self.rescale_slope_list = rescale_slope_list
        self.rescale_intercept_list = rescale_intercept_list
        self.scale_slope_list = scale_slope_list
        self.scale_intercept_list = scale_intercept_list
        self.bits_allocated = bits_allocated
        self.bits_stored = bits_stored
        self.scanner_make = scanner_make


class ImageBasic(object):
    def __init__(self,
                 label, sitk_im,
                 series_instance_UID=None,
                 series_number=None,
                 bits_allocated=None,
                 bits_stored=None,
                 rescale_slope=None,
                 rescale_intercept=None,
                 scale_slope=None,
                 scale_intercept=None,
                 filepath_list=None):
        self.label = label
        self.im = sitk_im
        self.series_instance_UID = series_instance_UID
        self.series_number = series_number
        self.filepath_list = filepath_list
        # image data information
        self.bits_allocated = bits_allocated
        self.bits_stored = bits_stored
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
        self.scale_slope = scale_slope
        self.scale_intercept = scale_intercept
    def get_image(self):
        return self.im
    def get_label(self):
        return self.label
    def get_filepath_list(self):
        return self.filepath_list
    def __str__(self):
        return "%s (image=[size%s%s], SeriesUID=%s)" % (self.label, str(sitk.GetArrayFromImage(self.im).shape),
                                                        type(self.im), self.series_instance_UID)


class ImageGeometric(ImageBasic):
    def __init__(self,
                 label, sitk_im,
                 series_instance_UID=None,
                 series_number=None,
                 bits_allocated=None,
                 bits_stored=None,
                 rescale_slope=None,
                 rescale_intercept=None,
                 scale_slope=None,
                 scale_intercept=None,
                 filepath_list=None):
        super().__init__(label, sitk_im, series_instance_UID, series_number,
                         bits_allocated, bits_stored,
                         rescale_slope, rescale_intercept, scale_slope, scale_intercept, filepath_list)
        self.roi_mask_image_PD, self.roi_prop_dict_PD = None, None
        self.roi_mask_image_T1, self.roi_prop_dict_T1 = None, None
        self.roi_mask_image_T2, self.roi_prop_dict_T2 = None, None
        self.roi_mask_image_DW, self.roi_prop_dict_DW = None, None

    def set_proton_density_mask(self, mask_sitk_image):
        assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_proton_density_mask() expects a" \
                                                        "SimpleITK image"
        self.roi_mask_image_PD = mask_sitk_image

    def set_T1_mask(self, mask_sitk_image, mask_prop_dict):
        assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_T1_mask() expects a" \
                                                        "SimpleITK image"
        # check the geom image and mask are on the same grid
        assert check_image_same_grid(self.get_image(), mask_sitk_image), \
            "ImageGeometric::set_T1_mask(): mask image grid does not match the geometric image!"
        mask_arr = sitk.GetArrayFromImage(mask_sitk_image)
        mu.log("ImageGeometric::set_T1_mask(): detected mask value range [%d, %d]" %
               (np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.roi_mask_image_T1 = mask_sitk_image
        # check the geom image and each of the parmeter maps are on the same grid
        self.roi_prop_dict_T1 = OrderedDict()
        for p, pmap_sitk_image in mask_prop_dict.items():
            assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_T1_mask() expects parameter map [%s] to" \
                                                            "be a SimpleITK image" % p
            assert check_image_same_grid(self.get_image(), pmap_sitk_image), \
                "ImageGeometric::set_T1_mask(): parameter map [%s] image grid does not match the geometric image!" % p
            self.roi_prop_dict_T1[p] = pmap_sitk_image

    def set_T2_mask(self, mask_sitk_image, mask_prop_dict):
        assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_T2_mask() expects a" \
                                                        "SimpleITK image"
        # check the geom image and mask are on the same grid
        assert check_image_same_grid(self.get_image(), mask_sitk_image), \
            "ImageGeometric::set_T2_mask(): mask image grid does not match the geometric image!"
        mask_arr = sitk.GetArrayFromImage(mask_sitk_image)
        mu.log("ImageGeometric::set_T2_mask(): detected mask value range [%d, %d]" %
               (np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.roi_mask_image_T2 = mask_sitk_image
        # check the geom image and each of the parmeter maps are on the same grid
        self.roi_prop_dict_T2 = OrderedDict()
        for p, pmap_sitk_image in mask_prop_dict.items():
            assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_T2_mask() expects parameter map [%s] to" \
                                                            "be a SimpleITK image" % p
            assert check_image_same_grid(self.get_image(), pmap_sitk_image), \
                "ImageGeometric::set_T2_mask(): parameter map [%s] image grid does not match the geometric image!" % p
            self.roi_prop_dict_T2[p] = pmap_sitk_image

    def set_DW_mask(self, mask_sitk_image, mask_prop_dict):
        assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_DW_mask() expects a" \
                                                        "SimpleITK image"
        # check the geom image and mask are on the same grid
        assert check_image_same_grid(self.get_image(), mask_sitk_image), \
            "ImageGeometric::set_DW_mask(): mask image grid does not match the geometric image!"
        mask_arr = sitk.GetArrayFromImage(mask_sitk_image)
        mu.log("ImageGeometric::set_DW_mask(): detected mask value range [%d, %d]" %
               (np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.roi_mask_image_DW = mask_sitk_image
        # check the geom image and each of the parmeter maps are on the same grid
        self.roi_prop_dict_DW = OrderedDict()
        for p, pmap_sitk_image in mask_prop_dict.items():
            assert isinstance(mask_sitk_image, sitk.Image), "ImageGeometric::set_DW_mask() expects parameter map [%s] to" \
                                                            "be a SimpleITK image" % p
            assert check_image_same_grid(self.get_image(), pmap_sitk_image), \
                "ImageGeometric::set_DW_mask(): parameter map [%s] image grid does not match the geometric image!" % p
            self.roi_prop_dict_DW[p] = pmap_sitk_image

    def get_T1_mask(self):
        assert self.roi_mask_image_T1 is not None, "ImageGeometric::get_T1_mask(): no mask available"
        return self.roi_mask_image_T1
    def get_T2_mask(self):
        assert self.roi_mask_image_T2 is not None, "ImageGeometric::get_T2_mask(): no mask available"
        return self.roi_mask_image_T2
    def get_DW_mask(self):
        assert self.roi_mask_image_DW is not None, "ImageGeometric::get_DW_mask(): no mask available"
        return self.roi_mask_image_DW

    def get_T1_mask_and_pmaps(self):
        assert self.roi_prop_dict_T1 is not None, "ImageGeometric::get_T1_mask_and_pmaps(): no property map dict available"
        return self.get_T1_mask(), self.roi_prop_dict_T1
    def get_T2_mask_and_pmaps(self):
        assert self.roi_prop_dict_T2 is not None, "ImageGeometric::get_T2_mask_and_pmaps(): no property map dict available"
        return self.get_T2_mask(), self.roi_prop_dict_T2
    def get_DW_mask_and_pmaps(self):
        assert self.roi_prop_dict_DW is not None, "ImageGeometric::get_DW_mask_and_pmaps(): no property map dict available"
        return self.get_DW_mask(), self.roi_prop_dict_DW


class ImageProtonDensity(ImageBasic):
    def __init__(self,
                 label, sitk_im,
                 geometry_sitk_im=None,
                 series_instance_UID=None,
                 series_number=None,
                 bits_allocated=None,
                 bits_stored=None,
                 rescale_slope=None,
                 rescale_intercept=None):
        super().__init__(label, sitk_im, series_instance_UID, series_number,
                         bits_allocated, bits_stored, rescale_slope, rescale_intercept)
        self.geometry_image = geometry_sitk_im
        self.roi_image = None
    def __str__(self):
        return "%s\n\t\t\t\t\tReferenced Geometry: %s" % (super().__str__(),
                                                          str(self.geometry_image))

class ImageSetT1VIR(ImageSetAbstract):
    def __init__(self,
                 set_label,
                 sitk_im_list,
                 inversion_time_list,
                 repetition_time_list,
                 geometry_image,
                 series_instance_UIDs=None, series_numbers=None,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope_list=None,
                 rescale_intercept_list=None,
                 scale_slope_list=None,
                 scale_intercept_list=None,
                 scanner_make=None, scanner_model=None, scanner_serial_number=None, scanner_field_strength=None,
                 study_date=None, study_time=None,
                 image_filepaths_list=None):
        super().__init__(set_label=set_label,
                         sitk_im_list=sitk_im_list,
                         measurement_variable_list=inversion_time_list,
                         measurement_variable_name="inversion time",
                         repetition_time_list=repetition_time_list,
                         measurement_variable_units="ms",
                         geometry_image=geometry_image,
                         series_instance_UIDs=series_instance_UIDs, series_numbers=series_numbers,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         rescale_slope_list=rescale_slope_list, rescale_intercept_list=rescale_intercept_list,
                         scale_slope_list=scale_slope_list, scale_intercept_list=scale_intercept_list,
                         scanner_make=scanner_make, scanner_model=scanner_model, scanner_serial_number=scanner_serial_number,
                         scanner_field_strength=scanner_field_strength, date_acquired=study_date, time_acquired=study_time,
                         image_filepaths_list=image_filepaths_list)
    def get_inversion_times(self):
        return super().get_measurement_variables()
    def set_T1_roi_mask(self, roi_sitk_image):
        super().set_roi_image(roi_sitk_image)
    def set_T1_pmap_dict(self, roi_pmap_dict):
        super().set_roi_parameter_map_dict(roi_pmap_dict)
    def get_T1_pmap_dict(self):
        return self.roi_pmap_dict
    def update_ROI_mask(self):
        if self.geometry_image is None:
            mu.log("ImageSetT1VIR::update_ROI_mask(): no geometry image set", LogLevels.LOG_WARNING)
            return None
        if not (len(self.image_list) > 0):
            mu.log("ImageSetT1VIR::update_ROI_mask(): no images in set", LogLevels.LOG_WARNING)
            return None
        t1_mask_im, t1_pmap_dict = self.geometry_image.get_T1_mask_and_pmaps()
        mask_arr = sitk.GetArrayFromImage(t1_mask_im)
        mu.log("ImageSetT1VIR[%s]::update_ROI_mask() : orig mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        resampled_mask = sitk.Resample(t1_mask_im,
                                       self.image_list[0],
                                       sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        mask_arr = sitk.GetArrayFromImage(resampled_mask)
        mu.log("ImageSetT1VIR[%s]::update_ROI_mask() : resampled mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.set_T1_roi_mask(resampled_mask)
        # resample and store any associated parameter maps
        resampled_pmap_dict = OrderedDict()
        for p, pmap in t1_pmap_dict.items():
            resampled_pmap_dict[p] = sitk.Resample(pmap,
                                                   self.image_list[0],
                                                   sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        self.set_T1_pmap_dict(resampled_pmap_dict)
        # resample and store any associated parameter maps
        super().resample_derived_maps()

    def write_roi_pdf_page(self, c, sup_title=None, mask_override_sitk_im=None, include_pmap_pages=False):
        self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                 all_roi_values=list(T1_ROI_LABEL_IDX_MAP.values()))
        t1_pmap_dict = self.get_T1_pmap_dict()
        if include_pmap_pages and (t1_pmap_dict is not None):
            for param_name, pmap in t1_pmap_dict.items():
                self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                         all_roi_values=list(T1_ROI_LABEL_IDX_MAP.values()),
                                         parameter_map_name=param_name,
                                         parameter_map=pmap)


class ImageSetT1VFA(ImageSetAbstract):
    def __init__(self,
                 set_label,
                 sitk_im_list,
                 flip_angle_list,
                 repetition_time_list,
                 geometry_image,
                 series_instance_UIDs=None, series_numbers=None,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope_list=None,
                 rescale_intercept_list=None,
                 scale_slope_list=None,
                 scale_intercept_list=None,
                 scanner_make=None, scanner_model=None, scanner_serial_number=None, scanner_field_strength=None,
                 study_date=None, study_time=None,
                 image_filepaths_list=None):
        super().__init__(set_label=set_label,
                         sitk_im_list=sitk_im_list,
                         measurement_variable_list=flip_angle_list,
                         measurement_variable_name="flip angle",
                         measurement_variable_units="deg.",
                         repetition_time_list=repetition_time_list,
                         geometry_image=geometry_image,
                         series_instance_UIDs=series_instance_UIDs, series_numbers=series_numbers,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         rescale_slope_list=rescale_slope_list, rescale_intercept_list=rescale_intercept_list,
                         scale_slope_list=scale_slope_list, scale_intercept_list=scale_intercept_list,
                         scanner_make=scanner_make, scanner_model=scanner_model, scanner_serial_number=scanner_serial_number,
                         scanner_field_strength=scanner_field_strength, date_acquired=study_date, time_acquired=study_time,
                         image_filepaths_list=image_filepaths_list)
    def get_flip_angles(self):
        return super().get_measurement_variables()
    def set_T1_roi_mask(self, roi_sitk_image):
        super().set_roi_image(roi_sitk_image)
    def set_T1_pmap_dict(self, roi_pmap_dict):
        super().set_roi_parameter_map_dict(roi_pmap_dict)
    def get_T1_pmap_dict(self):
        return self.roi_pmap_dict
    def update_ROI_mask(self):
        if self.geometry_image is None:
            mu.log("ImageSetT1VFA::update_ROI_mask(): no geometry image set", LogLevels.LOG_WARNING)
        if not (len(self.image_list) > 0):
            mu.log("ImageSetT1VFA::update_ROI_mask(): no images in set", LogLevels.LOG_WARNING)
            return None
        t1_mask_im, t1_pmap_dict = self.geometry_image.get_T1_mask_and_pmaps()
        mask_arr = sitk.GetArrayFromImage(t1_mask_im)
        mu.log("ImageSetT1VFA[%s]::update_ROI_mask() : orig mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        resampled_mask = sitk.Resample(t1_mask_im,
                                       self.image_list[0],
                                       sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        mask_arr = sitk.GetArrayFromImage(resampled_mask)
        mu.log("ImageSetT1VFA[%s]::update_ROI_mask() : resampled mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.set_T1_roi_mask(resampled_mask)
        # resample and store any associated parameter maps
        resampled_pmap_dict = OrderedDict()
        for p, pmap in t1_pmap_dict.items():
            resampled_pmap_dict[p] = sitk.Resample(pmap,
                                                   self.image_list[0],
                                                   sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        self.set_T1_pmap_dict(resampled_pmap_dict)
        # resample and store any associated parameter maps
        super().resample_derived_maps()
    def write_roi_pdf_page(self, c, sup_title=None, mask_override_sitk_im=None, include_pmap_pages=False):
        self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                 all_roi_values=list(T1_ROI_LABEL_IDX_MAP.values()))
        t1_pmap_dict = self.get_T1_pmap_dict()
        if include_pmap_pages and (t1_pmap_dict is not None):
            for param_name, pmap in t1_pmap_dict.items():
                self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                         all_roi_values=list(T1_ROI_LABEL_IDX_MAP.values()),
                                         parameter_map_name=param_name,
                                         parameter_map=pmap)


class ImageSetT2MSE(ImageSetAbstract):
    def __init__(self,
                 set_label,
                 sitk_im_list,
                 echo_time_list,
                 repetition_time_list,
                 geometry_image,
                 series_instance_UIDs=None, series_numbers=None,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope_list=None,
                 rescale_intercept_list=None,
                 scale_slope_list=None,
                 scale_intercept_list=None,
                 scanner_make=None, scanner_model=None, scanner_serial_number=None, scanner_field_strength=None,
                 study_date=None, study_time=None,
                 image_filepaths_list=None):
        super().__init__(set_label=set_label,
                         sitk_im_list=sitk_im_list,
                         measurement_variable_list=echo_time_list,
                         measurement_variable_name="echo time",
                         measurement_variable_units="ms",
                         repetition_time_list=repetition_time_list,
                         geometry_image=geometry_image,
                         series_instance_UIDs=series_instance_UIDs, series_numbers=series_numbers,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         rescale_slope_list=rescale_slope_list, rescale_intercept_list=rescale_intercept_list,
                         scale_slope_list=scale_slope_list, scale_intercept_list=scale_intercept_list,
                         scanner_make=scanner_make, scanner_model=scanner_model, scanner_serial_number=scanner_serial_number,
                         scanner_field_strength=scanner_field_strength, date_acquired=study_date, time_acquired=study_time,
                         image_filepaths_list=image_filepaths_list)
    def get_echo_times(self):
        return super().get_measurement_variables()
    def set_T2_roi_mask(self, roi_sitk_image):
        super().set_roi_image(roi_sitk_image)
    def set_T2_pmap_dict(self, roi_pmap_dict):
        super().set_roi_parameter_map_dict(roi_pmap_dict)
    def get_T2_pmap_dict(self):
        return self.roi_pmap_dict
    def update_ROI_mask(self):
        if self.geometry_image is None:
            mu.log("ImageSetT2MSE::update_ROI_mask(): no geometry image set", LogLevels.LOG_WARNING)
            return None
        if not (len(self.image_list) > 0):
            mu.log("ImageSetT2MSE::update_ROI_mask(): no images in set", LogLevels.LOG_WARNING)
            return None
        t2_mask_im, t2_pmap_dict = self.geometry_image.get_T2_mask_and_pmaps()
        mask_arr = sitk.GetArrayFromImage(t2_mask_im)
        mu.log("ImageSetT2MSE[%s]::update_ROI_mask() : orig mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        resampled_mask = sitk.Resample(t2_mask_im,
                                       self.image_list[0],
                                       sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        mask_arr = sitk.GetArrayFromImage(resampled_mask)
        mu.log("ImageSetT2MSE[%s]::update_ROI_mask() : resampled mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.set_T2_roi_mask(resampled_mask)
        # resample and store any associated parameter maps
        resampled_pmap_dict = OrderedDict()
        for p, pmap in t2_pmap_dict.items():
            resampled_pmap_dict[p] = sitk.Resample(pmap,
                                                   self.image_list[0],
                                                   sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        self.set_T2_pmap_dict(resampled_pmap_dict)
        # resample and store any associated parameter maps
        super().resample_derived_maps()
    def write_roi_pdf_page(self, c, sup_title=None, mask_override_sitk_im=None, include_pmap_pages=False):
        self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                 all_roi_values=list(T2_ROI_LABEL_IDX_MAP.values()))
        t2_pmap_dict = self.get_T2_pmap_dict()
        if include_pmap_pages and (t2_pmap_dict is not None):
            for param_name, pmap in t2_pmap_dict.items():
                self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                         all_roi_values=list(T2_ROI_LABEL_IDX_MAP.values()),
                                         parameter_map_name=param_name,
                                         parameter_map=pmap)

class ImageSetDW(ImageSetAbstract):
    def __init__(self,
                 set_label,
                 sitk_im_list,
                 b_value_list,
                 repetition_time_list,
                 geometry_image,
                 series_instance_UIDs=None, series_numbers=None,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope_list=None,
                 rescale_intercept_list=None,
                 scale_slope_list=None,
                 scale_intercept_list=None,
                 scanner_make=None, scanner_model=None, scanner_serial_number=None, scanner_field_strength=None,
                 study_date=None, study_time=None,
                 image_filepaths_list=None):
        super().__init__(set_label=set_label,
                         sitk_im_list=sitk_im_list,
                         measurement_variable_list=b_value_list,
                         measurement_variable_name="b-value",
                         measurement_variable_units="s/µm²",
                         repetition_time_list=repetition_time_list,
                         geometry_image=geometry_image,
                         series_instance_UIDs=series_instance_UIDs, series_numbers=series_numbers,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         rescale_slope_list=rescale_slope_list, rescale_intercept_list=rescale_intercept_list,
                         scale_slope_list=scale_slope_list, scale_intercept_list=scale_intercept_list,
                         scanner_make=scanner_make, scanner_model=scanner_model, scanner_serial_number=scanner_serial_number,
                         scanner_field_strength=scanner_field_strength, date_acquired=study_date, time_acquired=study_time,
                         image_filepaths_list=image_filepaths_list)
    def get_b_values(self):
        return super().get_measurement_variables()
    def set_dw_roi_mask(self, roi_sitk_image):
        super().set_roi_image(roi_sitk_image)
    def set_dw_pmap_dict(self, roi_pmap_dict):
        super().set_roi_parameter_map_dict(roi_pmap_dict)
    def get_dw_pmap_dict(self):
        return self.roi_pmap_dict
    def update_ROI_mask(self):
        if self.geometry_image is None:
            mu.log("ImageSetDW::update_ROI_mask(): no geometry image set", LogLevels.LOG_WARNING)
            return None
        if not (len(self.image_list) > 0):
            mu.log("ImageSetDW::update_ROI_mask(): no images in set", LogLevels.LOG_WARNING)
            return None
        dw_mask_im, dw_pmap_dict = self.geometry_image.get_DW_mask_and_pmaps()
        mask_arr = sitk.GetArrayFromImage(dw_mask_im)
        mu.log("ImageSetDW[%s]::update_ROI_mask() : orig mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        resampled_mask = sitk.Resample(dw_mask_im,
                                       self.image_list[0],
                                       sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        mask_arr = sitk.GetArrayFromImage(resampled_mask)
        mu.log("ImageSetDW[%s]::update_ROI_mask() : resampled mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.set_dw_roi_mask(resampled_mask)
        # resample and store any associated ROI parameter maps (spherical/cylindrical coords.)
        resampled_pmap_dict = OrderedDict()
        for p, pmap in dw_pmap_dict.items():
            resampled_pmap_dict[p] = sitk.Resample(pmap,
                                                   self.image_list[0],
                                                   sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        self.set_dw_pmap_dict(resampled_pmap_dict)
        # resample and store any associated parameter maps
        super().resample_derived_maps()


    def write_roi_pdf_page(self, c, sup_title=None, mask_override_sitk_im=None, include_pmap_pages=False):
        self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                 all_roi_values=list(DW_ROI_LABEL_IDX_MAP.values()))
        dw_pmap_dict = self.get_dw_pmap_dict()
        if include_pmap_pages and (dw_pmap_dict is not None):
            for param_name, pmap in dw_pmap_dict.items():
                self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                         all_roi_values=list(DW_ROI_LABEL_IDX_MAP.values()),
                                         parameter_map_name=param_name,
                                         parameter_map=pmap)



class ImageSetT2Star(ImageSetAbstract):
    def __init__(self,
                 set_label,
                 sitk_im_list,
                 echo_time_list,
                 repetition_time_list,
                 geometry_image,
                 series_instance_UIDs=None, series_numbers=None,
                 bits_allocated=None, bits_stored=None,
                 rescale_slope_list=None,
                 rescale_intercept_list=None,
                 scale_slope_list=None,
                 scale_intercept_list=None,
                 scanner_make=None, scanner_model=None, scanner_serial_number=None, scanner_field_strength=None,
                 study_date=None, study_time=None,
                 image_filepaths_list=None):
        super().__init__(set_label=set_label,
                         sitk_im_list=sitk_im_list,
                         measurement_variable_list=echo_time_list,
                         measurement_variable_name="echo time",
                         measurement_variable_units="ms",
                         repetition_time_list=repetition_time_list,
                         geometry_image=geometry_image,
                         series_instance_UIDs=series_instance_UIDs, series_numbers=series_numbers,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         rescale_slope_list=rescale_slope_list, rescale_intercept_list=rescale_intercept_list,
                         scale_slope_list=scale_slope_list, scale_intercept_list=scale_intercept_list,
                         scanner_make=scanner_make, scanner_model=scanner_model, scanner_serial_number=scanner_serial_number,
                         scanner_field_strength=scanner_field_strength, date_acquired=study_date, time_acquired=study_time,
                         image_filepaths_list=image_filepaths_list)
    def get_echo_times(self):
        return super().get_measurement_variables()
    def set_T2_roi_mask(self, roi_sitk_image):
        super().set_roi_image(roi_sitk_image)
    def set_T2_pmap_dict(self, roi_pmap_dict):
        super().set_roi_parameter_map_dict(roi_pmap_dict)
    def get_T2_pmap_dict(self):
        return self.roi_pmap_dict
    def update_ROI_mask(self):
        if self.geometry_image is None:
            mu.log("ImageSetT2Star::update_ROI_mask(): no geometry image set", LogLevels.LOG_WARNING)
            return None
        if not (len(self.image_list) > 0):
            mu.log("ImageSetT2Star::update_ROI_mask(): no images in set", LogLevels.LOG_WARNING)
            return None
        t2_mask_im, t2_pmap_dict = self.geometry_image.get_T2_mask_and_pmaps()
        mask_arr = sitk.GetArrayFromImage(t2_mask_im)
        mu.log("ImageSetT2Star[%s]::update_ROI_mask() : orig mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        resampled_mask = sitk.Resample(t2_mask_im,
                                       self.image_list[0],
                                       sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        mask_arr = sitk.GetArrayFromImage(resampled_mask)
        mu.log("ImageSetT2Star[%s]::update_ROI_mask() : resampled mask values [%d, %d]" %
               (self.get_set_label(), np.min(mask_arr), np.max(mask_arr)), LogLevels.LOG_INFO)
        self.set_T2_roi_mask(resampled_mask)
        # resample and store any associated parameter maps
        resampled_pmap_dict = OrderedDict()
        for p, pmap in t2_pmap_dict.items():
            resampled_pmap_dict[p] = sitk.Resample(pmap,
                                                   self.image_list[0],
                                                   sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
        self.set_T2_pmap_dict(resampled_pmap_dict)
    def write_roi_pdf_page(self, c, sup_title=None, mask_override_sitk_im=None, include_pmap_pages=False):
        self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                  all_roi_values=list(T2_ROI_LABEL_IDX_MAP.values()))
        t2_pmap_dict = self.get_T2_pmap_dict()
        if include_pmap_pages and (t2_pmap_dict is not None):
            for param_name, pmap in t2_pmap_dict.items():
                self._write_roi_pdf_page(c, sup_title, mask_override_sitk_im,
                                         all_roi_values=list(T2_ROI_LABEL_IDX_MAP.values()),
                                         parameter_map_name=param_name,
                                         parameter_map=pmap)