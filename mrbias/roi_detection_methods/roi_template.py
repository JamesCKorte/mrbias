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
from enum import IntEnum
from collections import OrderedDict
from abc import ABC, abstractmethod

import yaml

import SimpleITK as sitk
import pandas as pd
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
from mrbias.misc_utils import LogLevels, OrientationOptions, PhantomOptions
from mrbias.misc_utils import ROI_IDX_LABEL_MAP, T1_ROI_LABEL_IDX_MAP, T2_ROI_LABEL_IDX_MAP, DW_ROI_LABEL_IDX_MAP


# for future expansion / different phantoms
class ROITypeOptions(IntEnum):
    SPHERE = 1
    CYLINDER = 2

class ROIImage(object):
    def __init__(self, im, roi_dict):
        self.image = im
        self.roi_dict = roi_dict

    def get_mask_image(self):
        """
        Create a template mask image of the same image grid as the template image
        Args:
            roi_dict (OrderedDict): with a roi_idx key and values of type ROI (supports ROISphere and ROICylinder)
        Returns:
            SimpleITK.Image: a template mask with 0 as background and ROIs with values as defined in points
        """
        # Create a numpy image array of zeros with the same size as the geometric image
        fixed_geo_arr = sitk.GetArrayFromImage(self.image)
        fixed_geo_spacing = np.array(self.image.GetSpacing())
        mask_arr = np.zeros_like(fixed_geo_arr, dtype=np.uint16)
        roi_dict = self.roi_dict
        # Interogate ROIs to see what property maps are available, then create empty maps to be filled
        prop_list = []
        for roi_dx, roi in roi_dict.items():
            for p in roi.properties:
                if not (p in prop_list):
                    prop_list.append(p)
        prop_map_dict = OrderedDict()
        for p in prop_list:
            prop_map_dict[p] = np.zeros_like(fixed_geo_arr, dtype=float)
        # Get each ROI to fill in its information in the mask array, and the property maps
        for roi_dx, roi in roi_dict.items():
            # rely on the concrete class to draw its own ROI on the mask image
            roi.draw(mask_arr, fixed_geo_spacing, prop_map_dict)
        # Convert the roi mask array to a simpleITK image with matched spatial properties to the
        # fixed geometric image
        masked_image = sitk.GetImageFromArray(mask_arr)
        masked_image.SetOrigin(self.image.GetOrigin())
        masked_image.SetDirection(self.image.GetDirection())
        masked_image.SetSpacing(self.image.GetSpacing())
        # Convert each of the property maps to a simpleITK image with matched spatial properties
        # to the fixed geometric image
        prop_image_dict = OrderedDict()
        for p, arr in prop_map_dict.items():
            prop_image = sitk.GetImageFromArray(arr)
            prop_image.SetOrigin(self.image.GetOrigin())
            prop_image.SetDirection(self.image.GetDirection())
            prop_image.SetSpacing(self.image.GetSpacing())
            prop_image_dict[p] = prop_image
        return masked_image, prop_image_dict

    def get_slice_dx(self, slice_orient=OrientationOptions.AXI):
        slice_vec = []
        for roi in self.roi_dict.values():
            if slice_orient == OrientationOptions.AXI:
                slice_vec.append(roi.get_slice_dx())
            elif slice_orient == OrientationOptions.COR:
                slice_vec.append(roi.get_cntr_cor_slice_dx())
            elif slice_orient == OrientationOptions.SAG:
                slice_vec.append(roi.get_cntr_sag_slice_dx())
        return int(np.median(np.array(slice_vec)))

    def get_centroid_dx(self):
        return [self.get_slice_dx(OrientationOptions.AXI),
                self.get_slice_dx(OrientationOptions.COR),
                self.get_slice_dx(OrientationOptions.SAG)]

    def get_roi_values(self):
        return list(self.roi_dict.keys())

    def get_roi_dict(self):
        return self.roi_dict

    def get_image(self):
        return self.image


class ROITemplate(object):
    def __init__(self, template_dir,
                 dcm_subdir="dicom",
                 t1_rois_file="default_T1_rois.yaml",
                 t2_rois_file="default_T2_rois.yaml",
                 dw_rois_file="default_DW_rois.yaml"):
        dcm_dir = os.path.join(template_dir, dcm_subdir)
        t1_yaml_file = os.path.join(template_dir, t1_rois_file)
        t2_yaml_file = os.path.join(template_dir, t2_rois_file)
        dw_yaml_file = os.path.join(template_dir, dw_rois_file)
        if not os.path.isdir(dcm_dir):
            mu.log("ROITemplate::__init__(): invalid dicom dir : %s" % dcm_dir, LogLevels.LOG_WARNING)
        if not os.path.isfile(t1_yaml_file):
            mu.log("ROITemplate::__init__(): invalid t1 roi yaml file : %s" % t1_yaml_file, LogLevels.LOG_WARNING)
        if not os.path.isfile(t2_yaml_file):
            mu.log("ROITemplate::__init__(): invalid t2 roi yaml file : %s" % t2_yaml_file, LogLevels.LOG_WARNING)
        if not os.path.isfile(dw_yaml_file):
            mu.log("ROITemplate::__init__(): invalid dw roi yaml file : %s" % dw_yaml_file, LogLevels.LOG_WARNING)
        # load the image file
        mu.log("ROITemplate::init(): loading template geometry image from DCM dir: %s" %
               dcm_dir, LogLevels.LOG_INFO)
        files_sorted, series_uid, rescale_slope, rescale_intercept, \
            scale_slope, scale_intercept, is_philips = mu.parse_dicom_dir_for_info(dcm_dir)
        r_val = mu.load_image_from_filelist(files_sorted, series_uid,
                                            rescale_slope, rescale_intercept,
                                            scale_slope, scale_intercept, philips_scaling=is_philips)
        self.image, rescale_slope, rescale_intercept, scale_slope, scale_intercept = r_val
        # parse the ROI yaml files and create roi image objects
        self.t1_roi_image = ROIImage(self.image, self.parse_t1_yaml(t1_yaml_file))
        self.t2_roi_image = ROIImage(self.image, self.parse_t2_yaml(t2_yaml_file))
        self.dw_roi_image = ROIImage(self.image, self.parse_dw_yaml(dw_yaml_file))

    def get_image(self):
        return self.image

    def parse_t1_yaml(self, yaml_file):
        return self.__parse_roi_yaml_file(yaml_file, T1_ROI_LABEL_IDX_MAP)

    def parse_t2_yaml(self, yaml_file):
        return self.__parse_roi_yaml_file(yaml_file, T2_ROI_LABEL_IDX_MAP)

    def parse_dw_yaml(self, yaml_file):
        return self.__parse_roi_yaml_file(yaml_file, DW_ROI_LABEL_IDX_MAP)

    def __parse_roi_yaml_file(self, yaml_file, roi_label_idx_map):
        roi_dict = OrderedDict()

        try:
            with open(yaml_file) as file:
                in_dict = yaml.full_load(file)
        except FileNotFoundError:
            mu.log("Error: The file '%s' does not exist." %
                   yaml_file, LogLevels.LOG_INFO)
            in_dict = OrderedDict()

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
                                       "specified in yaml file : %s" % (roi_label, field, yaml_file),
                                       LogLevels.LOG_WARNING)
                        roi_radius_mm = yaml_roi["roi_radius_mm"]
                        ctr_vox_coords = yaml_roi["ctr_vox_coords"]
                        assert isinstance(roi_radius_mm, float), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "roi_radius_mm expected datatype is float (not %s)" % \
                                                                 type(roi_radius_mm)
                        assert isinstance(ctr_vox_coords, list), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "ctr_vox_coords expected datatype is list (not %s)" % \
                                                                 type(ctr_vox_coords)
                        roi_dict[roi_dx] = ROISphere(roi_label, roi_dx, ctr_vox_coords, roi_radius_mm)
                    if roi_type == "cylinder":
                        # check all cylinder fields are available and create Cylindrical ROI
                        for field in ['roi_radius_mm', 'ctr_vox_coords', 'roi_height_mm']:
                            if not (field in yaml_roi.keys()):
                                mu.log("ROITemplate::__parse_roi_yaml_file(): skipping ROI(%s) no expected field '%s' "
                                       "specified in yaml file : %s" % (roi_label, field, yaml_file),
                                       LogLevels.LOG_WARNING)
                        roi_radius_mm = yaml_roi["roi_radius_mm"]
                        ctr_vox_coords = yaml_roi["ctr_vox_coords"]
                        roi_height_mm = yaml_roi["roi_height_mm"]
                        assert isinstance(roi_radius_mm, float), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "roi_radius_mm expected datatype is float (not %s)" % \
                                                                 type(roi_radius_mm)
                        assert isinstance(ctr_vox_coords, list), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "ctr_vox_coords expected datatype is list (not %s)" % \
                                                                 type(ctr_vox_coords)
                        assert isinstance(roi_height_mm, float), "ROITemplate::__parse_roi_yaml_file(): " \
                                                                 "roi_height_mm expected datatype is float (not %s)" % \
                                                                 type(roi_height_mm)
                        roi_dict[roi_dx] = ROICylinder(roi_label, roi_dx, ctr_vox_coords, roi_radius_mm, roi_height_mm)
                else:
                    mu.log("ROITemplate::__parse_roi_yaml_file(): skipping ROI(%s) no field 'roi_type' "
                           "specified in yaml file : %s" % (roi_label, yaml_file), LogLevels.LOG_WARNING)
            else:
                mu.log("ROITemplate::__parse_roi_yaml_file(): ROI(%s) not specified in yaml file : %s" %
                       (roi_label, yaml_file), LogLevels.LOG_INFO)
        # return the ROI dictionary
        return roi_dict

    def get_T1_mask_image(self):
        return self.t1_roi_image.get_mask_image()

    def get_T2_mask_image(self):
        return self.t2_roi_image.get_mask_image()

    def get_DW_mask_image(self):
        return self.dw_roi_image.get_mask_image()

    def get_t1_slice_dx(self, slice_orient=OrientationOptions.AXI):
        return self.t1_roi_image.get_slice_dx(slice_orient)

    def get_t2_slice_dx(self, slice_orient=OrientationOptions.AXI):
        return self.t2_roi_image.get_slice_dx(slice_orient)

    def get_dw_slice_dx(self, slice_orient=OrientationOptions.AXI):
        return self.dw_roi_image.get_slice_dx(slice_orient)

    def get_t1_roi_dict(self):
        return self.t1_roi_image.get_roi_dict()

    def get_t2_roi_dict(self):
        return self.t2_roi_image.get_roi_dict()

    def get_dw_roi_dict(self):
        return self.dw_roi_image.get_roi_dict()

    def get_t1_roi_values(self):
        return self.t1_roi_image.get_roi_values()

    def get_t2_roi_values(self):
        return self.t2_roi_image.get_roi_values()

    def get_dw_roi_values(self):
        return self.dw_roi_image.get_roi_values()

    def get_t1_centroid_dx(self):
        return self.t1_roi_image.get_centroid_dx()

    def get_t2_centroid_dx(self):
        return self.t2_roi_image.get_centroid_dx()

    def get_dw_centroid_dx(self):
        return self.dw_roi_image.get_centroid_dx()




class ROI(ABC):
    def __init__(self, label, roi_index):
        self.label = label
        self.roi_index = roi_index
        self.properties = []
        assert self.roi_index in ROI_IDX_LABEL_MAP.keys(), "ROI::__init__: roi index is invalid, idx=%d" % self.roi_index

    """
    Draw the ROI into the passed mask array, marking the ROI with the global ROI index
    - also generate template coordinates to associate with each point in space, to
      allow the analysis of variation across the ROI (i.e. from the centre to the perimeter)

    The function adds data to the passed lbl array
    - additionally, there is a returned dictionary of additional properties of the ROI
      this has [key] -> [item] = [property_name] -> [value_at_location_array]
    """
    def get_centroid_dx(self):
        return np.array([self.get_cntr_sag_slice_dx(),
                         self.get_cntr_cor_slice_dx(),
                         self.get_slice_dx()])

    @abstractmethod
    def draw(self, lbl_arr, spacing, prop_map_dict=None):
        return None

    @abstractmethod
    def get_slice_dx(self):
        return None

    @abstractmethod
    def get_cntr_cor_slice_dx(self):
        return None

    @abstractmethod
    def get_cntr_sag_slice_dx(self):
        return None


class ROISphere(ROI):
    def __init__(self, label, roi_index,
                 ctr_vox_coords, radius_mm):
        super().__init__(label, roi_index)
        self.ctr_vox_coords = ctr_vox_coords
        self.radius_mm = radius_mm
        self.properties = ['radial_dist_mm', 'height_dist_mm']
        mu.log("\t\tROISphere::__init__(): sphere (%d : %s) created!" % (roi_index, label), LogLevels.LOG_INFO)

    def draw(self, lbl_arr, spacing, prop_map_dict=None):
        # calculate how many voxels to achieve the radius
        radius_vox = self.radius_mm / spacing
        z, y, x = np.ogrid[:lbl_arr.shape[0], :lbl_arr.shape[1], :lbl_arr.shape[2]]
        # Assigns the masked pixels in the copy image array to corresponding pixel values
        distance_from_centre = np.sqrt(((x - self.ctr_vox_coords[0]) / radius_vox[0]) ** 2 +
                                       ((y - self.ctr_vox_coords[1]) / radius_vox[1]) ** 2 +
                                       ((z - self.ctr_vox_coords[2]) / radius_vox[2]) ** 2)
        sphere_mask = distance_from_centre <= 1.0
        # assign the label to the correct spatial location
        lbl_arr[sphere_mask] = self.roi_index

        if prop_map_dict is not None:
            # record the radial distance (in mm) relative to the centre of the ROI
            if 'radial_dist_mm' in prop_map_dict.keys():
                distance_from_centre_mm = np.sqrt(((x - self.ctr_vox_coords[0]) * spacing[0]) ** 2 +
                                                  ((y - self.ctr_vox_coords[1]) * spacing[1]) ** 2 +
                                                  ((z - self.ctr_vox_coords[2]) * spacing[2]) ** 2)
                prop_map_dict['radial_dist_mm'][sphere_mask] = distance_from_centre_mm[sphere_mask]
            # record the vertical position (in mm) relative to the centre of the ROI
            if 'height_dist_mm' in prop_map_dict.keys():
                vertical_position_mm = (z - self.ctr_vox_coords[2]) * spacing[2] + (0.0 * x) + (0.0 * y)
                prop_map_dict['height_dist_mm'][sphere_mask] = vertical_position_mm[sphere_mask]

    def get_slice_dx(self):
        return self.ctr_vox_coords[2]

    def get_cntr_cor_slice_dx(self):
        return self.ctr_vox_coords[1]

    def get_cntr_sag_slice_dx(self):
        return self.ctr_vox_coords[0]


class ROICylinder(ROI):
    def __init__(self, label, roi_index,
                 ctr_vox_coords, radius_mm, height_mm):
        super().__init__(label, roi_index)
        self.ctr_vox_coords = ctr_vox_coords
        self.radius_mm = radius_mm
        self.height_mm = height_mm
        self.properties = ['radial_dist_mm', 'height_dist_mm']
        mu.log("\t\tROICylinder::__init__(): cylinder (%d : %s) created!" % (roi_index, label), LogLevels.LOG_INFO)

    def draw(self, lbl_arr, spacing, prop_map_dict=None):
        # calculate how many voxels to achieve the radius
        radius_vox = self.radius_mm / spacing
        height_vox = self.height_mm / spacing[2]
        z, y, x = np.ogrid[:lbl_arr.shape[0], :lbl_arr.shape[1], :lbl_arr.shape[2]]
        # Assigns the masked pixels in the copy image array to corresponding pixel values
        distance_from_centre = np.sqrt(((x - self.ctr_vox_coords[0]) / radius_vox[0]) ** 2 +
                                       ((y - self.ctr_vox_coords[1]) / radius_vox[1]) ** 2)
        vertical_height_abs = np.abs((z - self.ctr_vox_coords[2]) / (height_vox / 2.))
        cylinder_mask = (distance_from_centre <= 1.0) & (vertical_height_abs <= 1.0)
        # assign the label to the correct spatial location
        lbl_arr[cylinder_mask] = self.roi_index

        if prop_map_dict is not None:
            # record the radial distance (in mm) relative to the centre of the ROI
            if 'radial_dist_mm' in prop_map_dict.keys():
                distance_from_centre_mm = np.sqrt(((x - self.ctr_vox_coords[0]) * spacing[0]) ** 2 +
                                                  ((y - self.ctr_vox_coords[1]) * spacing[1]) ** 2) + (0.0 * z)
                prop_map_dict['radial_dist_mm'][cylinder_mask] = distance_from_centre_mm[cylinder_mask]
            # record the vertical position (in mm) relative to the centre of the ROI
            if 'height_dist_mm' in prop_map_dict.keys():
                vertical_position_mm = (z - self.ctr_vox_coords[2]) * spacing[2] + (0.0 * x) + (0.0 * y)
                prop_map_dict['height_dist_mm'][cylinder_mask] = vertical_position_mm[cylinder_mask]

    def get_slice_dx(self):
        return self.ctr_vox_coords[2]

    def get_cntr_cor_slice_dx(self):
        return self.ctr_vox_coords[1]

    def get_cntr_sag_slice_dx(self):
        return self.ctr_vox_coords[0]



if __name__ == "__main__":
    main()
