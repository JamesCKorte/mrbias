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
import scipy.stats as spstat
import lmfit
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


class NormalisationOptions(IntEnum):
    VOXEL_MAX = 1
    ROI_MAX = 2
NORM_OPT_STR_MAP = {NormalisationOptions.VOXEL_MAX: "NrmVoxMax",
                    NormalisationOptions.ROI_MAX: "NrmROIMax"}
NORM_SETTING_STR_ENUM_MAP = {"voxel_max": NormalisationOptions.VOXEL_MAX,
                             "roi_max": NormalisationOptions.ROI_MAX}

class AveragingOptions(IntEnum):
    AVERAGE_ROI = 1
    AVERAGE_PER_SLICE = 2
AV_OPT_STR_MAP = {AveragingOptions.AVERAGE_ROI: "AvROI",
                  AveragingOptions.AVERAGE_PER_SLICE: "AvSlice"}
AV_SETTING_STR_ENUM_MAP = {"voxels_in_ROI": AveragingOptions.AVERAGE_ROI,
                           "voxels_in_slice": AveragingOptions.AVERAGE_PER_SLICE}

class ExclusionOptions(IntEnum):
    CLIPPED_VALUES = 1
EXCL_OPT_STR_MAP = {ExclusionOptions.CLIPPED_VALUES: "ExclClip"}
EXCL_SETTING_STR_ENUM_MAP = {"clipped": ExclusionOptions.CLIPPED_VALUES}


class OptiOptions(IntEnum):
    SCIPY = 1
    LMFIT = 2
    LINREG = 3
OPTI_OPT_STR_MAP = {OptiOptions.SCIPY: "scipy",
                    OptiOptions.LMFIT: "lmfit",
                    OptiOptions.LINREG: "linreg"}
OPTI_SETTING_STR_ENUM_MAP = {"scipy": OptiOptions.SCIPY,
                             "lmfit": OptiOptions.LMFIT,
                             "linreg": OptiOptions.LINREG}

GOODNESS_OF_FIT_PARAMS = ['chisqr', 'redchi', 'aic', 'bic'];
GOODNESS_OF_FIT_DESC_DICT = {'chisqr': "Chi-square statistic",
                             'redchi': "Reduced Chi-square statistic",
                             'aic':    "Akaike Information Criterion statistic",
                             'bic':    "Bayesian Information Criterion statistic"}


class CurveFitROI(imset.ImageSetROI):
    def __init__(self, label,
                 voxel_data_array, voxel_data_xyz, measurement_variable_vector,
                 measurement_variable_name, measurement_variable_units,
                 reference_value, initialisation_value, exclusion_list=None, exclusion_label=None,
                 voxel_pmap_dict=None, derived_map_dict=None,
                 rescale_slope_list=None, rescale_intercept_list=None,
                 scale_slope_list=None, scale_intercept_list=None,
                 bits_allocated=None, bits_stored=None,
                 scanner_make=None,
                 color_settings=None):
        super().__init__(label, voxel_data_array, voxel_data_xyz, measurement_variable_vector,
                         measurement_variable_name, measurement_variable_units,
                         voxel_pmap_dict=voxel_pmap_dict, derived_map_dict=derived_map_dict,
                         rescale_slope_list=rescale_slope_list, rescale_intercept_list=rescale_intercept_list,
                         scale_slope_list=scale_slope_list, scale_intercept_list=scale_intercept_list,
                         bits_allocated=bits_allocated, bits_stored=bits_stored,
                         scanner_make=scanner_make)
        mu.log("\t\tCurveFitROI::__init__(): ref value [%0.2f] and init value [%0.2f]" %
               (reference_value, initialisation_value), LogLevels.LOG_INFO)

        self.reference_value = reference_value
        self.initialisation_value = initialisation_value
        # user exclusion list
        self.exclusion_list = exclusion_list
        self.exclusion_label = exclusion_label
        # a dictionary for every curve fit parameter
        # same shape as voxel data array (unless the curve fit is to an average ROI signal)
        self.voxel_fit_param_array_dict = {}
        self.voxel_fit_err_param_array_dict = {}
        self.voxel_fit_goodness_array_dict = {}
        # exclusion/inclusion list (include all as default)
        n_vox, n_meas = self.voxel_data_array.shape
        self.include_vector = np.ones((n_meas), dtype=int)
        self.exclude_reason_vector = [[] for x in range(n_meas)]
        # a measure of standard deviation and the central voxel indexes if averaging is applied to the ROI
        self.voxel_data_stddev = None
        self.av_voxel_data_xyz = copy.deepcopy(self.voxel_data_xyz)
        # flags
        self.is_clipped = False
        self.is_averaged = False
        self.is_slice_averaged = False
        self.is_normalised = False
        # colour settings
        self.colour = mu.ColourSettings()
        if color_settings is not None:
            self.colour = color_settings

    def exclude_clipped_values(self, percent_clipped_voxels):
        if self.is_averaged or self.is_slice_averaged or self.is_normalised:
            mu.log("\t\tCurveFitROI::exclude_clipped_values(): skipping exclusion as averaging or "
                   "normalisation has been applied [exclusion should be performed first on"
                   "raw voxel values]", LogLevels.LOG_WARNING)
            return None
        # search for clipped values
        n_vox, n_meas = self.voxel_data_array.shape
        parital_clipping = False
        clip_val_list = []
        for meas_dx in range(n_meas):
            # determine clip value for each measurement, based on:
            # - number of bits allocated in raw dicom unsigned integer data
            # - the mapping from raw integer to floating point values (image scaling)
            raw_clip_value = float(2 ** self.bits_stored) - 6.0  # technically should be minus 1, but like to add a bit of headroom
            clip_value = self.rescale_slope_list[meas_dx]*raw_clip_value + self.rescale_intercept_list[meas_dx]
            if self.scanner_make == "Philips":
                clip_value = clip_value / (self.rescale_slope_list[meas_dx] * self.scale_slope_list[meas_dx])
            clip_val_list.append(clip_value)

            # only check for clipping in included measurements (those not excluded by a prior exclusion)
            if self.include_vector[meas_dx]:

                if np.any(np.greater(self.voxel_data_array[:, meas_dx], clip_value)):
                    self.is_clipped = True
                    # if entire ROI at measurement x is clipped then ignore
                    if np.sum(np.greater(self.voxel_data_array[:, meas_dx], clip_value)) == n_vox:
                        self.include_vector[meas_dx] = 0
                        self.exclude_reason_vector[meas_dx].append('clipped')
                        mu.log("\t\tCurveFitROI::exclude_clipped_values(%d): full clipping found in %s "
                               "(ref val=%0.2f) [measurement %d / %d]" %
                               (clip_value, self.label, self.reference_value, meas_dx + 1, n_meas), LogLevels.LOG_INFO)
                    # otherwise mark it as being partially clipped
                    else:
                        parital_clipping = True
                        mu.log("\t\tCurveFitROI::exclude_clipped_values(%d): partial clipping found in %s "
                               "(ref val=%0.2f) [measurement %d / %d]" %
                               (clip_value, self.label, self.reference_value, meas_dx + 1, n_meas), LogLevels.LOG_INFO)
        # convert clip value list to np array for comparison
        clip_val_arr = np.array(clip_val_list)

        # if there is partial clipping (only some voxels clipped in a measurement) then try and reduce the voxels down to those
        # without clipping in any of measurements (making sure to exclude the completely clipped measurements)
        # if it reduces the number of voxels in the ROI too much then exclude the partially clipped measurements
        if parital_clipping:
            n_clean = 0
            for vox_dx in range(n_vox):
                if not np.any(np.greater(self.voxel_data_array[vox_dx, self.include_vector.astype(bool)],
                                         clip_val_arr[self.include_vector.astype(bool)])):
                    n_clean = n_clean + 1
            # if there are enough unclipped voxels then prune down to the clean voxels
            if n_clean/n_vox*100. > percent_clipped_voxels:
                mu.log("\t\tCurveFitROI::exclude_clipped_values: removing clipped voxels in %s "
                       "(ref val=%0.2f)  left %0.1f pcnt of original voxels (%d/%d) [threshold is >= %0.2f pcnt]" %
                       (self.label, self.reference_value,
                        n_clean/n_vox*100., n_clean, n_vox, percent_clipped_voxels), LogLevels.LOG_INFO)
                # create new voxel_data and voxel_xyz arrays
                voxel_data_array = np.zeros([n_clean, n_meas])
                voxel_data_xyz = (np.zeros([n_clean]), np.zeros([n_clean]), np.zeros([n_clean]))
                # copy the valid stuff (the unclipped voxel measurements)
                clean_dx = 0
                for vox_dx in range(n_vox):
                    if not np.any(np.greater(self.voxel_data_array[vox_dx, self.include_vector.astype(bool)],
                                             clip_val_arr[self.include_vector.astype(bool)])):
                        voxel_data_array[clean_dx] = self.voxel_data_array[vox_dx, :]
                        voxel_data_xyz[0][clean_dx] = self.voxel_data_xyz[0][vox_dx]
                        voxel_data_xyz[1][clean_dx] = self.voxel_data_xyz[1][vox_dx]
                        voxel_data_xyz[2][clean_dx] = self.voxel_data_xyz[2][vox_dx]
                        clean_dx = clean_dx + 1
                # assign the clean data to the class data
                self.voxel_data_array = voxel_data_array
                self.voxel_data_xyz = voxel_data_xyz
            # if pruning would reduce too many voxels (below threshold) then just exclude the measurements which are clipped
            else:
                mu.log("\t\tCurveFitROI::exclude_clipped_values: excluding partially clipped measurements as removing clipped voxels in %s "
                       "(ref val=%0.2f) would leave only %0.1f pcnt of original voxels (%d/%d) [threshold is >= %0.2f pcnt]" %
                       (self.label, self.reference_value,
                        n_clean/n_vox*100., n_clean, n_vox, percent_clipped_voxels), LogLevels.LOG_INFO)
                for meas_dx, clip_value in enumerate(clip_val_list):
                    if np.any(np.greater(self.voxel_data_array[:, meas_dx], clip_value)):
                        self.include_vector[meas_dx] = 0
                        self.exclude_reason_vector[meas_dx].append('clipped')

    def exclude_user_values(self):
        if self.exclusion_list is not None:
            # search for measurement values that match the user exclusion list
            n_vox, n_meas = self.voxel_data_array.shape
            for meas_dx, meas_val in enumerate(self.meas_var_vector):
                if np.any(np.isclose(meas_val, self.exclusion_list)):
                    mu.log("\t\tCurveFitROI::exclude_user_values(): in %s excluding value %s [measurement %d / %d]" %
                           (self.label, str(meas_val), meas_dx+1, n_meas), LogLevels.LOG_INFO)
                    self.include_vector[meas_dx] = 0
                    self.exclude_reason_vector[meas_dx].append(self.exclusion_label)

    def cast_to_float(self):
        self.voxel_data_array = self.voxel_data_array.astype(dtype=np.float64)

    def average_over_roi(self):
        n_vox, n_meas = self.voxel_data_array.shape
        # average voxel signal data
        self.voxel_data_stddev = np.reshape(np.nanstd(self.voxel_data_array, axis=0), (1, n_meas))
        self.voxel_data_array = np.reshape(np.nanmean(self.voxel_data_array, axis=0), (1, n_meas))
        # average parameter maps
        if self.voxel_pmap_dict is not None:
            voxel_pmap_dict = OrderedDict()
            for pmap_name, pmap_arr in self.voxel_pmap_dict.items():
                voxel_pmap_dict[pmap_name] = np.array([np.nanmean(pmap_arr)])
            self.voxel_pmap_dict = voxel_pmap_dict
        # average derived maps (i.e. scanner ADC map)
        if self.derived_map_dict is not None:
            derived_map_dict = OrderedDict()
            for pmap_label, (pmap_arr, pmap_name, pmap_unit) in self.derived_map_dict.items():
                pmap_arr_av = np.array([np.nanmean(pmap_arr)])
                derived_map_dict[pmap_label] = (pmap_arr_av, pmap_name, pmap_unit)
            self.derived_map_dict = derived_map_dict

        # TODO: review code and check if this 2D/3D check is still required (remove it if not)
        if len(self.voxel_data_xyz) == 2: #2D
            self.av_voxel_data_xyz = (np.array([int(np.round(np.nanmean(self.voxel_data_xyz[0])))]),
                                        np.array([int(np.round(np.nanmean(self.voxel_data_xyz[1])))]))
        elif len(self.voxel_data_xyz) == 3: #3D
            self.av_voxel_data_xyz = (np.array([int(np.round(np.nanmean(self.voxel_data_xyz[0])))]),
                                      np.array([int(np.round(np.nanmean(self.voxel_data_xyz[1])))]),
                                      np.array([int(np.round(np.nanmean(self.voxel_data_xyz[2])))]))
        self.is_averaged = True

    def average_over_each_slice(self):
        n_vox, n_meas = self.voxel_data_array.shape
        # create a coordinate and measurement arrays per slice (for signal, pmaps, dmaps)
        slice_dx_list = np.unique(self.voxel_data_xyz[0])
        vox_dict = OrderedDict()
        pmap_dict = OrderedDict()
        derived_dict = OrderedDict()
        for slice_num in slice_dx_list:
            vox_dict[slice_num] = [[],[]] # xyz data, signal data
            if self.voxel_pmap_dict is not None:
                pmap_dict[slice_num] = OrderedDict()
                for pmap_name, pmap_arr in self.voxel_pmap_dict.items():
                    pmap_dict[slice_num][pmap_name] = []
            if self.derived_map_dict is not None:
                derived_dict[slice_num] = OrderedDict()
                for dmap_label, (dmap_arr, dmap_name, dmap_unit) in self.derived_map_dict.items():
                    derived_dict[slice_num][dmap_label] = ([], dmap_name, dmap_unit)
        # link voxel data to a slice
        for v_dx in range(n_vox):
            slice_num = self.voxel_data_xyz[0][v_dx]
            # add the signal data for each slice
            vox_dict[slice_num][0].append([self.voxel_data_xyz[0][v_dx],
                                          self.voxel_data_xyz[1][v_dx],
                                          self.voxel_data_xyz[2][v_dx]])
            vox_dict[slice_num][1].append(self.voxel_data_array[v_dx])
            # add the parameter map data (i.e. cylindrical coords) for each slice
            if self.voxel_pmap_dict is not None:
                for pmap_name, pmap_arr in self.voxel_pmap_dict.items():
                    pmap_dict[slice_num][pmap_name].append(pmap_arr[v_dx])
            # add the derived map data (i.e. ADC map) for each slice
            if self.derived_map_dict is not None:
                for dmap_label, (dmap_arr, dmap_name, dmap_unit) in self.derived_map_dict.items():
                    derived_dict[slice_num][dmap_label][0].append(dmap_arr[v_dx])

        # average over each slice
        n_slices = len(slice_dx_list)
        vox_x, vox_y, vox_z = [], [], []
        vox_data_av_array = np.zeros([n_slices, n_meas])
        vox_data_stddev_array = np.zeros([n_slices, n_meas])
        vox_data_xyz_array = np.zeros([3, n_slices])
        if self.voxel_pmap_dict is not None:
            pmap_dict_av = OrderedDict()
            for pmap_name, pmap_arr in self.voxel_pmap_dict.items():
                pmap_dict_av[pmap_name] = np.zeros([n_slices])
        if self.derived_map_dict is not None:
            dmap_dict_av = OrderedDict()
            for dmap_label, (dmap_arr, dmap_name, dmap_unit) in self.derived_map_dict.items():
                dmap_dict_av[dmap_label] = (np.zeros([n_slices]), dmap_name, dmap_unit)
        for slice_dx, slice_num in enumerate(slice_dx_list):
            # average the slice coordinates to get a spatial centroid
            slice_xyz = np.array(vox_dict[slice_num][0])
            print("slice list:", slice_dx_list)
            print("slice xyz shape:", slice_xyz.shape)
            vox_data_xyz_array[0, slice_dx] = int(np.round(np.nanmean(slice_xyz[:, 0])))
            vox_data_xyz_array[1, slice_dx] = int(np.round(np.nanmean(slice_xyz[:, 1])))
            vox_data_xyz_array[2, slice_dx] = int(np.round(np.nanmean(slice_xyz[:, 2])))
            # average the signal data on a given slice
            slice_data = vox_dict[slice_num][1]
            vox_data_av_array[slice_dx, :] = np.reshape(np.nanmean(slice_data, axis=0), (1, n_meas))
            vox_data_stddev_array[slice_dx, :] = np.reshape(np.nanstd(slice_data, axis=0), (1, n_meas))
            # average the parameter maps on a given slice
            if self.voxel_pmap_dict is not None:
                for pmap_name, pmap_arr in self.voxel_pmap_dict.items():
                    pmap_dict_av[pmap_name][slice_dx] = np.array([np.nanmean(pmap_dict[slice_num][pmap_name])])
            # average the derived maps on a given slice
            if self.derived_map_dict is not None:
                for dmap_label, (dmap_arr, dmap_name, dmap_unit) in self.derived_map_dict.items():
                    dmap_dict_av[dmap_label][0][slice_dx] = np.array([np.nanmean(derived_dict[slice_num][dmap_label][0])])

        # assign to class variables
        self.voxel_data_array = vox_data_av_array
        self.voxel_data_stddev = vox_data_av_array
        self.av_voxel_data_xyz = (vox_data_xyz_array[0, :],
                                  vox_data_xyz_array[1, :],
                                  vox_data_xyz_array[2, :])
        if self.voxel_pmap_dict is not None:
            self.voxel_pmap_dict = pmap_dict_av
        if self.derived_map_dict is not None:
            self.derived_map_dict = dmap_dict_av
        # flag slice averaging complete
        self.is_slice_averaged = True

    def normalise_to_max_in_roi(self):
        # check there included measurements
        # norm_factor = 1.0
        # if np.any(self.include_vector.astype(bool)):
        #     norm_factor = np.nanmax(self.voxel_data_array[:, self.include_vector.astype(bool)].flatten())
        # else:
        #     norm_factor = np.nanmax(self.voxel_data_array.flatten())
        #     mu.log(
        #         "\t\tCurveFitROI::normalise_to_max_in_roi (ROI:%s): all measurements excluded so normalising to max value in ROI" %
        #         (self.label), LogLevels.LOG_INFO)
        norm_factor = np.nanmax(self.voxel_data_array.flatten())
        self.voxel_data_array = self.voxel_data_array / norm_factor
        if self.is_averaged or self.is_slice_averaged:
            self.voxel_data_stddev = self.voxel_data_stddev/norm_factor
        self.is_normalised = True

    def normalise_to_max_per_voxel(self):
        n_vox, n_meas = self.voxel_data_array.shape
        # check there included measurements
        norm_factor = 1.0
        # if np.any(self.include_vector.astype(bool)):
        #     for vox_dx in range(n_vox):
        #         vox_series = self.voxel_data_array[vox_dx, :]
        #         norm_factor = np.nanmax(vox_series[self.include_vector.astype(bool)])
        #         self.voxel_data_array[vox_dx, :] = vox_series/norm_factor
        # else:
        #     norm_factor = np.nanmax(self.voxel_data_array.flatten())
        #     mu.log(
        #         "\t\tCurveFitROI::normalise_to_max_per_voxel (ROI:%s): all measurements excluded so normalising to max value in ROI" %
        #         (self.label), LogLevels.LOG_INFO)
        #     self.voxel_data_array = self.voxel_data_array/norm_factor
        for vox_dx in range(n_vox):
            vox_series = self.voxel_data_array[vox_dx, :]
            norm_factor = np.nanmax(vox_series)
            self.voxel_data_array[vox_dx, :] = vox_series/norm_factor
        # corner case
        if self.is_averaged or self.is_slice_averaged:
            self.voxel_data_stddev = self.voxel_data_stddev / norm_factor
        self.is_normalised = True


    def estimate_cf_start_point(self, cf_model):
        n_vox, n_meas = self.voxel_data_array.shape
        av_sig = np.reshape(np.nanmean(self.voxel_data_array, axis=0), (1, n_meas))
        meas_vec = np.array(self.meas_var_vector)
        av_sig = av_sig.flatten()
        self.initialisation_value = cf_model.estimate_cf_start_point(meas_vec[self.include_vector.astype(bool)],
                                                                     av_sig[self.include_vector.astype(bool)],
                                                                     self.initialisation_value,
                                                                     self)
        
    def get_number_fit_voxels(self):
        return self.voxel_data_array.shape[0]

    def get_voxel_series(self, voxel_dx):
        assert voxel_dx < self.get_number_fit_voxels(), "CurveFitROI::get_voxel_series() voxel_dx (%d) exceeds " \
                                                        "the size of the voxel data array [%d, %d]" % \
                                                        (voxel_dx,
                                                         self.voxel_data_array.shape[0],
                                                         self.voxel_data_array.shape[1])
        return self.voxel_data_array[voxel_dx, self.include_vector.astype(bool)]
    def get_voxel_std_series(self):
        return self.voxel_data_stddev[0, self.include_vector.astype(bool)]
    def get_measurement_series(self):
        return np.array(self.meas_var_vector)[self.include_vector.astype(bool)]

    def get_excluded_voxel_series(self, voxel_dx):
        return self.voxel_data_array[voxel_dx, np.logical_not(self.include_vector.astype(bool))]
    def get_excluded_measurement_series(self):
        return np.array(self.meas_var_vector)[np.logical_not(self.include_vector.astype(bool))]
    def get_excluded_reason_series(self):
        r_list = []
        for list_dx, excluded in enumerate(np.logical_not(self.include_vector.astype(bool))):
            if excluded:
                r_list.append(self.exclude_reason_vector[list_dx])
        return r_list

    def set_voxel_fit_data(self, vox_data_dict):
        assert isinstance(vox_data_dict, dict), "CurveFitROI::set_voxel_fit_data() expects datatype 'dict' " \
                                                "(not %s)" % (type(vox_data_dict))
        self.voxel_fit_param_array_dict = vox_data_dict

    def set_voxel_fit_err_data(self, vox_err_data_dict):
        assert isinstance(vox_err_data_dict, dict), "CurveFitROI::set_voxel_fit_err_data() expects datatype 'dict' " \
                                                    "(not %s)" % (type(vox_err_data_dict))
        self.voxel_fit_err_param_array_dict = vox_err_data_dict

    def set_voxel_fit_goodness_data(self, vox_gfit_data_dict):
        assert isinstance(vox_gfit_data_dict, dict), "CurveFitROI::set_voxel_fit_goodness_data() expects datatype 'dict' " \
                                                     "(not %s)" % (type(vox_gfit_data_dict))
        self.voxel_fit_goodness_array_dict = vox_gfit_data_dict

    def has_been_fit(self):
        has_been_fit = True
        for param_name, param_fit_vec in self.voxel_fit_param_array_dict.items():
            if len(param_fit_vec) < 1:
                has_been_fit = False
        return has_been_fit


    def get_voxel_dataframe_column_names(self, cf_model):
        meas_str = "%s (%s)" % (self.meas_var_name, self.meas_var_units)
        col_names = ["RoiIndex", "RoiLabel", "VoxelID", "VoxelX", "VoxelY", "VoxelZ", meas_str, "VoxelSignal"]
        for param_name in cf_model.get_ordered_parameter_symbols():
            col_names.append(param_name)
        col_names.append("%s_reference" % cf_model.get_symbol_of_interest())
        col_names.append("%s_initialise" % cf_model.get_symbol_of_interest())
        col_names.append("Included")
        col_names.append("Averaged")
        col_names.append("Normalised")
        col_names.append("Preprocessing")
        col_names.append("ModelName")
        col_names.append("ImageSetLabel")
        # add columns for ROI parameters (i.e. cylindrical coordinates from template)
        if self.voxel_pmap_dict is not None:
            for param_name in self.voxel_pmap_dict.keys():
                col_names.append(param_name)
        # add columns for derived maps (i.e. ADC maps)
        if self.derived_map_dict is not None:
            for pmap_label in self.derived_map_dict.keys():
                col_names.append("DerivedMap_%s_type" % pmap_label)
                col_names.append("DerivedMap_%s_value" % pmap_label)
                col_names.append("DerivedMap_%s_unit" % pmap_label)
        return col_names

    def add_voxel_data_to_df(self, df, cf_model):
        data_list = []
        include_meas_vals = self.get_measurement_series()
        exclude_meas_vals = self.get_excluded_measurement_series()
        voxel_xyz = self.voxel_data_xyz
        if self.is_averaged or self.is_slice_averaged:
            voxel_xyz = self.av_voxel_data_xyz
        for vox_dx in range(self.get_number_fit_voxels()):
            # pull out the parameter maps for the current voxel
            vox_pmap_dict = OrderedDict()
            for param_name, param_arr in self.voxel_pmap_dict.items():
                vox_pmap_dict[param_name] = param_arr[vox_dx]
            # pull out the derived map details for the current voxel
            vox_derived_map_dict = OrderedDict()
            if self.derived_map_dict is not None:
                for pmap_label, (pmap_arr, pmap_name, pmap_unit) in self.derived_map_dict.items():
                    vox_derived_map_dict[pmap_label] = (pmap_arr[vox_dx], pmap_name, pmap_unit)
            # loop over the measurement series
            for meas_vals, vox_series, is_included in zip([include_meas_vals, exclude_meas_vals],
                                                          [self.get_voxel_series(vox_dx), self.get_excluded_voxel_series(vox_dx)],
                                                          [True, False]):
                for meas_val, vox_val in zip(meas_vals, vox_series):
                    vox_z, vox_y, vox_x = None, None, None
                    assert len(voxel_xyz) >= 2, "CurveFitROI::add_voxel_data_to_df(): unexpected data shape not 2D or 3D"
                    if len(voxel_xyz) == 2:
                        vox_y, vox_x = voxel_xyz[0][vox_dx], \
                                       voxel_xyz[1][vox_dx]
                    if len(voxel_xyz) == 3:
                        vox_z, vox_y, vox_x = voxel_xyz[0][vox_dx], \
                                              voxel_xyz[1][vox_dx], \
                                              voxel_xyz[2][vox_dx]
                    vox_meas_list = [mu.ROI_LABEL_IDX_MAP[self.label], self.label, vox_dx, vox_x, vox_y, vox_z, meas_val, vox_val]
                    for param_name in cf_model.get_ordered_parameter_symbols():
                        if self.has_been_fit():
                            vox_meas_list.append(self.voxel_fit_param_array_dict[param_name][vox_dx])
                        else:
                            vox_meas_list.append(np.nan)
                    vox_meas_list.append(self.reference_value)
                    vox_meas_list.append(self.initialisation_value)
                    vox_meas_list.append(is_included)
                    vox_meas_list.append(self.is_averaged or self.is_slice_averaged)
                    vox_meas_list.append(self.is_normalised)
                    vox_meas_list.append(cf_model.get_preproc_name())
                    vox_meas_list.append(cf_model.get_model_name())
                    vox_meas_list.append(cf_model.get_imageset_name())
                    # append on the parameter map variables for this voxel
                    for param_name, param_val in vox_pmap_dict.items():
                        vox_meas_list.append(param_val)
                    # append on the derived map data
                    for pmap_label, (pmap_val, pmap_name, pmap_unit) in vox_derived_map_dict.items():
                        vox_meas_list.append(pmap_name)
                        vox_meas_list.append(pmap_val)
                        vox_meas_list.append(pmap_unit)
                    # append the row to the datalist
                    data_list.append(vox_meas_list)
        # create the ROI dataframe and append it to the passed dataframe
        df_roi = pd.DataFrame(data_list, columns=self.get_voxel_dataframe_column_names(cf_model))
        df_roi[['Included', 'Averaged', 'Normalised']] = df_roi[['Included', 'Averaged', 'Normalised']].astype(bool)
        df[['Included', 'Averaged', 'Normalised']] = df[['Included', 'Averaged', 'Normalised']].astype(bool)
        return pd.concat([df, df_roi], ignore_index=True)



    def get_fit_summary_dataframe_column_names(self, cf_model):
        col_names = ["RoiIndex", "RoiLabel"]
        symbol_of_interest = cf_model.get_symbol_of_interest()
        for param_name in cf_model.get_ordered_parameter_symbols():
            col_names.append("%s (mean)" % param_name)
            col_names.append("%s (std.dev.)" % param_name)
            if param_name == symbol_of_interest:
                col_names.append("%s (mean error)" % param_name)
                col_names.append("%s (mean percent error)" % param_name)
        col_names.append("%s_reference" % cf_model.get_symbol_of_interest())
        col_names.append("%s_initialise" % cf_model.get_symbol_of_interest())
        col_names.append("Averaged")
        col_names.append("Normalised")
        col_names.append("Clipped")
        col_names.append("Preprocessing")
        col_names.append("ModelName")
        col_names.append("ImageSetLabel")
        return col_names

    # for dataframe output but not logged in pdf
    def get_fit_summary_dataframe_column_names_suppliment(self, cf_model):
        col_names = ["PhantomMake", "PhantomModel", "PhantomSN", "PhantomTemp",
                     "ScannerMake", "ScannerModel", "ScannerSN", "FieldStrength",
                     "ExperimentDate", "ExperimentTime"]
        for gfit_param in GOODNESS_OF_FIT_PARAMS:
            col_names.append(gfit_param)
        return col_names

    def add_fit_summary_to_df(self, df, cf_model, include_all=False):
        data_list = [mu.ROI_LABEL_IDX_MAP[self.label], self.label]
        symbol_of_interest = cf_model.get_symbol_of_interest()
        ref_val = self.reference_value
        for param_name in cf_model.get_ordered_parameter_symbols():
            if self.has_been_fit():
                fit_vec = self.voxel_fit_param_array_dict[param_name]
                fit_err_vec = self.voxel_fit_err_param_array_dict[param_name]
                mean_val = np.nanmean(fit_vec)
                data_list.append(mean_val)
                if self.is_averaged:
                    data_list.append(fit_err_vec[0]) # use the covariance of the parameter fit on a single average signal
                else:
                    data_list.append(np.nanstd(fit_vec)) # calculate the stddev of the parameter fit on all voxels
                                                         # or in the case of slice averaging the stddev of fit on all slices
                if param_name == symbol_of_interest:
                    data_list.append(mean_val-ref_val)
                    data_list.append(100.0 * (mean_val-ref_val)/ref_val)
            else:
                data_list.append(np.nan) # mean
                data_list.append(np.nan) # std.dev
                if param_name == symbol_of_interest:
                    data_list.append(np.nan) # error
                    data_list.append(np.nan) # percent error
        data_list.append(ref_val)
        data_list.append(self.initialisation_value)
        data_list.append(self.is_averaged or self.is_slice_averaged)
        data_list.append(self.is_normalised)
        data_list.append(self.is_clipped)
        data_list.append(cf_model.get_preproc_name())
        data_list.append(cf_model.get_model_name())
        data_list.append(cf_model.get_imageset_name())
        col_names = self.get_fit_summary_dataframe_column_names(cf_model)
        # add supplimentary rows if requested
        if include_all:
            # phantom details
            ref_phan = cf_model.reference_phantom
            data_list.append(ref_phan.make)
            data_list.append(ref_phan.model)
            data_list.append(ref_phan.serial_number)
            data_list.append(ref_phan.temperature)
            # scanner details
            im_set = cf_model.image_set
            data_list.append(im_set.scanner_make)
            data_list.append(im_set.scanner_model)
            data_list.append(im_set.scanner_serial_number)
            data_list.append(im_set.scanner_field_strength)
            # acquisition timestamp
            data_list.append(im_set.date)
            data_list.append(im_set.time)
            col_names = col_names + self.get_fit_summary_dataframe_column_names_suppliment(cf_model)
            # goodness of fit parameters
            for gfit_param in GOODNESS_OF_FIT_PARAMS:
                mean_val = np.nanmean(self.voxel_fit_goodness_array_dict[gfit_param])
                data_list.append(mean_val)
        # create the ROI dataframe and append it to the passed dataframe
        df_roi = pd.DataFrame([data_list], columns=col_names)
        df_roi[['Averaged', 'Normalised', 'Clipped']] = df_roi[['Averaged', 'Normalised', 'Clipped']].astype(bool)
        df[['Averaged', 'Normalised', 'Clipped']] = df[['Averaged', 'Normalised', 'Clipped']].astype(bool)
        return pd.concat([df, df_roi], ignore_index=True)


    def visualise(self, cf_model, ax, line_alpha=0.8, marker_alpha=0.5, show_y_label=False):
        roi_rgb = self.colour.get_ROI_colour(self.label)
        # DRAW THE RAW DATA
        if self.is_averaged:
            ax.errorbar(self.get_measurement_series(), self.get_voxel_series(0), yerr=self.get_voxel_std_series(),
                        color=(roi_rgb[0], roi_rgb[1], roi_rgb[2], marker_alpha), fmt='.', linewidth=1)
            for x, y, reason_list in zip(self.get_excluded_measurement_series(),
                                         self.get_excluded_voxel_series(0),
                                         self.get_excluded_reason_series()):
                if "clipped" in reason_list:
                    ax.scatter(x, y, s=10, c='k', marker='x', linewidth=1)
                else:
                    ax.scatter(x, y, s=10, c='k', marker='o', linewidth=1)
        else:
            for vox_dx in range(self.get_number_fit_voxels()):
                ax.scatter(self.get_measurement_series(), self.get_voxel_series(vox_dx),
                           s=2, c=[(roi_rgb[0], roi_rgb[1], roi_rgb[2], marker_alpha)], marker='.', linewidth=1)
                for x, y, reason_list in zip(self.get_excluded_measurement_series(),
                                             self.get_excluded_voxel_series(vox_dx),
                                             self.get_excluded_reason_series()):
                    if "clipped" in reason_list:
                        ax.scatter(x, y, s=10, c='k', marker='x', linewidth=1)
                    else:
                        ax.scatter(x, y, s=10, c='k', marker='o', linewidth=1)
        # DRAW THE FIT
        average_param_fit = None
        symbol_of_interest = cf_model.get_symbol_of_interest()
        if self.has_been_fit():
            val_min, val_max = np.min(self.meas_var_vector), np.max(self.meas_var_vector)
            meas_fine_vec = np.linspace(val_min * 0.9, val_max * 1.1, 100)
            if self.is_averaged:
                param_names = cf_model.get_ordered_parameter_symbols()
                fitted_kwargs = {}
                for param in param_names:
                    fitted_kwargs[param] = self.voxel_fit_param_array_dict[param][0]
                fitted_kwargs[cf_model.get_meas_parameter_symbol()] = meas_fine_vec
                fit_fine_vec = cf_model.fit_function(**fitted_kwargs)
                ax.plot(meas_fine_vec, fit_fine_vec, c=roi_rgb,
                        alpha=line_alpha, linewidth=1.0)
                average_param_fit = fitted_kwargs[symbol_of_interest]
            else:
                # plot the mean fit +- std fit
                param_names = cf_model.get_ordered_parameter_symbols()
                mean_fitted_kwargs = {}
                mean_plus_std_fitted_kwargs = {}
                mean_minus_std_fitted_kwargs = {}
                for param in param_names:
                    mean_param = np.nanmean(self.voxel_fit_param_array_dict[param])
                    mean_stddev = np.nanstd(self.voxel_fit_param_array_dict[param])
                    mean_fitted_kwargs[param] = mean_param
                    mean_plus_std_fitted_kwargs[param] = mean_param + mean_stddev
                    mean_minus_std_fitted_kwargs[param] = mean_param - mean_stddev
                for param_kwargs, fmt_str in zip([mean_fitted_kwargs, mean_plus_std_fitted_kwargs, mean_minus_std_fitted_kwargs],
                                                 ["-", "--", "--"]):
                    param_kwargs[cf_model.get_meas_parameter_symbol()] = meas_fine_vec
                    fit_fine_vec = cf_model.fit_function(**param_kwargs)
                    ax.plot(meas_fine_vec, fit_fine_vec, c=roi_rgb, linestyle=fmt_str,
                            alpha=line_alpha, linewidth=1.0)
                average_param_fit = mean_fitted_kwargs[symbol_of_interest]
            # add a title
            if symbol_of_interest == "ADC":
                ax.set_title("%s\n%s=%0.1f %s\n%s_ref=%0.1f %s" % (self.label,
                                                                    symbol_of_interest, average_param_fit, "µm²/s",
                                                                    symbol_of_interest, self.reference_value, "µm²/s"), fontsize=10)
            else:
                ax.set_title("%s\n%s=%0.1f %s\n(%s_ref=%0.1f %s)" % (self.label,
                                                                    symbol_of_interest, average_param_fit, "ms",
                                                                    symbol_of_interest, self.reference_value, "ms"),
                            fontsize=10)
        # FORMAT
        ax.grid('on')
        ax.set_xlabel('%s (%s)' % (self.meas_var_name, self.meas_var_units))
        if show_y_label:
            ax.set_ylabel('signal (a.u.)')
        if self.is_normalised:
            ax.set_ylim(-0.05, 1.05)
            if not show_y_label:
                ax.set_yticklabels([])



class CurveFitAbstract(ABC):
    """
    This class is the AbstractCurveFit class that inherits the AbstractBaseClass (ABC) and is used to create the
    specific MR sequence curve fit class.
    """
    def __init__(self, imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                 exclusion_list=None, exclusion_label=None, optimisation_lib=OptiOptions.LMFIT,
                 use_2D_roi=False, centre_offset_2D_list=[0]):
        """

        Args:
            image_set: ImageSet concrete class which contains the measurement set and associated ROI masks
            reference_phantom: ReferencePhantomAbstract concrete class with reference values for each ROI
            initialisation_phantom: Phantom object with initialisation values for the curve fit
            preprocessing_options: a dictionary of pre-processing options
        """
        self.image_set = imageset
        self.reference_phantom = reference_phantom
        self.initialisation_phantom = initialisation_phantom
        self.preproc_dict = preprocessing_options
        self.exclusion_list = exclusion_list
        self.exclusion_label = exclusion_label
        self.opti_lib = optimisation_lib
        self.cf_rois = OrderedDict()

        self.use_2D_roi = use_2D_roi
        self.centre_offset_2D_list = centre_offset_2D_list

        mu.log("CurveFit(%s)::__init__(): creating curve fit ROIs ..." % self.get_model_name(), LogLevels.LOG_INFO)
        # construct a set of curve fitting ROIs including the raw voxel data
        # - get the raw data
        im_set_roi_dict = self.image_set.get_ROI_data()

        # generate color settings to be used by all ROIs
        roi_index_list = []
        for roi_dx, im_roi in im_set_roi_dict.items():
            roi_index_list.append(roi_dx)
        self.colour = mu.ColourSettings(roi_index_list=roi_index_list)

        # find a central slice over all the ROIs (central slice for image)
        centre_dx_list = []
        if self.use_2D_roi:
            all_z_arr = []
            for roi_dx, im_roi in im_set_roi_dict.items():
                all_z_arr = all_z_arr + im_roi.voxel_data_xyz[0].tolist()
            all_z_arr = np.array(all_z_arr).flatten()
            for centre_offset_2D in self.centre_offset_2D_list:
                centre_dx_list.append(int(np.mean(all_z_arr)) + centre_offset_2D)
        # - link the data with an (x,y,z) coordinate
        for roi_dx, im_roi in im_set_roi_dict.items():
            ref_roi = self.reference_phantom.get_roi(im_roi.label)
            init_roi = self.initialisation_phantom.get_roi(im_roi.label)
            voxel_data_arr = im_roi.voxel_data_array
            voxel_xyz_arr = im_roi.voxel_data_xyz
            voxel_pmap_dict = im_roi.voxel_pmap_dict
            derived_map_dict = im_roi.derived_map_dict
            # use only a central slice if 2D ROI is configured
            if self.use_2D_roi:
                mu.log("CurveFit(%s)::__init__(): adding the 2D region only ..." % self.get_model_name(),
                       LogLevels.LOG_INFO)
                assert len(centre_dx_list)
                voxel_data_arr = []
                voxel_x_arr = []
                voxel_y_arr = []
                voxel_z_arr = []
                for vox_data, x, y, z in zip(im_roi.voxel_data_array,
                                             im_roi.voxel_data_xyz[0],
                                             im_roi.voxel_data_xyz[1],
                                             im_roi.voxel_data_xyz[2]):
                    if x in centre_dx_list:
                        voxel_data_arr.append(vox_data)
                        voxel_x_arr.append(x)
                        voxel_y_arr.append(y)
                        voxel_z_arr.append(z)
                voxel_data_arr = np.array(voxel_data_arr)
                voxel_xyz_arr = (voxel_x_arr, voxel_y_arr, voxel_z_arr)
                # handle 2D for the ROI pmaps  (i.e. cylindrical coordinates)
                voxel_pmap_dict = OrderedDict()
                for pmap_name, pmap_arr in im_roi.voxel_pmap_dict.items():
                    pmap_2D_arr = []
                    for pmap_data, x in zip(pmap_arr,
                                            im_roi.voxel_data_xyz[0]):
                        if x in centre_dx_list:
                            pmap_2D_arr.append(pmap_data)
                    voxel_pmap_dict[pmap_name] = np.array(pmap_2D_arr)
                # handle 2D for derived maps (i.e. ADC maps)
                derived_map_dict = OrderedDict()
                if im_roi.derived_map_dict is not None:
                    for pmap_label, (pmap_arr, pmap_name, pmap_units) in im_roi.derived_map_dict.items():
                        dmap_2D_arr = []
                        for dmap_data, x in zip(pmap_arr,
                                                im_roi.voxel_data_xyz[0]):
                            if x in centre_dx_list:
                                dmap_2D_arr.append(dmap_data)
                        derived_map_dict[pmap_label] = (np.array(dmap_2D_arr), pmap_name, pmap_units)
            # - link it with a reference and initialisation value
            cf_roi = CurveFitROI(label=im_roi.label,
                                 voxel_data_array=voxel_data_arr,
                                 voxel_data_xyz=voxel_xyz_arr,
                                 measurement_variable_vector=im_roi.meas_var_vector,
                                 measurement_variable_name=im_roi.meas_var_name,
                                 measurement_variable_units=im_roi.meas_var_units,
                                 reference_value=ref_roi.value,
                                 initialisation_value=init_roi.value,
                                 exclusion_list=exclusion_list,
                                 exclusion_label=exclusion_label,
                                 voxel_pmap_dict=voxel_pmap_dict,
                                 derived_map_dict=derived_map_dict,
                                 rescale_slope_list=im_roi.rescale_slope_list,
                                 rescale_intercept_list=im_roi.rescale_intercept_list,
                                 scale_slope_list=im_roi.scale_slope_list,
                                 scale_intercept_list=im_roi.scale_intercept_list,
                                 bits_allocated=im_roi.bits_allocated,
                                 bits_stored=im_roi.bits_stored,
                                 scanner_make=im_roi.scanner_make,
                                 color_settings=self.colour)
            self.cf_rois[roi_dx] = cf_roi


        # check ROIs exist
        self.rois_found = len(list(self.cf_rois.values()))
        if self.rois_found:
            # pre-process the voxel data for curvefitting
            self.preprocess_roi_data()
            # fit the model to the processed voxel data
            self.model_fit_roi_data()
            # log the summary table
            self.log_summary_dataframe()
        else:
            mu.log("CurveFit(%s)::__init__(): no ROIs found on imageset (%s)" % (self.get_model_name(), self.get_imageset_name()), LogLevels.LOG_WARNING)


    def preprocess_roi_data(self):
        mu.log("CurveFit(%s)::preprocess_roi_data(): pre-processing the ROIs ..."
               % self.get_model_name(), LogLevels.LOG_INFO)
        # handle user specified exclusion
        self.__preproc_exclude_user_list()
        # regardless of input datatype (i.e. uint16) cast to float for further averaging, normalisation & model fitting
        self.__preproc_cast_to_float()
        # handle other exclusion rules
        if 'exclude' in self.preproc_dict.keys():
            if self.preproc_dict['exclude'] == ExclusionOptions.CLIPPED_VALUES:
                mu.log("CurveFit(%s)::preprocess_roi_data(): \t\t excluding any clipped values ..."
                       % self.get_model_name(), LogLevels.LOG_INFO)
                if not ('percent_clipped_threshold' in self.preproc_dict.keys()):
                    mu.log("CurveFit(%s)::preprocess_roi_data(): \t\t exclusion threshold not defined, defaulting to 100 pct ..."
                           % self.get_model_name(), LogLevels.LOG_WARNING)
                    self.preproc_dict['percent_clipped_threshold'] = 100.0
                self.__preproc_exclude_clipped_values(percent_clipped_values=self.preproc_dict['percent_clipped_threshold'])
        if 'average' in self.preproc_dict.keys():
            if self.preproc_dict['average'] == AveragingOptions.AVERAGE_ROI:
                mu.log("CurveFit(%s)::preprocess_roi_data(): \t\t averaging the signal over the whole ROI ..."
                       % self.get_model_name(),
                       LogLevels.LOG_INFO)
                self.__preproc_average_values_over_roi()
            elif self.preproc_dict['average'] == AveragingOptions.AVERAGE_PER_SLICE:
                mu.log("CurveFit(%s)::preprocess_roi_data(): \t\t averaging the signal over each slice ..."
                       % self.get_model_name(),
                       LogLevels.LOG_INFO)
                self.__preproc_average_values_over_slice()
        if 'normalise' in self.preproc_dict.keys():
            if self.preproc_dict['normalise'] == NormalisationOptions.ROI_MAX:
                mu.log("CurveFit(%s)::preprocess_roi_data(): \t\t normalising to the ROI maximum ..."
                       % self.get_model_name(), LogLevels.LOG_INFO)
                self.__preproc_normalise_to_max_in_roi()
            elif self.preproc_dict['normalise'] == NormalisationOptions.VOXEL_MAX:
                mu.log("CurveFit(%s)::preprocess_roi_data(): \t\t normalising to each voxel maximum ..."
                       % self.get_model_name(), LogLevels.LOG_INFO)
                self.__preproc_normalise_each_voxel_max()
        # perform an estimate of the initialisation value for each ROI
        self.__preproc_estimate_cf_start_point()

    # todo: check if other scanner data has different number range / clipping value
    def __preproc_exclude_clipped_values(self, percent_clipped_values=10):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.exclude_clipped_values(percent_clipped_values)

    def __preproc_exclude_user_list(self):
        if self.exclusion_list is not None:
            for roi_dx, cf_roi in self.cf_rois.items():
                cf_roi.exclude_user_values()

    def __preproc_cast_to_float(self):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.cast_to_float()

    def __preproc_average_values_over_roi(self):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.average_over_roi()

    def __preproc_average_values_over_slice(self):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.average_over_each_slice()

    def __preproc_normalise_to_max_in_roi(self):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.normalise_to_max_in_roi()

    def __preproc_normalise_each_voxel_max(self):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.normalise_to_max_per_voxel()

    def __preproc_estimate_cf_start_point(self):
        for roi_dx, cf_roi in self.cf_rois.items():
            cf_roi.estimate_cf_start_point(self)

    def is_normalised(self):
        return 'normalise' in self.preproc_dict.keys()

    def model_fit_roi_data(self):
        n_roi = len(self.cf_rois.keys())
        for roi_count, (roi_dx, cf_roi) in enumerate(self.cf_rois.items()):
            n_cf_vox = cf_roi.get_number_fit_voxels()
            mu.log("CurveFit(%s)::model_fit_roi_data(): fitting data for %s [%d/%d] for %d voxels ..." %
                   (self.get_model_name(), cf_roi.label, roi_count + 1, n_roi, n_cf_vox), LogLevels.LOG_INFO)
            # create a datastructure to store the curve fit result parameters for each voxel
            voxel_fit_param_array_dict = {}
            voxel_fit_param_err_array_dict = {}
            ord_param_names = self.get_ordered_parameter_symbols()
            for fit_param in ord_param_names:
                voxel_fit_param_array_dict[fit_param] = []
                voxel_fit_param_err_array_dict[fit_param] = []
            # create datatstructes to store the goodness of fit parameters
            voxel_fit_goodness_array_dict = {}
            for gfit_param in GOODNESS_OF_FIT_PARAMS:
                voxel_fit_goodness_array_dict[gfit_param] = []
            # loop over all the voxels/signals in a ROI
            for vox_dx in range(n_cf_vox):
                # mu.log("CurveFit(%s)::model_fit_roi_data():         voxel [%d/%d]"
                #        % (self.get_model_name(), vox_dx + 1, n_cf_vox), LogLevels.LOG_INFO)
                # get the pre-processed/non-excluded voxels and measurment data
                measurement_series = cf_roi.get_measurement_series()
                voxel_series = cf_roi.get_voxel_series(vox_dx)
                if len(voxel_series) > 0:
                    # fit it!
                    try:
                        # FIT WITH SELECTED LIBRARY
                        # ---------------------------------------------
                        #
                        # SCIPY
                        if self.opti_lib == OptiOptions.SCIPY:
                            # mu.log("CurveFit(%s)::model_fit_roi_data(): fitting data with SCIPY ..." % self.get_model_name(),
                            #        LogLevels.LOG_INFO)
                            popt, pcov = spopt.curve_fit(self.fit_function,
                                                         measurement_series,
                                                         voxel_series,
                                                         p0=self.get_initial_parameters(roi_dx, vox_dx),
                                                         bounds=self.fit_parameter_bounds())
                            perr = np.sqrt(np.diag(pcov))
                            assert len(popt) == len(ord_param_names)
                            assert len(perr) == len(ord_param_names)
                            for p_name, opti_param, opti_err in zip(ord_param_names, popt, perr):
                                voxel_fit_param_array_dict[p_name].append(opti_param)
                                voxel_fit_param_err_array_dict[p_name].append(opti_err)
                        #
                        # LMFIT
                        elif self.opti_lib == OptiOptions.LMFIT:
                            # mu.log("CurveFit(%s)::model_fit_roi_data(): fitting data with LMFIT ..." % self.get_model_name(),
                            #        LogLevels.LOG_INFO)
                            mdl = lmfit.Model(self.fit_function,
                                              independent_vars=[self.get_meas_parameter_symbol()],
                                              nan_policy='propagate')
                            mld_pars = mdl.make_params()
                            # add the initial pars and bounds
                            bnds = self.fit_parameter_bounds()
                            bnds_min = [b for b in bnds[0]]
                            bnds_max = [b for b in bnds[1]]
                            for par_lbl, par_init, par_bound_min, par_bound_max in \
                                    zip(ord_param_names,
                                        self.get_initial_parameters(roi_dx, vox_dx),
                                        bnds_min, bnds_max):
                                if par_bound_max is np.inf:
                                    mld_pars.add(par_lbl, value=par_init, min=par_bound_min)
                                else:
                                    mld_pars.add(par_lbl, value=par_init, min=par_bound_min, max=par_bound_max)
                            # fit to the data
                            kwargs = {self.get_meas_parameter_symbol(): measurement_series,
                                      'params': mld_pars,
                                      'calc_covar': True}
                            n_point_to_fit = len(voxel_series)
                            n_model_params = len(mld_pars)
                            if n_point_to_fit > n_model_params:
                                result = mdl.fit(voxel_series, **kwargs)
                                # get the err out
                                popt = []
                                perr = []
                                for par_lbl in ord_param_names:
                                    popt.append(result.best_values[par_lbl])
                                    perr.append(np.inf)
                                # if its available use it
                                if result.covar is not None:
                                    perr = np.sqrt(np.diag(result.covar))

                                assert len(popt) == len(ord_param_names), print(popt, ord_param_names)
                                assert len(perr) == len(ord_param_names), print(perr, ord_param_names)
                                for p_name, opti_param, opti_err in zip(ord_param_names, popt, perr):
                                    voxel_fit_param_array_dict[p_name].append(opti_param)
                                    voxel_fit_param_err_array_dict[p_name].append(opti_err)

                                # store the goodness of fit parameters
                                for gfit_param in GOODNESS_OF_FIT_PARAMS:
                                    voxel_fit_goodness_array_dict[gfit_param].append(getattr(result, gfit_param))
                            else:
                                mu.log("CurveFit(%s)::model_fit_roi_data(): number of model parameters (%d) should not "
                                       "exceed the number of measurement samples (%d)" %
                                       (self.get_model_name(), n_model_params, n_point_to_fit), LogLevels.LOG_WARNING)
                                for p_name in ord_param_names:
                                    voxel_fit_param_array_dict[p_name].append(np.nan)
                                    voxel_fit_param_err_array_dict[p_name].append(np.nan)
                        #
                        # LINEAR REGRESSION
                        elif self.opti_lib == OptiOptions.LINREG:
                            voxel_series = self.linearise_voxel_series(measurement_series, voxel_series)
                            popt_res = spstat.linregress(measurement_series, voxel_series)
                            self.nonlinearise_fitted_params(popt_res.slope, popt_res.intercept,
                                                            popt_res.stderr, popt_res.intercept_stderr,
                                                            voxel_fit_param_array_dict,
                                                            voxel_fit_param_err_array_dict)
                        else:
                            mu.log("CurveFit(%s)::model_fit_roi_data(): please select a valid optimisation library choice" %
                                   self.get_model_name(), LogLevels.LOG_WARNING)

                    except RuntimeError as e:
                        mu.log("CurveFit(%s)::model_fit_roi_data(): failed curve fit!" %
                               self.get_model_name(), LogLevels.LOG_WARNING)
                        for p_name in ord_param_names:
                            voxel_fit_param_array_dict[p_name].append(np.nan)
                            voxel_fit_param_err_array_dict[p_name].append(np.nan)
            # store the fit data in the curvefitROI
            cf_roi.set_voxel_fit_data(voxel_fit_param_array_dict)
            cf_roi.set_voxel_fit_err_data(voxel_fit_param_err_array_dict)
            cf_roi.set_voxel_fit_goodness_data(voxel_fit_goodness_array_dict)

    def get_voxel_dataframe(self):
        # check ROI data exists
        if self.rois_found:
            cf_roi_list = list(self.cf_rois.values())
            col_names = cf_roi_list[0].get_voxel_dataframe_column_names(self)
            df = pd.DataFrame(columns=col_names)
            # loop over ROIs and add the data to the dataframe
            for cf_roi in cf_roi_list:
                df = cf_roi.add_voxel_data_to_df(df, self)
            return df
        return None

    def get_summary_dataframe(self, include_all=False):
        # check ROI data exists
        if self.rois_found:
            cf_roi_list = list(self.cf_rois.values())
            col_names = cf_roi_list[0].get_fit_summary_dataframe_column_names(self)
            if include_all:
                col_names = col_names + cf_roi_list[0].get_fit_summary_dataframe_column_names_suppliment(self)
            df = pd.DataFrame(columns=col_names)
            # loop over ROIs and add the data to the dataframe
            for cf_roi in cf_roi_list:
                df = cf_roi.add_fit_summary_to_df(df, self, include_all=include_all)
            return df
        return None


    def write_data(self, data_dir, write_voxel_data=False):
        if self.rois_found:
            if write_voxel_data:
                self.write_voxel_dataframe(data_dir)
            self.write_fit_summary_dataframe(data_dir)

    def write_voxel_dataframe(self, data_dir):
        # check ROI data exists
        if self.rois_found:
            df = self.get_voxel_dataframe()
            voxel_datafilename = os.path.join(data_dir, "voxel_data.csv")
            mu.log("CurveFit(%s)::write_voxel_dataframe(): writing voxel curve fit data to %s" %
                   (self.get_model_name(), voxel_datafilename), LogLevels.LOG_INFO)
            df.to_csv(voxel_datafilename)
        else:
            mu.log("CurveFit(%s)::write_voxel_dataframe(): no voxel curve fit data to write (skipping)" %
                   self.get_model_name(), LogLevels.LOG_WARNING)

    def write_fit_summary_dataframe(self, data_dir):
        # check ROI data exists
        if self.rois_found:
            df = self.get_summary_dataframe(include_all=True)
            summary_datafilename = os.path.join(data_dir, "model_fit_summary.csv")
            mu.log("CurveFit(%s)::write_voxel_dataframe(): writing model curve fit summary to %s" %
                   (self.get_model_name(), summary_datafilename), LogLevels.LOG_INFO)
            df.to_csv(summary_datafilename)
        else:
            mu.log("CurveFit(%s)::write_fit_summary_dataframe(): no voxel curve fit data to write (skipping)" %
                   self.get_model_name(), LogLevels.LOG_WARNING)

    # output the categorisation, set grouping & reference images (via the log)
    def log_summary_dataframe(self):
        # check ROI data exists
        if self.rois_found:
            df = self.get_summary_dataframe(include_all=False)
            header_str, col_names, row_fmt_str = self.__get_summary_dataframe_fmt()
            table_width = len(header_str)
            mu.log("=" * table_width, LogLevels.LOG_INFO)
            mu.log(header_str,  LogLevels.LOG_INFO)
            mu.log("=" * table_width, LogLevels.LOG_INFO)
            # cur_image_set = ""
            for idx, r in df.iterrows():
                val_list = []
                for name in col_names:
                    val_list.append(r[name])
                mu.log(row_fmt_str % tuple(val_list), LogLevels.LOG_INFO)
            mu.log("=" * table_width, LogLevels.LOG_INFO)

        # write the model equation and a list of parameters
        mu.log(" " * table_width, LogLevels.LOG_INFO)
        for eqn_line in self.get_model_eqn_strs():
            mu.log(eqn_line, LogLevels.LOG_INFO)
        mu.log(" " * table_width, LogLevels.LOG_INFO)
        header_str = "|  Parameter  |  Description                                                |    Init Val.    |     Min Val.     |     Max Val.     |"
        table_width = len(header_str)
        mu.log("-" * table_width, LogLevels.LOG_INFO)
        mu.log(header_str, LogLevels.LOG_INFO)
        mu.log("-" * table_width, LogLevels.LOG_INFO)
        for p_name, descr, init_v, min_v, max_v in self.get_model_eqn_parameter_strs():
            mu.log("| %11s | %-59s | %15s | %15s  | %15s  |" % (p_name, descr, init_v, min_v, max_v),
                   LogLevels.LOG_INFO)
        mu.log("-" * table_width, LogLevels.LOG_INFO)

    def __get_summary_dataframe_fmt(self):
        col_names = ["RoiIndex", "RoiLabel"]
        header_str = "| ROI_DX | ROI LABEL |"
        row_fmt_str = "| %6d | %9s |"
        symbol_of_interest = self.get_symbol_of_interest()
        for param_name in self.get_ordered_parameter_symbols():
            header_str = "%s %s |" % (header_str, " %4s " % param_name)
            header_str = "%s %s |" % (header_str, "%4s_var" % param_name)
            row_fmt_str = row_fmt_str + " %6.1f | %8.1f |"
            col_names.append("%s (mean)" % param_name)
            col_names.append("%s (std.dev.)" % param_name)
            if param_name == symbol_of_interest:
                header_str = "%s %s |" % (header_str, "%3s_err" % param_name)
                header_str = "%s %s |" % (header_str, "%3s_pct.err" % param_name)
                row_fmt_str = row_fmt_str + " %7.1f | %11.1f |"
                col_names.append("%s (mean error)" % param_name)
                col_names.append("%s (mean percent error)" % param_name)
        header_str = "%s %s |" % (header_str, "%3s_ref " % symbol_of_interest)
        header_str = "%s %s |" % (header_str, "%3s_init" % symbol_of_interest)
        row_fmt_str = row_fmt_str + " %8.1f | %8.1f |"
        col_names.append("%s_reference" % symbol_of_interest)
        col_names.append("%s_initialise" % symbol_of_interest)
        header_str = "%s AVRGD | NORMLD | CLIPD |" % header_str
        row_fmt_str = row_fmt_str + " %5s | %6s | %5s |"
        col_names.append("Averaged")
        col_names.append("Normalised")
        col_names.append("Clipped")
        return header_str, col_names, row_fmt_str


    def write_pdf_summary_pages(self, c, is_system, include_pmap_pages=False):
        if self.rois_found:
            self.write_pdf_fit_table_page(c)
            self.write_pdf_roi_page(c, include_pmap_pages)
            self.write_pdf_voxel_fit_page(c)
            self.write_pdf_fit_accuracy_page(c, is_system)

    def write_pdf_roi_page(self, c, include_pmap_pages=False):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font
        sup_title = "CurveFit [%s - %s] <%s>" % (self.get_model_name(),
                                                 self.get_preproc_name(),
                                                 self.get_imageset_name())
        # get some data from the image set
        mask_im = self.construct_mask_from_preproc_rois()
        self.image_set.write_roi_pdf_page(c, sup_title, mask_override_sitk_im=mask_im,
                                          include_pmap_pages=include_pmap_pages)

    def construct_mask_from_preproc_rois(self):
        # make an empty image
        all_mask_im = self.image_set.get_roi_image()
        mask_arr = np.zeros_like(sitk.GetArrayFromImage(all_mask_im))
        # add the voxels which are used in analysis
        for roi_dx, cf_roi in self.cf_rois.items():
            for x, y, z in zip(cf_roi.voxel_data_xyz[0], cf_roi.voxel_data_xyz[1], cf_roi.voxel_data_xyz[2]):
                mask_arr[x, y, z] = roi_dx
        # make the simpleITK image
        mask_im = sitk.GetImageFromArray(mask_arr)
        mask_im.SetOrigin(all_mask_im.GetOrigin())
        mask_im.SetSpacing(all_mask_im.GetSpacing())
        mask_im.SetDirection(all_mask_im.GetDirection())
        return mask_im

    def write_pdf_fit_table_page(self, c):
        # check ROI data exists
        if len(list(self.cf_rois.values())):
            # pdf settings
            pdf = mu.PDFSettings()
            c.setFont(pdf.font_name, pdf.font_size)  # set to a fixed width font

            sup_title = "CurveFit [%s - %s] <%s>" % (self.get_model_name(),
                                                     self.get_preproc_name(),
                                                     self.get_imageset_name())
            c.drawString(pdf.left_margin*6, pdf.page_height - pdf.top_margin, sup_title)


            c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font
            df = self.get_summary_dataframe(include_all=True)
            header_str, col_names, row_fmt_str = self.__get_summary_dataframe_fmt()
            table_width = len(header_str)

            # -------------------------------------------------------------
            # TABLE WITH THE CURVE FIT SUMMARY
            # -------------------------------------------------------------
            # header
            off_dx = 3.5
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - off_dx * pdf.small_line_width,
                         "=" * table_width)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - (off_dx + 1) * pdf.small_line_width, header_str)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - (off_dx + 2) * pdf.small_line_width,
                         "=" * table_width)
            off_dx = off_dx + 3

            # table content
            for line_dx, (idx, r) in enumerate(df.iterrows()):
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
                         "SIGNAL EQUATION:")
            off_dx += 2
            for eqn_line in self.get_model_eqn_strs():
                c.drawString(pdf.left_margin*2, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx), eqn_line)
                off_dx = off_dx + 1
            off_dx = off_dx + 1
            # table of parameters
            header_str = "|  Parameter  |  Description                                                |    Init Val.    |     Min Val.     |     Max Val.     |"
            table_width = len(header_str)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx), "-" * table_width)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx + 1), header_str)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx + 2), "-" * table_width)
            off_dx = off_dx + 3
            for p_name, descr, init_v, min_v, max_v in self.get_model_eqn_parameter_strs():
                c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx),
                             "| %11s | %-59s | %15s | %15s  | %15s  |" % (p_name, descr, init_v, min_v, max_v))
                off_dx = off_dx + 1
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx), "-" * table_width)
            s_dx += off_dx + 6

            # -------------------------------------------------------------
            # TABLE WITH GOODNESS OF FIT
            # -------------------------------------------------------------
            # build the header
            off_dx = 0
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx),
                         "GOODNESS OF FIT:")
            off_dx += 2
            header_str = "| ROI_DX | ROI LABEL |"
            row_fmt_str = "| %6d | %9s |"
            for gfit_param in GOODNESS_OF_FIT_PARAMS:
                header_str += " %15s |" % gfit_param
                row_fmt_str += " %15s |"
            table_width = len(header_str)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx), "-" * table_width)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx + 1), header_str)
            c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx + 2), "-" * table_width)
            off_dx = off_dx + 3

            # table content
            for line_dx, (idx, r) in enumerate(df.iterrows()):
                val_list = [r.RoiIndex, r.RoiLabel]
                for name in GOODNESS_OF_FIT_PARAMS:
                    val_list.append("%8.7f" % r[name])
                c.drawString(pdf.left_margin,
                             pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + line_dx + off_dx),
                             row_fmt_str % tuple(val_list))
            # final borderline
            c.drawString(pdf.left_margin,
                         pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + line_dx + off_dx + 1),
                         "=" * table_width)
            s_dx += line_dx + off_dx + 2
            off_dx=0.5
            for name in GOODNESS_OF_FIT_PARAMS:
                c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width * (s_dx + off_dx),
                             "%15s : %s" % (name, GOODNESS_OF_FIT_DESC_DICT[name]))
                off_dx += 1

            c.showPage()  # new page

    def write_pdf_voxel_fit_page(self, c):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font
        sup_title = "CurveFit [%s - %s] <%s>" % (self.get_model_name(),
                                                 self.get_preproc_name(),
                                                 self.get_imageset_name())
        # draw the voxel fit summary figure
        # -----------------------------------------------------------
        # setup figure
        roi_vec = list(self.cf_rois.values())
        n_rois = len(roi_vec)
        n_rows = 2
        rois_per_row = int(np.ceil(n_rois/float(n_rows)))
        f, axes_arr = plt.subplots(n_rows, rois_per_row)
        if n_rows == 1 or rois_per_row == 1: # hacky/likely to fail
            axes_arr = [axes_arr]
        f.suptitle(sup_title)
        f.set_size_inches(14, 8)
        # loop over the axes and plot the roi decay/recovery curves
        roi_dx = 0
        for axes_row in axes_arr:
            for ax_dx, ax in enumerate(axes_row):
                if roi_dx < n_rois:
                    show_y_label = False
                    if ax_dx == 0:
                        show_y_label = True
                    roi_vec[roi_dx].visualise(self, ax=ax, show_y_label=show_y_label)
                    roi_dx = roi_dx + 1
                else:
                    # hide any unused axes
                    ax.axis('off')
        # set the spacing between subplots
        if self.is_normalised():
            f.subplots_adjust(left=0.1,
                              bottom=0.1,
                              right=0.9,
                              top=0.8,
                              wspace=0.1,
                              hspace=0.6)
        else:
            f.subplots_adjust(left=0.1,
                              bottom=0.1,
                              right=0.9,
                              top=0.8,
                              wspace=0.5,
                              hspace=0.6)
        # rotate the x-axis labels
        roi_dx = 0
        for axes_row in axes_arr:
            for ax_dx, ax in enumerate(axes_row):
                if roi_dx < n_rois:
                    ax.tick_params("x", rotation=45)
                    roi_dx = roi_dx + 1
        # draw it on the pdf
        pil_f = mu.mplcanvas_to_pil(f)
        width, height = pil_f.size
        height_3d, width_3d = pdf.page_width * (height / width), pdf.page_width
        c.drawImage(ImageReader(pil_f),
                    0,
                    pdf.page_height - pdf.top_margin - height_3d - 2.5*pdf.line_width,
                    width_3d,
                    height_3d)
        c.drawString(pdf.left_margin,
                     pdf.line_width,
                     "Included measurements are denoted with colour markers. Excluded measurements are denoted with black markers for (crosses) clipped or (circles) user excluded measurements.")
        plt.close(f)
        # -------------------------------------------------------------
        c.showPage()  # new page

    def write_pdf_fit_accuracy_page(self, c, is_system,
                                    central_limit_pcnt=50,
                                    central_ticks=[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
                                    abs_limit_line=25):
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font
        sup_title = "CurveFit [%s - %s] <%s>" % (self.get_model_name(),
                                                 self.get_preproc_name(),
                                                 self.get_imageset_name())
        # get the fit data and restructure
        df = self.get_voxel_dataframe()
        df = df.sort_values(['RoiIndex'], ascending=[True])
        df = df.drop_duplicates(subset=["RoiLabel", "VoxelID"])
        symbol_of_interest = self.get_symbol_of_interest()
        reference_label = "%s_reference" % symbol_of_interest
        error_name = "%s (%sbias)" % (self.get_symbol_of_interest(), "%")
        df[error_name] = 100.0*(df[symbol_of_interest]-df[reference_label])/df[reference_label]
        df_plot = df[["RoiLabel",
                      "VoxelID",
                      symbol_of_interest,
                      reference_label,
                      error_name]]
        err_min, err_max = np.min(df[error_name]), np.max(df[error_name])

        # setup the figure
        with sns.axes_style(mu.SEABORN_STYLE):
            # create the number of axes based on the range of error in the results
            # - ax1: for error exceeding the central limit
            # - ax2: always present showing error around zero (+- central limit)
            # - ax3: for error bellow the central limit
            ax1, ax2, ax3 = None, None, None
            if (err_min < -central_limit_pcnt) and (central_limit_pcnt < err_max):
                f, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 2, 1]})
            elif (err_min < -central_limit_pcnt):
                f, (ax2, ax3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
            elif (central_limit_pcnt < err_max):
                f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
            else:
                f, ax2 = plt.subplots(1, 1)
            f.suptitle(sup_title)
            f.set_size_inches(12, 8)
            f.subplots_adjust(bottom=0.2)
            # create the color pallete
            roi_labels = df_plot.drop_duplicates(subset=["RoiLabel"]).RoiLabel
            col_pal = []
            for roi_label in roi_labels:
                col_pal.append(self.colour.get_ROI_colour(roi_label))

            axs = [ax for ax in [ax1, ax2, ax3] if (ax is not None)]
            for ax in axs:
                # draw the error plot
                box_linewidth = 1
                strip_jitter = 0.15
                strip_alpha = 0.5
                if 'average' in self.preproc_dict.keys():
                    box_linewidth = 0
                    strip_jitter = 0
                    strip_alpha = 0.9
                sns.boxplot(x="RoiLabel", y=error_name, data=df_plot, ax=ax,
                            boxprops=dict(alpha=.5), showfliers=False, linewidth=box_linewidth,
                            order=roi_labels, palette=col_pal)
                sns.stripplot(x="RoiLabel", y=error_name, data=df_plot, ax=ax,
                              jitter=strip_jitter, dodge=False, alpha=strip_alpha,
                              order=roi_labels, palette=col_pal, hue="RoiLabel", legend=False)

            # update the x labels to include the reference value
            xtick_label_vec = []
            for roi_name in roi_labels:
                df_ROIS = df_plot.drop_duplicates(subset=["RoiLabel"])
                df_roi = df_ROIS[df_ROIS["RoiLabel"] == roi_name]
                roi_ref_relax_val = df_roi[reference_label]
                if self.get_model_name() == "DWCurveFit2param":
                    xtick_label_vec.append("%s\n(%s=%0.2f µm²/s)" % (roi_name,
                                                                        reference_label.replace("_reference", "_ref"),
                                                                        roi_ref_relax_val))
                else:
                    xtick_label_vec.append("%s\n(%s=%0.2f ms)" % (roi_name,
                                                                reference_label.replace("_reference", "_ref"),
                                                                roi_ref_relax_val))
            # update the x-axis label
            for ax in [ax1, ax2, ax3]:
                if ax is not None:
                    ax.set_xlabel("ROI Label")

            # set the axis limits
            ax2.set(ylim=(-central_limit_pcnt, central_limit_pcnt))
            if ax1 is not None:
                ax1.set(ylim=(central_limit_pcnt, err_max+10)) # 10 to padd past the highest value
            if ax3 is not None:
                ax3.set(ylim=(err_min-10, -central_limit_pcnt)) # 10 to padd below the lowest value
            # turn off ax1 x-axis labels
            if ax1 is not None:
                ax1.get_xaxis().set_ticklabels([])
                ax1.set_xlabel("")
                ax1.set_ylabel("")
            # draw the x-axis on the lowest plot available
            if ax3 is not None:
                ax3.set_ylabel("")
                ax3.get_xaxis().set_ticklabels(xtick_label_vec)
                plt.setp(ax3.get_xticklabels(), rotation=45, horizontalalignment="right")
                ax2.get_xaxis().set_ticklabels([])
                ax2.set_xlabel("")
            else:
                ax2.get_xaxis().set_ticklabels(xtick_label_vec)
                plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment="right")

            # add the central detail
            ax2.set(yticks=central_ticks)
            ax2.axhline(y=0, linewidth=2, color='gray', alpha=0.5)
            for ylim in [-abs_limit_line, abs_limit_line]:
                ax2.axhline(y=ylim, linewidth=2, color='gray', linestyle="--", alpha=0.5)

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

    def get_imageset_name(self):
        return self.image_set.get_set_label()

    def get_preproc_name(self):
        preproc_str = ""
        for key, str_map in zip(['average', 'normalise', 'exclude'],
                                [AV_OPT_STR_MAP, NORM_OPT_STR_MAP, EXCL_OPT_STR_MAP]):
            if key in self.preproc_dict.keys():
                if preproc_str == "":
                    preproc_str = str_map[self.preproc_dict[key]]
                else:
                    preproc_str = "%s_%s" % (preproc_str, str_map[self.preproc_dict[key]])
        # add details of clipping threshold
        if 'exclude' in self.preproc_dict.keys():
            if self.preproc_dict['exclude'] == ExclusionOptions.CLIPPED_VALUES:
                preproc_str = "%s-%dpct" % (preproc_str, self.preproc_dict['percent_clipped_threshold'])
        # add 2D tag if using a central slice
        if self.use_2D_roi:
            preproc_str = "%s_2D_" % preproc_str
            for centre_offset_2D in self.centre_offset_2D_list:
                if centre_offset_2D >= 0:
                    preproc_str = "%s+%d" % (preproc_str, np.abs(centre_offset_2D))
                else:# centre_offset_2D< 0:
                    preproc_str = "%s-%d" % (preproc_str, np.abs(centre_offset_2D))
        # user exclusion tag
        if (isinstance(self.exclusion_list, list) and (len(self.exclusion_list) > 0)) \
                and (self.exclusion_label is not None):
            if preproc_str == "":
                preproc_str = self.exclusion_label
            else:
                preproc_str = "%s_%s" % (preproc_str, self.exclusion_label)
        return preproc_str

    def get_imset_model_preproc_name(self):
        return "%s_%s_%s" % (self.get_imageset_name(), self.get_model_name(), self.get_preproc_name())

    def get_model_eqn_parameter_strs(self):
        eqn_param_strs = []
        for p_name, (descr, init_v, min_v, max_v) in self.eqn_param_map.items():
            eqn_param_strs.append((p_name, descr, init_v, min_v, max_v))
        return eqn_param_strs

    @abstractmethod
    def estimate_cf_start_point(self, meas_vec, av_sig, init_val, cf_roi):
        return None

    @abstractmethod
    def get_model_name(self):
        return None

    @abstractmethod
    def get_meas_parameter_name(self):
        return None

    @abstractmethod
    def get_symbol_of_interest(self):
        return None
    def get_ordered_parameter_symbols(self):
        return None
    @abstractmethod
    def get_meas_parameter_symbol(self):
        return None

    @abstractmethod
    def fit_function(self, **kwargs):
        return None

    @abstractmethod
    def get_initial_parameters(self, roi_dx, voxel_dx):
        return None

    @abstractmethod
    def fit_parameter_bounds(self):
        return None

    @abstractmethod
    def get_model_eqn_strs(self):
        return None

    def linearise_voxel_series(self, measurement_series, voxel_series):
        mu.log("\t\tCurveFitAbsract::linearise_voxel_series(): not implemented for this model (%d) returning an unmodified voxel series" %
               self.get_model_name(), LogLevels.LOG_WARNING)
        return voxel_series

    def nonlinearise_fitted_params(self, slope, intercept, slope_err, intercept_err,
                                   voxel_fit_param_array_dict, voxel_fit_param_err_array_dict):
        mu.log("\t\tCurveFitAbsract::nonlinearise_fitted_params(): not implemented for this model (%d)" %
               self.get_model_name(), LogLevels.LOG_WARNING)


if __name__ == '__main__':
    main()
