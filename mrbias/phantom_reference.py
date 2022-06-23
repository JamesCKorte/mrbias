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
  13-June-2022  :               (James Korte) : GitHub Release   MR-BIAS v1.0
"""

import os
from enum import IntEnum
from abc import ABC, abstractmethod
from collections import OrderedDict
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
import mrbias.misc_utils as mu
from mrbias.misc_utils import LogLevels


def main():
    # setup the logger to write to file
    mu.initialise_logger("phantom_reference.log", force_overwrite=True, write_to_screen=True)


    field_strengths = [1.5, 3.0]
    for field in field_strengths:
        ref_phan_batch1 = ReferencePhantomCalibreSystem1(field_strength=field,    # Tesla
                                                         temperature=20.0)        # Celsius
        ref_phan_batch2 = ReferencePhantomCalibreSystem2(field_strength=field,    # Tesla
                                                         temperature=20.0)        # Celsius
        ref_phan_batch2p5 = ReferencePhantomCalibreSystem2p5(field_strength=field,  # Tesla
                                                             temperature=20.0)      # Celsius
        ReferencePhantomCalibreSystemFitInit(field_strength=field,  # Tesla
                                             temperature=20.0)      # Celsius


    mu.log("------ FIN -------", LogLevels.LOG_INFO)



class PhantomOptions(IntEnum):
    SYSTEM_PHANTOM_CALIBER_BATCH1 = 1
    SYSTEM_PHANTOM_CALIBER_BATCH2 = 2
    SYSTEM_PHANTOM_CALIBER_BATCH2p5 = 3

class ReferenceROI(object):
    def __init__(self, label, value, units, value_uncertainty=None):
        self.value = value
        self.label = label
        self.units = units
        self.value_uncertainty = value_uncertainty
        if value_uncertainty is None:
            mu.log("\t\t\t\tReferenceROI::__init__(): %s created with reference value %0.2f %s " % (self.label,
                                                                                                    self.value,
                                                                                                    self.units),
                   LogLevels.LOG_INFO)
        else:
            mu.log("\t\t\t\tReferenceROI::__init__(): %s created with reference value "
                   "%0.2f+-%0.2f %s " % (self.label,
                                         self.value,
                                         self.value_uncertainty,
                                         self.units),
                   LogLevels.LOG_INFO)


class ReferencePhantomAbstract(ABC):
    def __init__(self, phantom_type,
                 field_strength=None, temperature=None,
                 make=None, model=None, serial_number=None):
        assert isinstance(phantom_type, PhantomOptions), "ReferencePhantomAbstract::__init__(): phantom_type is " \
                                                         "expected to be type PhantomOptions (not %s)" \
                                                         % type(phantom_type)
        self.type = phantom_type
        self.field_strength = field_strength
        self.temperature = temperature
        self.ref_rois = OrderedDict()
        self.load_reference_values()
        # origin information
        self.make = make
        self.model = model
        self.serial_number = serial_number

    @abstractmethod
    def load_reference_values(self):
        return None

    def get_roi(self, label):
        if not (label in self.ref_rois.keys()):
            mu.log("ReferencePhantomAbstract::get_roi(): label (%s) not found", LogLevels.LOG_WARNING)
            return None
        return self.ref_rois[label]

    def get_roi_by_dx(self, roi_dx):
        return self.get_roi(mu.ROI_IDX_LABEL_MAP[roi_dx])



class ReferencePhantomCaibreSystem(ReferencePhantomAbstract):
    def __init__(self, phantom_type, field_strength,
                 t1_reference_file, t2_reference_file,
                 t1_concentration_roi_map, t2_concentration_roi_map,
                 temperature=None, serial_number=None):
        assert os.path.isfile(t1_reference_file), "ReferencePhantomCaibreSystem::__init__(): unable to locate " \
                                                  "T1 reference file (%s)" % t1_reference_file
        assert os.path.isfile(t2_reference_file), "ReferencePhantomCaibreSystem::__init__(): unable to locate " \
                                                  "T2 reference file (%s)" % t2_reference_file
        self.t1_reference_file = t1_reference_file
        self.t2_reference_file = t2_reference_file
        self.t1_concentration_roi_map = t1_concentration_roi_map
        self.t2_concentration_roi_map = t2_concentration_roi_map
        self.concentration_error_tol = 1e-3
        self.temp_error_tol = 5 # degrees celsius
        if temperature is None:
            temperature = 20.0
            mu.log("ReferencePhantomCaibreSystem::__init__(): "
                   "no temperature provided (using default value of 20.0 degrees celsuis)", LogLevels.LOG_WARNING)
        super().__init__(phantom_type, field_strength, temperature,
                         "CaliberMRI", "SystemPhantom", serial_number)

    def load_reference_values(self):
        self.load_t1_values()
        self.load_t2_values()

    def load_t1_values(self):
        mu.log("ReferencePhantomCaibreSystem::reading T1 reference file: %s" % self.t1_reference_file, LogLevels.LOG_INFO)
        df = pd.read_csv(self.t1_reference_file)
        # check for expected columns
        for c in ["NiCl2 Concentration (mM)", "Temp (C)", "T1 reported (ms)"]:
            assert (c in df.columns), "ReferencePhantomCaibreSystem::load_t1_values(): reference csv is missing " \
                                      "required column '%s' please check file: %s" % (c, self.t1_reference_file)
        # convert and temp columns (i.e. '~20' convert to 20.0)
        if df["Temp (C)"].dtype == object and isinstance(df.iloc[0]["Temp (C)"], str):
            df["Temp (C)"] = df["Temp (C)"].str.replace('~', '')
            df["Temp (C)"] = df["Temp (C)"].astype(float)
        # map the concentrations to the MR-BIAS ROI naming scheme
        # ---------------------------------------------------------
        concentration_list = np.array(list(self.t1_concentration_roi_map.values()))
        inv_map = {v: k for k, v in self.t1_concentration_roi_map.items()}
        df = df.sort_values(['NiCl2 Concentration (mM)', 'Temp (C)'], ascending=[False, True])
        df["roi_label"] = ""
        for roi_concentration in df["NiCl2 Concentration (mM)"].unique().tolist():
            # find the concentration which is closest matching ROI
            nearest_contrn, error_contrn = mu.find_nearest_float(roi_concentration, concentration_list)
            assert error_contrn < self.concentration_error_tol, "ReferencePhantomCaibreSystem::load_t1_values(): " \
                                                                "closest matching T1 roi has a concentration error (%0.5f)" \
                                                                "exceeding tolerance (%0.5f)" %\
                                                                (error_contrn, self.concentration_error_tol)
            # label the dataframe rows with the ROI label
            df.loc[np.isclose(df['NiCl2 Concentration (mM)'], nearest_contrn), 'roi_label'] = inv_map[nearest_contrn]
        # ---------------------------------------------------------
        # iterate over the valid rois and add to class ref_roi dictionary
        for roi_label in df["roi_label"].unique().tolist():
            # find the closest temperature
            df_roi = df[df.roi_label == roi_label]
            temp_vec = np.array(df["Temp (C)"].unique().tolist())
            nearest_temp, error_temp = mu.find_nearest_float(self.temperature, temp_vec)
            if error_temp > self.temp_error_tol:
                mu.log("ReferencePhantomCaibreSystem::load_t1_values() - closest temperature in reference file (%0.2f)"
                       "exceeds the temperature tolerance (%0.2f deg celsuis)" % (nearest_temp, self.temp_error_tol),
                       LogLevels.LOG_WARNING)
            t1_value = df_roi[np.isclose(df_roi["Temp (C)"], nearest_temp)]["T1 reported (ms)"].values[0]
            t1_value_uncertainty = None
            if ("T1 uncertainty (ms)" in df_roi.columns):
                t1_value_uncertainty = df_roi[np.isclose(df_roi["Temp (C)"], nearest_temp)]["T1 uncertainty (ms)"].values[0]
            self.ref_rois[roi_label] = ReferenceROI(roi_label,
                                                    value=t1_value,
                                                    units="ms",
                                                    value_uncertainty=t1_value_uncertainty)

    def load_t2_values(self):
        mu.log("ReferencePhantomCaibreSystem::reading T2 reference file: %s" % self.t2_reference_file, LogLevels.LOG_INFO)
        df = pd.read_csv(self.t2_reference_file)
        # check for expected columns
        for c in ["MnCl2 Concentration (mM)", "Temp (C)", "T2 reported (ms)"]:
            assert (c in df.columns), "ReferencePhantomCaibreSystem::load_t2_values(): reference csv is missing " \
                                      "required column '%s' please check file: %s" % (c, self.t2_reference_file)
        # convert and temp columns (i.e. '~20' convert to 20.0)
        if df["Temp (C)"].dtype == object and isinstance(df.iloc[0]["Temp (C)"], str):
            df["Temp (C)"] = df["Temp (C)"].str.replace('~', '')
            df["Temp (C)"] = df["Temp (C)"].astype(float)
        # map the concentrations to the MR-BIAS ROI naming scheme
        # ---------------------------------------------------------
        concentration_list = np.array(list(self.t2_concentration_roi_map.values()))
        inv_map = {v: k for k, v in self.t2_concentration_roi_map.items()}
        df = df.sort_values(['MnCl2 Concentration (mM)', 'Temp (C)'], ascending=[False, True])
        df["roi_label"] = ""
        for roi_concentration in df["MnCl2 Concentration (mM)"].unique().tolist():
            # find the concentration which is closest matching ROI
            nearest_contrn, error_contrn = mu.find_nearest_float(roi_concentration, concentration_list)
            assert error_contrn < self.concentration_error_tol, "ReferencePhantomCaibreSystem::load_t2_values(): " \
                                                                "closest matching T2 roi has a concentration error (%0.5f)" \
                                                                "exceeding tolerance (%0.5f)" %\
                                                                (error_contrn, self.concentration_error_tol)
            # label the dataframe rows with the ROI label
            df.loc[np.isclose(df['MnCl2 Concentration (mM)'], nearest_contrn), 'roi_label'] = inv_map[nearest_contrn]
        # ---------------------------------------------------------
        # iterate over the valid rois and add to class ref_roi dictionary
        for roi_label in df["roi_label"].unique().tolist():
            # find the closest temperature
            df_roi = df[df.roi_label == roi_label]
            temp_vec = np.array(df["Temp (C)"].unique().tolist())
            nearest_temp, error_temp = mu.find_nearest_float(self.temperature, temp_vec)
            if error_temp > self.temp_error_tol:
                mu.log("ReferencePhantomCaibreSystem::load_t2_values() - closest temperature in reference file (%0.2f)"
                       "exceeds the temperature tolerance (%0.2f deg celsuis)" % (nearest_temp, self.temp_error_tol),
                       LogLevels.LOG_WARNING)
            t2_value = df_roi[np.isclose(df_roi["Temp (C)"], nearest_temp)]["T2 reported (ms)"].values[0]
            t2_value_uncertainty = None
            if ("T2 uncertainty (ms)" in df_roi.columns):
                t2_value_uncertainty = df_roi[np.isclose(df_roi["Temp (C)"], nearest_temp)]["T2 uncertainty (ms)"].values[0]
            self.ref_rois[roi_label] = ReferenceROI(roi_label,
                                                    value=t2_value,
                                                    units="ms",
                                                    value_uncertainty=t2_value_uncertainty)





class ReferencePhantomCalibreSystem1(ReferencePhantomCaibreSystem):
    def __init__(self, field_strength, temperature=None, serial_number=None):
        calibre_sys_phantom_dir = os.path.join(mu.reference_phantom_values_directory(),
                                              "caliber_system_phantom", "batch1_sn_lte_130-0041")
        mu.log("ReferencePhantomCalibreSystem1::__init__() [%0.1f T, %s deg. celsius]" % (field_strength, temperature),
               LogLevels.LOG_INFO)
        t1_reference_file = None
        t2_reference_file = None
        if mu.isclose(field_strength, 1.5, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1-Batch1_1p5T_userCreated_20210726.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2-Batch1_1p5T_userCreated_20210726.csv")
        elif mu.isclose(field_strength, 3.0, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1-Batch1_3T_userCreated_20210726.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2-Batch1_3T_userCreated_20210726.csv")
        super().__init__(PhantomOptions.SYSTEM_PHANTOM_CALIBER_BATCH1, field_strength,
                         t1_reference_file, t2_reference_file,
                         self.get_t1_concentration_roi_map(),
                         self.get_t2_concentration_roi_map(),
                         temperature, serial_number)

    @staticmethod
    def get_t1_concentration_roi_map():
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t1_concentration_roi_map = {"t1_roi_14": 69.68,
                                    "t1_roi_13": 49.122,
                                    "t1_roi_12": 34.59,
                                    "t1_roi_11": 24.326,
                                    "t1_roi_10": 17.07,
                                    "t1_roi_9": 11.936,
                                    "t1_roi_8": 8.297,
                                    "t1_roi_7": 5.731,
                                    "t1_roi_6": 3.912,
                                    "t1_roi_5": 2.617,
                                    "t1_roi_4": 1.72,
                                    "t1_roi_3": 1.072,
                                    "t1_roi_2": 0.623,
                                    "t1_roi_1": 0.299}
        return t1_concentration_roi_map

    @staticmethod
    def get_t2_concentration_roi_map():
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t2_concentration_roi_map = {"t2_roi_14": 1.704,
                                    "t2_roi_13": 1.104,
                                    "t2_roi_12": 0.849,
                                    "t2_roi_11": 0.599,
                                    "t2_roi_10": 0.421,
                                    "t2_roi_9": 0.296,
                                    "t2_roi_8": 0.207,
                                    "t2_roi_7": 0.145,
                                    "t2_roi_6": 0.101,
                                    "t2_roi_5": 0.069,
                                    "t2_roi_4": 0.047,
                                    "t2_roi_3": 0.031,
                                    "t2_roi_2": 0.021,
                                    "t2_roi_1": 0.013}
        return t2_concentration_roi_map


class ReferencePhantomCalibreSystem2(ReferencePhantomCaibreSystem):
    def __init__(self, field_strength, temperature=None, serial_number=None):
        calibre_sys_phantom_dir = os.path.join(mu.reference_phantom_values_directory(),
                                              "caliber_system_phantom", "batch2_sn_gte_130-0042")
        mu.log("ReferencePhantomCalibreSystem2::__init__() [%0.1f T, %s deg. celsius]" % (field_strength, temperature),
               LogLevels.LOG_INFO)
        t1_reference_file = None
        t2_reference_file = None
        if mu.isclose(field_strength, 1.5, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1-Batch2_Extrap-1p5T_userCreated-20210726.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2-Batch2_Extrap-1p5T_dl-20210726.csv")
        elif mu.isclose(field_strength, 3.0, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1-Batch2_3T_dl-20210726.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2-Batch2_3T_dl-20210726.csv")
        # map the concentrations to the MRI-BIAS ROI naming scheme
        super().__init__(PhantomOptions.SYSTEM_PHANTOM_CALIBER_BATCH2, field_strength,
                         t1_reference_file, t2_reference_file,
                         self.get_t1_concentration_roi_map(),
                         self.get_t2_concentration_roi_map(),
                         temperature, serial_number)

    @staticmethod
    def get_t1_concentration_roi_map():
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t1_concentration_roi_map = {"t1_roi_14": 65.3,
                                    "t1_roi_13": 46,
                                    "t1_roi_12": 32.7,
                                    "t1_roi_11": 23.3,
                                    "t1_roi_10": 16.5,
                                    "t1_roi_9": 11.3,
                                    "t1_roi_8": 7.74,
                                    "t1_roi_7": 5.43,
                                    "t1_roi_6": 3.68,
                                    "t1_roi_5": 2.52,
                                    "t1_roi_4": 1.64,
                                    "t1_roi_3": 1.04,
                                    "t1_roi_2": 0.6,
                                    "t1_roi_1": 0.29}
        return t1_concentration_roi_map

    @staticmethod
    def get_t2_concentration_roi_map():
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t2_concentration_roi_map = {"t2_roi_14": 1.5996,
                                    "t2_roi_13": 1.1274,
                                    "t2_roi_12": 0.7902,
                                    "t2_roi_11": 0.5555,
                                    "t2_roi_10": 0.4276,
                                    "t2_roi_9": 0.2768,
                                    "t2_roi_8": 0.193,
                                    "t2_roi_7": 0.1353,
                                    "t2_roi_6": 0.0934,
                                    "t2_roi_5": 0.0626,
                                    "t2_roi_4": 0.0434,
                                    "t2_roi_3": 0.0282,
                                    "t2_roi_2": 0.0181,
                                    "t2_roi_1": 0.0108}
        return t2_concentration_roi_map


class ReferencePhantomCalibreSystem2p5(ReferencePhantomCaibreSystem):
    def __init__(self, field_strength, temperature=None, serial_number=None):
        calibre_sys_phantom_dir = os.path.join(mu.reference_phantom_values_directory(),
                                              "caliber_system_phantom", "batch2p5_sn_gte_130-0133")
        mu.log("ReferencePhantomCalibreSystem2p5::__init__() [%0.1f T, %s deg. celsius]" % (field_strength, temperature),
               LogLevels.LOG_INFO)
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t1_concentration_roi_map = self.get_t1_concentration_roi_map()
        # point it a valid reference file
        t1_reference_file = None
        t2_reference_file = None
        if mu.isclose(field_strength, 1.5, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1-Batch2p5_Extrap-1p5T_userCreated-20210726.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2-Batch2p5_Extrap-1p5T_dl-20210726.csv")
        elif mu.isclose(field_strength, 3.0, abs_tol=0.01):
            # note no T1 3.0T values given for batch 2.5 (using batch 2.0 values)
            mu.log("ReferencePhantomCalibreSystem2p5::__init__(): "
                   "no T1 3.0T values given for batch 2.5 (using batch 2.0 values)", LogLevels.LOG_WARNING)
            t1_concentration_roi_map = ReferencePhantomCalibreSystem2.get_t1_concentration_roi_map()
            calibre_sys_phantom_batch_2_dir = os.path.join(mu.reference_phantom_values_directory(),
                                                           "caliber_system_phantom", "batch2_sn_gte_130-0042")
            t1_reference_file = os.path.join(calibre_sys_phantom_batch_2_dir, "T1-Batch2_3T_dl-20210726.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2-Batch2p5_3T_dl-20210726.csv")

        super().__init__(PhantomOptions.SYSTEM_PHANTOM_CALIBER_BATCH2p5, field_strength,
                         t1_reference_file, t2_reference_file,
                         t1_concentration_roi_map,
                         self.get_t2_concentration_roi_map(),
                         temperature, serial_number)

    @staticmethod
    def get_t1_concentration_roi_map():
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t1_concentration_roi_map = {"t1_roi_14": 65.3,
                                    "t1_roi_13": 46,
                                    "t1_roi_12": 32.7,
                                    "t1_roi_11": 23.3,
                                    "t1_roi_10": 16.5,
                                    "t1_roi_9": 11.3,
                                    "t1_roi_8": 7.74,
                                    "t1_roi_7": 5.43,
                                    "t1_roi_6": 3.68,
                                    "t1_roi_5": 2.52,
                                    "t1_roi_4": 1.64,
                                    "t1_roi_3": 1.04,
                                    "t1_roi_2": 0.6,
                                    "t1_roi_1": 0.29}
        return t1_concentration_roi_map

    @staticmethod
    def get_t2_concentration_roi_map():
        # map the concentrations to the MRI-BIAS ROI naming scheme
        t2_concentration_roi_map = {"t2_roi_14": 1.5996,
                                    "t2_roi_13": 1.1274,
                                    "t2_roi_12": 0.7902,
                                    "t2_roi_11": 0.5555,
                                    "t2_roi_10": 0.4276,
                                    "t2_roi_9": 0.2768,
                                    "t2_roi_8": 0.193,
                                    "t2_roi_7": 0.1353,
                                    "t2_roi_6": 0.0934,
                                    "t2_roi_5": 0.0673,
                                    "t2_roi_4": 0.0434,
                                    "t2_roi_3": 0.0282,
                                    "t2_roi_2": 0.0181,
                                    "t2_roi_1": 0.0113}
        return t2_concentration_roi_map


class ReferencePhantomCalibreSystemFitInit(ReferencePhantomCaibreSystem):
    def __init__(self, field_strength, temperature=None):
        calibre_sys_phantom_dir = os.path.join(mu.reference_phantom_values_directory(),
                                              "caliber_system_phantom")
        mu.log("ReferencePhantomCalibreSystemFitInit::__init__() [%0.1f T, %s deg. celsius]" % (field_strength, temperature),
               LogLevels.LOG_INFO)
        t1_reference_file = None
        t2_reference_file = None
        if mu.isclose(field_strength, 1.5, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1_1p5T_curve_fit_init.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2_1p5T_curve_fit_init.csv")
        elif mu.isclose(field_strength, 3.0, abs_tol=0.01):
            t1_reference_file = os.path.join(calibre_sys_phantom_dir, "T1_3T_curve_fit_init.csv")
            t2_reference_file = os.path.join(calibre_sys_phantom_dir, "T2_3T_curve_fit_init.csv")
        super().__init__(PhantomOptions.SYSTEM_PHANTOM_CALIBER_BATCH1, field_strength,
                         t1_reference_file, t2_reference_file,
                         self.get_t1_concentration_roi_map(),
                         self.get_t2_concentration_roi_map(),
                         temperature)
    @staticmethod
    def get_t1_concentration_roi_map():
        return ReferencePhantomCalibreSystem1.get_t1_concentration_roi_map()
    @staticmethod
    def get_t2_concentration_roi_map():
        return ReferencePhantomCalibreSystem1.get_t2_concentration_roi_map()



if __name__ == "__main__":
    main()