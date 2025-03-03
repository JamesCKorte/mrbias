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
02-August-2021  :               (James Korte) : Initial code for      MR-BIAS v0.0.0
  23-June-2022  :               (James Korte) : GitHub Release        MR-BIAS v1.0.0
   16-Jan-2023  :               (James Korte) : Goodness of fit added MR-BIAS v1.0.1
   10-Feb-2024  :               (James Korte) : Increased total number of ROIs
"""

import re
import os
from collections import OrderedDict
from enum import IntEnum

import tempfile
import shutil
import copy

import pydicom as dcm
import pandas as pd

# to handle relative paths (for calls inside or outside package)
from pathlib import Path

import SimpleITK as sitk
import numpy as np

import matplotlib as mpl

import datetime

from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib.units import inch, cm
import PIL
import io


MRBIAS_VERSION_NUMBER = "1.1.0"
MRBIAS_VERSION_DATE   = "6th December 2024"
MRBIAS_URL            = "http://github.com/JamesCKorte/mrbias"
MRBIAS_RELAXOMETRY_DOI_URL = "https://doi.org/10.1088/1361-6560/acbcbb"
MRBIAS_DIFFUSION_DOI_URL = MRBIAS_RELAXOMETRY_DOI_URL #"https://doi.org/TBD"  # just point it at the original publication until DOI available

MRBIAS_REPORT_IMAGE_DPI = 500

# Want a unique ROI label (across T1/T2/PD to allow combined/mixed ROI type dataframes)
# The Label->IDX maps are then for each specific ROI type (T1/T2/PD) and also map to the input yaml files
ROI_IDX_LABEL_MAP = OrderedDict()
T1_ROI_LABEL_IDX_MAP = OrderedDict()
for i in range(18):
    ROI_IDX_LABEL_MAP[i+1] = "t1_roi_%d" % (i+1)
    T1_ROI_LABEL_IDX_MAP["t1_roi_%d" % (i+1)] = i+1
T2_ROI_LABEL_IDX_MAP = OrderedDict()
for i in range(18):
    ROI_IDX_LABEL_MAP[i+18+1] = "t2_roi_%d" % (i+1)
    T2_ROI_LABEL_IDX_MAP["t2_roi_%d" % (i+1)] = i+18+1

DW_ROI_LABEL_IDX_MAP = OrderedDict()
for i in range(18):
    ROI_IDX_LABEL_MAP[i+36+1] = "dw_roi_%d" % (i+1)
    DW_ROI_LABEL_IDX_MAP["dw_roi_%d" % (i+1)] = i+36+1
# create a reverse lookup
ROI_LABEL_IDX_MAP = {v: k for k, v in ROI_IDX_LABEL_MAP.items()}

class OrientationOptions(IntEnum):
    AXI = 1
    COR = 2
    SAG = 3

class PhantomOptions(IntEnum):
    RELAX_SYSTEM = 1
    RELAX_EUROSPIN = 2
    DIFFUSION_NIST = 3

SHARED_COLORMAP_SETTING = 'colorblind_rainbow'  # 'nipy_spectral'

# Colorblind rainbow color map taken from:
# https://personal.sron.nl/~pault/#sec:qualitative
def get_cb_rainbow_map():
    col_arr = [#np.array([232., 236., 251.])/255.,
               #np.array([221., 216., 239.])/255.,
               #np.array([209., 193., 225.])/255.,
               np.array([195., 168., 209.])/255.,
               np.array([181., 143., 194.])/255.,
               np.array([167., 120., 180.])/255.,
               np.array([155.,  98., 167.])/255.,
               np.array([140.,  78., 153.])/255.,
               np.array([111.,  76., 155.])/255.,
               np.array([ 96.,  89., 169.])/255.,
               np.array([ 85., 104., 184.])/255.,
               np.array([ 78., 121., 197.])/255.,
               np.array([ 77., 138., 198.])/255.,
               np.array([ 78., 150., 188.])/255.,
               np.array([ 84., 158., 179.])/255.,
               np.array([ 89., 165., 169.])/255.,
               np.array([ 96., 171., 158.])/255.,
               np.array([105., 177., 144.])/255.,
               np.array([119., 183., 125.])/255.,
               np.array([140., 188., 104.])/255.,
               np.array([166., 190.,  84.])/255.,
               np.array([190., 188.,  72.])/255.,
               np.array([209., 181.,  65.])/255.,
               np.array([221., 170.,  60.])/255.,
               np.array([228., 156.,  57.])/255.,
               np.array([231., 140.,  53.])/255.,
               np.array([230., 121.,  50.])/255.,
               np.array([228.,  99.,  45.])/255.,
               np.array([223.,  72.,  40.])/255.,
               np.array([218.,  34.,  34.])/255.,
               np.array([184.,  34.,  30.])/255.,
               np.array([149.,  33.,  27.])/255.,
               np.array([114.,  30.,  23.])/255.]
    return mpl.colors.ListedColormap(col_arr)


class ColourSettings(object):
    def __init__(self,
                 roi_index_list=None,
                 matplotlib_cmap_name=SHARED_COLORMAP_SETTING):
        self.cmap_name = matplotlib_cmap_name
        # default to a color range which covers all ROI indexes
        self.t1_label_idx_map = T1_ROI_LABEL_IDX_MAP
        self.t2_label_idx_map = T2_ROI_LABEL_IDX_MAP
        self.dw_label_idx_map = DW_ROI_LABEL_IDX_MAP
        # if the ROI index range is provided then adjust then correct the idx map
        if roi_index_list is not None:
            # make a new label -> idx map
            new_label_idx_map = OrderedDict()
            for roi_dx in roi_index_list:
                new_label_idx_map[ROI_IDX_LABEL_MAP[roi_dx]] = roi_dx
            # assign it to the appropriate class variable
            roi_label = ROI_IDX_LABEL_MAP[roi_index_list[0]]
            if roi_label in self.t1_label_idx_map.keys():
                self.t1_label_idx_map = new_label_idx_map
            elif roi_label in self.t2_label_idx_map.keys():
                self.t2_label_idx_map = new_label_idx_map
            elif roi_label in self.dw_label_idx_map.keys():
                self.dw_label_idx_map = new_label_idx_map

        if matplotlib_cmap_name == 'colorblind_rainbow':
            self.cmap = get_cb_rainbow_map()
            self.cmap_alpha = 0.9
        else:
            self.cmap = mpl.cm.get_cmap(self.cmap_name)
            self.cmap_alpha = 0.7
        t1_roi_dx_list = list(self.t1_label_idx_map.values())
        self.norm_t1 = mpl.colors.Normalize(vmin=np.min(t1_roi_dx_list)-1, vmax=np.max(t1_roi_dx_list)+1)
        t2_roi_dx_list = list(self.t2_label_idx_map.values())
        self.norm_t2 = mpl.colors.Normalize(vmin=np.min(t2_roi_dx_list)-1, vmax=np.max(t2_roi_dx_list)+1)
        dw_roi_dx_list = list(self.dw_label_idx_map.values())
        self.norm_dw = mpl.colors.Normalize(vmin=np.min(dw_roi_dx_list)-1, vmax=np.max(dw_roi_dx_list)+1)

    def get_ROI_colour(self, roi_label):
        if roi_label in self.t1_label_idx_map.keys():
            return self.cmap(self.norm_t1(self.t1_label_idx_map[roi_label]))
        elif roi_label in self.t2_label_idx_map.keys():
            return self.cmap(self.norm_t2(self.t2_label_idx_map[roi_label]))
        elif roi_label in self.dw_label_idx_map.keys():
            return self.cmap(self.norm_dw(self.dw_label_idx_map[roi_label]))
        else:
            log("ColourSettings::get_ROI_colour(): label %s not found returning default black" % roi_label,
                LogLevels.LOG_WARNING)
            return (0.0, 0.0, 0.0)

    def get_cmap(self):
        return self.cmap

    def get_cmap_alpha(self):
        return self.cmap_alpha

    # bit of a hacky way to determine type, but works within the existing visualisaiton framework
    def get_vmin(self, global_roi_indexes):
        roi_label = ROI_IDX_LABEL_MAP[global_roi_indexes[0]]
        if roi_label in self.t1_label_idx_map.keys():
            return np.min(list(self.t1_label_idx_map.values())) - 1
        elif roi_label in self.t2_label_idx_map.keys():
            return np.min(list(self.t2_label_idx_map.values())) - 1
        elif roi_label in self.dw_label_idx_map.keys():
            return np.min(list(self.dw_label_idx_map.values())) - 1
        else:
            return np.min(global_roi_indexes) - 1
    def get_vmax(self, global_roi_indexes):
        roi_label = ROI_IDX_LABEL_MAP[global_roi_indexes[0]]
        if roi_label in self.t1_label_idx_map.keys():
            return np.max(list(self.t1_label_idx_map.values())) + 1
        elif roi_label in self.t2_label_idx_map.keys():
            return np.max(list(self.t2_label_idx_map.values())) + 1
        elif roi_label in self.dw_label_idx_map.keys():
            return np.max(list(self.dw_label_idx_map.values())) + 1
        else:
            return np.max(global_roi_indexes) + 1

SEABORN_STYLE = "whitegrid"

class PDFSettings(object):
    def __init__(self):
        self.page_size = A3
        self.page_width, self.page_height = landscape(self.page_size)
        self.top_margin = 0.3 * inch
        self.left_margin = 0.3 * inch
        self.line_width = 0.2 * inch
        self.small_line_width = 0.15 * inch
        self.font_name = 'Courier' # set to a fixed width font
        self.font_size = 14
        self.small_font_size = 11


BLUE = np.array([29, 144, 170])/255.0
GREEN = np.array([140, 187, 78])/255.0
PURPLE = np.array([121, 88, 146])/255.0
RED = np.array([196, 68, 64])/255.0
ORANGE = np.array([243, 133, 51])/255.0
GREY = np.array([145, 145, 145])/255.0
BLACK = np.array([0.0, 0.0, 0.0])
WHITE = np.array([1.0, 1.0, 1.0])

key_fmt = "%s_%03d"

class LogLevels(IntEnum):
    LOG_ERROR = 1
    LOG_WARNING = 2
    LOG_INFO = 3
LOG_LEVEL_LIMIT = LogLevels.LOG_INFO
LOG_LEVEL_STR_DICT = {LogLevels.LOG_ERROR : "ERROR",
                      LogLevels.LOG_WARNING : "WARNING",
                      LogLevels.LOG_INFO : "INFO"}

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(object):#(metaclass=Singleton): # todo: change back to Singleton and create a re-routing function (call from initialise_logger)
    def __init__(self, filename=None, force_overwrite=False, write_to_screen=False):
        self.filename = filename
        self.file = None
        self.write_to_file = False
        self.write_to_screen = write_to_screen
        # set it all up
        if self.filename is not None:
            # check its not an existing file
            if os.path.isfile(self.filename) and (not force_overwrite):
                assert False, "misc_utils::Logger::__init__(): log file already exists will not overwrite : %s" % self.filename
            else:
                self.file = open(self.filename, "w")
                self.write_to_file = True

    def write(self, msg):
        # write to file if log file available
        if self.write_to_file:
            if self.file is not None:
                timestamp = datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")
                self.file.write("%s %s\n" % (timestamp, msg))
        # write to screen as well if requested
        if self.write_to_screen:
            print(msg)

    def __del__(self):
        if self.file is not None:
            self.file.close()

logger = None
def initialise_logger(filename=None, force_overwrite=False, write_to_screen=False):
    global logger
    logger = Logger(filename, force_overwrite, write_to_screen)
def detatch_logger():
    global logger
    logger = None

def log(str, log_level, log_limit=LOG_LEVEL_LIMIT):
    if log_level <= log_limit:
        msg = "[%s] %s" % (LOG_LEVEL_STR_DICT[log_level], str)
        if logger is None:
            print(msg)
        else:
            logger.write(msg)

# log a list of log entries (removing duplicates to minimise output to log)
def log_buffer(str_list, truncation_length, log_level):
    df = pd.DataFrame(str_list, columns=["msg"])
    df["msg_trunc"] = df["msg"].str.slice(0, truncation_length)
    msgs_unique = df["msg_trunc"].unique()
    for msg_unique in msgs_unique:
        df_unique = df[df["msg_trunc"].str.match(re.escape(msg_unique))]
        msg_long = df_unique.msg.iloc[0]
        msg_count = df_unique.shape[0]
        if msg_count > 1:
            log("%s [repeated %d times]" % (msg_long, msg_count), log_level)
        else:
            log(msg_long, log_level)



BASE_PATH = Path(__file__).parent
# should now work within module and outside module
def reference_data_directory():
    return (BASE_PATH / os.path.join("..", "data")).resolve()
def reference_template_directory():
    return (BASE_PATH / "roi_detection_templates").resolve()
def reference_phantom_values_directory():
    return (BASE_PATH / "roi_reference_values").resolve()



def natural_sort(ll):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(ll, key=alphanum_key)




# load up a SimpleITK image from a file list
# -------------------------------------------------
# also has some additional checks such as
# - correcting philips scaling from private dicom tags
#
# the "check_image_scaling" flag can be used to test the assumptions
# which the image scaling are based on
def load_image_from_filelist(files_sorted, series_uid,
                             rescale_slope, rescale_intercept,
                             scale_slope, scale_intercept,
                             series_descrp=None, check_image_scaling=False, philips_scaling=False):
    # Create a temp directory and symlink all of the files,
    # then read from there.
    im = None
    with tempfile.TemporaryDirectory() as tmpdir_name:
        # Try to create symbolic links to the original files from our tmpdir
        try:
            for f in files_sorted:
                os.symlink(os.path.abspath(f), os.path.join(tmpdir_name, os.path.basename(f)))
        except:
            # if it fails (permissions etc.)
            # copy the original files to a tmpdir
            for f in files_sorted:
                shutil.copy(os.path.abspath(f), os.path.join(tmpdir_name, os.path.basename(f)))
        # Now get the sorted list of file names
        reader = sitk.ImageSeriesReader()
        sitk_sorted_filenames = reader.GetGDCMSeriesFileNames(tmpdir_name, series_uid)
        reader.SetFileNames(sitk_sorted_filenames)
        # reader.MetaDataDictionaryArrayUpdateOn()  # not included as they do no impact image scaling ...
        # reader.LoadPrivateTagsOn()                # and they also increase read time
        im = reader.Execute()
        # undo the scaling to get the raw values
        assert isinstance(rescale_slope, float) and isinstance(rescale_intercept, float), \
            "Scale[%s]/Intercept[%s] not floats" % (str(rescale_slope), str(rescale_intercept))
        if np.isnan(rescale_slope) or np.isnan(rescale_intercept) or \
                (rescale_slope is None) or (rescale_intercept is None):
            log("ScanSession::_load_image_from_filelist(%s): Scale[%s]/Intercept[%s] are invalid"
                " setting them to [m=1.0, c=0.0]" % (series_descrp, str(rescale_slope), str(rescale_intercept)),
                LogLevels.LOG_WARNING)
            rescale_slope = 1.0
            rescale_intercept = 0.0

        # Check that image scaling assumptions are correct
        # 1. That SimpleITK loads raw integer values and then scales with
        #    PublicTags only (RescaleSlope, RescaleIntercept)
        # 2. That SimpleITK does not use PrivateTags such as those
        #    used in Philips dicom (ScaleSlope, ScaleIntercept)
        if check_image_scaling:
            im_sitk_ar = sitk.GetArrayFromImage(im)
            im_sitk_raw_ar = (im_sitk_ar - rescale_intercept)/rescale_slope # rescale back to raw with public tags
            # LOAD UP MANUALLY EACH SLICE AND SEE VALUES
            im_raw_min, im_raw_max = None, None
            for f_dx, fname in enumerate(sitk_sorted_filenames):
                ds = dcm.read_file(fname)
                im_arr = ds.pixel_array
                im_min, im_max = np.min(im_arr), np.max(im_arr)
                if f_dx == 0:
                    im_raw_min, im_raw_max = im_min, im_max
                else:
                    if im_min < im_raw_min:
                        im_raw_min = im_min
                    if im_max > im_raw_max:
                        im_raw_max = im_max
            im_public_min = rescale_slope*im_raw_min + rescale_intercept    # rescale to floating point values with public tags
            im_public_max = rescale_slope*im_raw_max + rescale_intercept    # rescale to floating point values with public tags

            # output
            log("\tpydcm: Scaled with PublicTags (min, max)  -> %0.4f, %0.4f" % (im_public_min, im_public_max), LogLevels.LOG_INFO)
            log("\tSimpleITK loaded (min, max)               -> %0.4f, %0.4f" % (np.min(im_sitk_ar), np.max(im_sitk_ar)), LogLevels.LOG_INFO)
            log("\tpydcm: raw (min, max)                     -> %0.4f, %0.4f" % (im_raw_min, im_raw_max), LogLevels.LOG_INFO)
            log("\tSimpleITK rescaled back to raw (min, max) -> %0.4f, %0.4f" % (np.min(im_sitk_raw_ar), np.max(im_sitk_raw_ar)), LogLevels.LOG_INFO)
            assert np.isclose(np.min(im_sitk_ar), im_public_min) and np.isclose(np.max(im_sitk_ar), im_public_max), \
                "ScanSession::_load_image_from_filelist(): image scaling check failed [sitk image range != pydcm raw values rescaled with public tags]"
            assert np.isclose(np.min(im_sitk_raw_ar), im_raw_min) and np.isclose(np.max(im_sitk_raw_ar), im_raw_max), \
                "ScanSession::_load_image_from_filelist(): image scaling check failed [sitk image range converted to raw with public tags != pydcm raw values]"

        if philips_scaling:
            im = _rescale_image_to_philips(im, rescale_slope, rescale_intercept, scale_slope, scale_intercept, series_descrp)

    return im, rescale_slope, rescale_intercept, scale_slope, scale_intercept

def parse_dicom_dir_for_info(dcm_dir):
    dicom_searcher = DICOMSearch(dcm_dir)
    df = dicom_searcher.get_df()
    series_uid = df.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID.iloc[0]
    assert df.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID.shape[0] == 1, "MiscUtils::parse_dicom_dir_for_info(): function can only handle one image series in the dicom directory"
    scanner_make = df.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
    bits_allocated = df.drop_duplicates(subset=["BitsAllocated"]).BitsAllocated.iloc[0]
    bits_stored = df.drop_duplicates(subset=["BitsStored"]).BitsStored.iloc[0]
    rescale_slope = df.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.iloc[0]
    rescale_intercept = df.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.iloc[0]
    assert df.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.shape[0] == 1, \
        "MiscUtils::parse_dicom_dir_for_info(): multiple RescaleIntercept values (i.e. per slice) for a single series not supported by MRBIAS"
    assert df.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.shape[0] == 1, \
        "MiscUtils::parse_dicom_dir_for_info(): multiple RescaleSlope values (i.e. per slice) for a single series not supported by MRBIAS"
    # get all the slices
    files_sorting = []
    for index, row in df.iterrows():
        files_sorting.append((row["ImageFilePath"], row["SliceLocation"]))
    # sort the slices by slice location before reading by sitk
    files_sorting.sort(key=lambda x: x[1])
    files_sorted = [x[0] for x in files_sorting]
    # handle manufacturor specific images
    scale_slope = None
    scale_intercept = None
    if scanner_make == "Philips":
        scale_slope = df.drop_duplicates(subset=["ScaleSlope"]).ScaleSlope.iloc[0]
        scale_intercept = df.drop_duplicates(subset=["ScaleIntercept"]).ScaleIntercept.iloc[0]
        assert df.drop_duplicates(subset=["ScaleIntercept"]).ScaleIntercept.shape[
                   0] == 1, "MiscUtils::parse_dicom_dir_for_info(): multiple ScaleIntercept values (i.e. per slice) for a single series not supported by MRBIAS"
        assert df.drop_duplicates(subset=["ScaleSlope"]).ScaleSlope.shape[
                   0] == 1, "MiscUtils::parse_dicom_dir_for_info(): multiple ScaleSlope values (i.e. per slice) for a single series not supported by MRBIAS"
    return files_sorted, series_uid, rescale_slope, rescale_intercept, scale_slope, scale_intercept, (scanner_make == "Philips")


def _rescale_image_to_raw(im, rescale_slope, rescale_intercept, series_decrip=None):
    if not (np.isclose(rescale_slope, 1.0) and np.isclose(rescale_intercept, 0.0)):
        log("ScanSession::_get_imageset_data_from_df(%s): rescaling image back to raw data" %
            series_decrip, LogLevels.LOG_INFO)
        im_arr = sitk.GetArrayFromImage(im)
        im_arr = (im_arr - rescale_intercept) / rescale_slope
        im_arr = im_arr.astype(np.uint16)
        im_raw = sitk.GetImageFromArray(im_arr)
        im_raw.SetOrigin(im.GetOrigin())
        im_raw.SetSpacing(im.GetSpacing())
        im_raw.SetDirection(im.GetDirection())
        return im_raw
    return im

# Chenevert et al. 2014 for details on Philips DICOM scaling
# DOI 10.1593/tlo.13811
def _rescale_image_to_philips(im, rescale_slope, rescale_intercept, scale_slope, scale_intercept, series_decrip=None):
    log("ScanSession::_rescale_philips_image(%s): correcting image scaling" %
        series_decrip, LogLevels.LOG_INFO)
    im_arr = sitk.GetArrayFromImage(im)
    im_arr = im_arr / (rescale_slope * scale_slope)
    im_phil = sitk.GetImageFromArray(im_arr)
    im_phil.SetOrigin(im.GetOrigin())
    im_phil.SetSpacing(im.GetSpacing())
    im_phil.SetDirection(im.GetDirection())
    return im_phil


# Scan a dicom directory and return a simpleITK image
def get_sitk_image_from_dicom_image_folder(dcm_dir, if_multiple_take_first=False):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
    assert len(series_ids) > 0, "mu::get_sitk_image_from_dicom_image_folder() : no dicom series found in %s" % dcm_dir
    if len(series_ids) > 1 and (not if_multiple_take_first):
        assert False, "mu::get_sitk_image_from_dicom_image_folder() : not able to handle multiple DCM series in %s" % dcm_dir
    series_id = series_ids[0]
    dcm_filenames = reader.GetGDCMSeriesFileNames(dcm_dir, series_id)
    assert len(dcm_filenames) > 0, "mu::get_sitk_image_from_dicom_image_folder() : no dicom files found in %s" % dcm_dir
    reader.SetFileNames(dcm_filenames)
    return reader.Execute()

def safe_dir_create(d, log_prefix=""):
    if os.path.isdir(d):
        log("%s folder already exists: %s" % (log_prefix, d), LogLevels.LOG_WARNING)
        return False
    else:
        os.mkdir(d)
    return True


# converserion of matplotlib figure to a PIL for output by PDFReport
def mplcanvas_to_pil(f):
    buf = io.BytesIO()
    f.savefig(buf, format='png', dpi=MRBIAS_REPORT_IMAGE_DPI)
    buf.seek(0)
    im = PIL.Image.open(buf)
    return im

def isclose(a, b, abs_tol):
    return np.isclose(a, b, atol=abs_tol)

def find_nearest_float(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], abs(array[idx]-value)

def sitk_deepcopy(im):
    im_arr = sitk.GetArrayFromImage(im)
    new_im = sitk.GetImageFromArray(copy.deepcopy(im_arr))
    new_im.SetOrigin(im.GetOrigin())
    new_im.SetSpacing(im.GetSpacing())
    new_im.SetDirection(im.GetDirection())
    return new_im

def make_sitk_zeros_image_like(im):
    im_arr = sitk.GetArrayFromImage(im)
    # create new one
    zero_im_arr = np.zeros_like(im_arr, dtype=im_arr.dtype)
    # Convert the zero array to a simpleITK image with matched spatial properties to the
    # initial  image
    zero_image = sitk.GetImageFromArray(zero_im_arr)
    zero_image.SetOrigin(im.GetOrigin())
    zero_image.SetDirection(im.GetDirection())
    zero_image.SetSpacing(im.GetSpacing())
    return zero_image

def get_homog_matrix_from_transform(sitk_tx):
    tx = sitk.Euler3DTransform(sitk_tx)
    tx_matrix = np.zeros([4, 4], dtype=float)
    tx_matrix[0:3, 0:3] = np.array([tx.GetMatrix()[0:3],
                                    tx.GetMatrix()[3:6],
                                    tx.GetMatrix()[6:9]])  # rotation matrix
    tx_matrix[3, 3] = 1.0
    tx_matrix[0:3, 3] = np.array(tx.GetTranslation()) # translation
    return tx_matrix

# pull out the relevant slice based on orientation
def get_slice_and_extent(im, orient, dx=None):
    im_slice, extent = None, [0, 0, 0, 0]
    im_arr = sitk.GetArrayFromImage(im)
    im_spacing = np.flip(np.array(im.GetSpacing()))
    if orient == OrientationOptions.AXI:
        if dx is None:
            dx = int(im_arr.shape[0]/2)
        im_slice = im_arr[dx, :, :]
        extent = [0, im_arr.shape[2] * im_spacing[2], 0, im_arr.shape[1] * im_spacing[1]]
    elif orient == OrientationOptions.COR:
        if dx is None:
            dx = int(im_arr.shape[1]/2)
        im_slice = im_arr[:, dx, :]
        extent = [0, im_arr.shape[2] * im_spacing[2], 0, im_arr.shape[0] * im_spacing[0]]
    elif orient == OrientationOptions.SAG:
        if dx is None:
            dx = int(im_arr.shape[2]/2)
        im_slice = im_arr[:, :, dx]
        extent = [0, im_arr.shape[1] * im_spacing[1], 0, im_arr.shape[0] * im_spacing[0]]
    assert im_slice is not None, "get_slice_and_extent() : " \
                                 "expected slice_orient parameter of AXI, COR, or SAG [not %s]" % slice_orient
    return im_slice, extent



# Which tags are important?
# ============================================
# ScanningSequence : is used to determine if a spin-echo or a gradient echo sequence
# MRAcquisitionType : is used to determine 2D/3D
class DICOMSearch(object):
    def __init__(self, target_dir, read_single_file=False, output_csv_filename=None):
        log("DICOMSearch::init(): searching target DCM dir: %s" % target_dir,
            LogLevels.LOG_INFO)
        private_tags = [dcm.tag.Tag(0x2001, 0x1020), dcm.tag.Tag(0x2005, 0x100E), dcm.tag.Tag(0x2005, 0x100D), dcm.tag.Tag(0x0019, 0x100C), dcm.tag.Tag(0x0043, 0x1030)]
        filepaths = []
        for path, dirs, files in os.walk(target_dir):
            for file in files:
                filepaths.append(os.path.join(path, file))
        # Checking all the SeriesInstanceUIDs in the DICOM files found
        dicom_files = []
        for filepath in filepaths:
            try:
                ds = dcm.dcmread(filepath)
                dicom_files.append((ds, ds.SeriesInstanceUID, filepath))
                if read_single_file:
                    break
            except dcm.errors.InvalidDicomError as e:
                pass#    print(e, filepath)
            except AttributeError as e:
                pass#    print(e, filepath)
        # Creating dictionary with UID as the key, and filepath list as the value
        dicom_dict = {}
        for ds, series_inst_uid, fpath in dicom_files:
            # check its an image
            if "ImageType" in ds.dir():
                if series_inst_uid in dicom_dict:
                    dicom_dict[series_inst_uid].append((ds, fpath))
                else:
                    dicom_dict[series_inst_uid] = [(ds, fpath)]
            else:
                log("'DICOMSearch::__init()__: skipping non-image file :%s" % fpath, LogLevels.LOG_WARNING)
        log('DICOMSearch::init():  search complete!  %d image sets with Unique IDs found' %  len(dicom_dict.keys()),
            LogLevels.LOG_INFO)
        # create a pandas dataframe from selected DICOM metadata
        dicom_data = []
        column_meta_names = ['ImageFilePath', 'ImageType', 'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
                             'StudyDate', 'StudyTime', 'StudyDescription', 'StudyInstanceUID',
                             'SOPInstanceUID',
                             'InstitutionName', 'InstitutionAddress', 'InstitutionalDepartmentName',
                             'Modality', 'Manufacturer', 'ManufacturerModelName', 'DeviceSerialNumber',
                             'SeriesDate', 'SeriesTime', 'SeriesDescription', 'ProtocolName',
                             'SeriesInstanceUID', 'SeriesNumber', 'AcquisitionDate', 'AcquisitionTime',
                             'BitsAllocated', 'BitsStored', 'ScanningSequence', 'ScanOptions',
                             'RescaleSlope', 'RescaleIntercept',
                             'ScaleSlope', 'ScaleIntercept', # Phillips Specific
                             'SequenceVariant', 'MRAcquisitionType',
                             'SliceThickness', 'PixelSpacing', 'Rows', 'Columns',
                             'FlipAngle', 'EchoTime', 'EchoNumbers', 'RepetitionTime', 'PixelBandwidth',
                             'NumberOfPhaseEncodingSteps', 'PercentSampling', 'SliceLocation',
                             "SequenceName", "MagneticFieldStrength", "InversionTime",
                             "DiffusionBValue", "DiffusionGradientOrientation", "StackID"]
        alternatives_dict = {"SequenceName" : [dcm.tag.Tag(0x2001, 0x1020), "PulseSequenceName"],
                             "ScaleSlope" : [dcm.tag.Tag(0x2005, 0x100E)],
                             "ScaleIntercept" : [dcm.tag.Tag(0x2005, 0x100D)],
                             "DiffusionBValue" : [dcm.tag.Tag(0x0019, 0x100C)],
                             "DiffusionGradientOrientation": [dcm.tag.Tag(0x0043, 0x1030)]}
                             # "ScanningSequence": ["EchoPulseSequence"], #[EchoPulseSequence"],
                             # "MRAcquisitionType": ["VolumetricProperties"],
                             # "AcquisitionDate": ["InstanceCreationDate", "ContentDate"],
                             # "AcquisitionTime": ["InstanceCreationTime", "ContentTime"]}
        for UID, ds_filepaths in dicom_dict.items():
            log('\tParsing DCM file (%s) / SeriesInstanceUID: %s' % (ds_filepaths[0][0].SeriesDescription, UID),
                LogLevels.LOG_INFO)
            log_info_vec = []
            log_warning_vec = []
            for ds, filepath in ds_filepaths:
                data_row = [filepath]
                available_tags = ds.dir()
                for tag_name in column_meta_names[1:]: # skip the "ImageFilePath"
                    if tag_name in available_tags:
                        data_row.append(ds[tag_name].value)
                    else:
                        # search for alternatives
                        alt_found = False
                        if tag_name in alternatives_dict.keys():
                            for alt_tag_name in alternatives_dict[tag_name]:
                                # add in any private tags
                                tag_added = False
                                if alt_tag_name in private_tags:
                                    if alt_tag_name in ds.keys():
                                        available_tags.append(alt_tag_name)
                                        tag_added = True
                                if alt_tag_name in available_tags:
                                    #log("DICOMSearch::__init__(): missing dicom tag (%s) in file (%s)" %
                                    #    (tag_name, filepath), LogLevels.LOG_WARNING)
                                    log_info_vec.append("\t\tDICOMSearch::__init__(): missing dicom tag (%s) ... using alternative tag (%s) with value (%s)" %
                                                        (tag_name, alt_tag_name, ds[alt_tag_name].value))
                                    data_row.append(ds[alt_tag_name].value)
                                    alt_found = True
                                    if tag_added:
                                        available_tags.pop()  # remove any recently added private tags
                                    break
                                if tag_added:
                                    available_tags.pop() # remove any recently added private tags
                        if not alt_found:
                            data_row.append(None)  # if doesn't exist so data field is left blank
                            if not (tag_name in ["InversionTime", "RescaleSlope", "RescaleIntercept", "PulseSequenceName", "SequenceName", "DiffusionBValue", "ScaleSlope", "ScaleIntercept"]):
                                log_warning_vec.append("\t\tDICOMSearch::__init__(): unable to locate dicom tag (%s) in file (%s)" %
                                                       (tag_name, filepath))
                                #     for d in available_tags:
                                #         print("----> ", d, "  : ",  ds[d])
                                #     assert False
                # append the row and move onto the next
                dicom_data.append(data_row)
            # log any warnings or info for this image
            log_buffer(log_info_vec, 95, LogLevels.LOG_INFO)
            log_buffer(log_warning_vec, 60, LogLevels.LOG_WARNING)
        # Creating the DICOM Dataframe
        df = pd.DataFrame(dicom_data,
                          columns=column_meta_names)

        # todo: remove this hack to make datafrome uniform by saving to disk (un-necessary disk write)
        #     : this converts columns that include lists get converted to strings etc.
        #     : the original abstract_scan object took a pandas CSV as input
        temp_filename = "temp.csv"
        if output_csv_filename is not None:
            temp_filename = output_csv_filename
        df.to_csv(temp_filename)
        self.df = pd.read_csv(temp_filename)
        # if not output file is specified then remove the temporary file
        if output_csv_filename is None:
            os.remove(temp_filename)

    def get_df(self):
        return self.df

    def save_df(self, df_filename):
        log('DICOMSearch::save_df() to %s' % df_filename, LogLevels.LOG_INFO)
        self.df.to_csv(df_filename)