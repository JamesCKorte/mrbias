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
"""

import re
import os
from collections import OrderedDict
from enum import IntEnum

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


MRBIAS_VERSION_NUMBER = "1.0.1"
MRBIAS_VERSION_DATE   = "16th January 2023"
MRBIAS_URL            = "http://github.com/JamesCKorte/mrbias"
MRBIAS_DOI_URL        = "https://doi.org/10.1088/1361-6560/acbcbb"

MRBIAS_REPORT_IMAGE_DPI = 500

# Want a unique ROI label (across T1/T2/PD to allow combined/mixed ROI type dataframes)
# The Label->IDX maps are then for each specific ROI type (T1/T2/PD) and also map to the input yaml files
ROI_IDX_LABEL_MAP = OrderedDict()
T1_ROI_LABEL_IDX_MAP = OrderedDict()
for i in range(14):
    ROI_IDX_LABEL_MAP[i+1] = "t1_roi_%d" % (i+1)
    T1_ROI_LABEL_IDX_MAP["t1_roi_%d" % (i+1)] = i+1
T2_ROI_LABEL_IDX_MAP = OrderedDict()
for i in range(14):
    ROI_IDX_LABEL_MAP[i+14+1] = "t2_roi_%d" % (i+1)
    T2_ROI_LABEL_IDX_MAP["t2_roi_%d" % (i+1)] = i+14+1
# create a reverse lookup
ROI_LABEL_IDX_MAP = {v: k for k, v in ROI_IDX_LABEL_MAP.items()}


class ColourSettings(object):
    def __init__(self, matplotlib_cmap_name='nipy_spectral'):
        self.cmap_name = matplotlib_cmap_name
        self.cmap = mpl.cm.get_cmap(self.cmap_name)
        t1_roi_dx_list = list(T1_ROI_LABEL_IDX_MAP.values())
        self.norm_t1 = mpl.colors.Normalize(vmin=np.min(t1_roi_dx_list)-1, vmax=np.max(t1_roi_dx_list)+1)
        t2_roi_dx_list = list(T2_ROI_LABEL_IDX_MAP.values())
        self.norm_t2 = mpl.colors.Normalize(vmin=np.min(t2_roi_dx_list)-1, vmax=np.max(t2_roi_dx_list)+1)

    def get_ROI_colour(self, roi_label):
        if roi_label in T1_ROI_LABEL_IDX_MAP.keys():
            return self.cmap(self.norm_t1(T1_ROI_LABEL_IDX_MAP[roi_label]))
        elif roi_label in T2_ROI_LABEL_IDX_MAP.keys():
            return self.cmap(self.norm_t2(T2_ROI_LABEL_IDX_MAP[roi_label]))
        else:
            log("ColourSettings::get_ROI_colour(): label %s not found returning default black" % roi_label,
                LogLevels.LOG_WARNING)
            return (0.0, 0.0, 0.0)

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