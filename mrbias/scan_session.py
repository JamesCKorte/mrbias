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

from abc import ABC, abstractmethod

import os
from collections import OrderedDict
from enum import IntEnum

import pydicom as dcm

import pandas as pd
pd.options.mode.chained_assignment = 'raise' # DEBUG: for finding SettingWithCopyWarning
import SimpleITK as sitk
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
from mrbias.misc_utils import LogLevels
from mrbias.image_sets import ImageGeometric
from mrbias.image_sets import ImageProtonDensity, ImageSetT1VIR, ImageSetT1VFA, ImageSetT2MSE, ImageSetT2Star, ImageSetDW
from mrbias.image_sets import ADCMap

# for pdf output
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas



def main():

    # -------------------------------
    # Basic test
    # ------------------------------
    # Run the ScanSessionSiemensSkyra class on two test dicom directories
    # Visually inspect output to ensure all types of images are located
    # todo: change to function with a boolean pass criteria

    # Skyra test
    dcm_dir_a = os.path.join(mu.reference_data_directory(), "mrbias_testset_A")
    dcm_dir_b = os.path.join(mu.reference_data_directory(), "mrbias_testset_B")
    test_dcm_dir_list = [dcm_dir_a, dcm_dir_b]
    test_ss_config_list = ["SiemensSkyra", "SiemensSkyra"]

    # Skyra DW test
    test_ss_config_list = []
    test_dcm_dir_list = []
    test_dcm_dir_list.append(r"I:\JK\MR-BIAS\Data_From_Maddie\Carr2022_data\01_MONTH")
    test_ss_config_list.append("DiffSiemensSkyra")


    # setup the logger to write to file
    mu.initialise_logger("scan_session.log", force_overwrite=True, write_to_screen=True)
    # setup a pdf to test the pdf reporting
    pdf = mu.PDFSettings()
    c = canvas.Canvas("scan_session.pdf", landscape(pdf.page_size))


    for dcm_dir, ss_type in zip(test_dcm_dir_list, test_ss_config_list):
        mu.log("="*100, LogLevels.LOG_INFO)
        mu.log("SCANNING DICOM DIR: %s" % dcm_dir, LogLevels.LOG_INFO)
        mu.log("="*100, LogLevels.LOG_INFO)
        # parse the DICOM directory and filter image sets
        if ss_type == "PhilipsMarlin":
            scan_session = SystemSessionPhilipsMarlin(dcm_dir)
        if ss_type == "SiemensSkyra":
            scan_session = SystemSessionSiemensSkyra(dcm_dir)
        elif ss_type == "PhilipsIngeniaAmbitionX":
            scan_session = SystemSessionPhilipsIngeniaAmbitionX(dcm_dir)
        elif ss_type == "DiffPhilipsIngeniaAmbitionX":
            scan_session = DiffusionSessionPhilipsIngeniaAmbitionX(dcm_dir)
        elif ss_type == "DiffSiemensSkyra":
            scan_session = DiffusionSessionSiemensSkyra(dcm_dir)
        scan_session.write_pdf_summary_page(c)


        geometric_images = scan_session.get_geometric_images()
        for geom_image in geometric_images:
            mu.log("Found GEO: %s" % type(geom_image), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(geom_image), LogLevels.LOG_INFO)

        pd_images = scan_session.get_proton_density_images()
        for pd_image in pd_images:
            mu.log("Found PD: %s" % type(pd_image), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(pd_image), LogLevels.LOG_INFO)

        t1_vir_imagesets = scan_session.get_t1_vir_image_sets()
        for t1_vir_imageset in t1_vir_imagesets:
            mu.log("Found T1(VIR): %s" % type(t1_vir_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t1_vir_imageset), LogLevels.LOG_INFO)

        t1_vfa_imagesets = scan_session.get_t1_vfa_image_sets()
        for t1_vfa_imageset in t1_vfa_imagesets:
            mu.log("Found T1(VFA): %s" % type(t1_vfa_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t1_vfa_imageset), LogLevels.LOG_INFO)

        t2_mse_imagesets = scan_session.get_t2_mse_image_sets()
        for t2_mse_imageset in t2_mse_imagesets:
            mu.log("Found T2(MSE): %s" % type(t2_mse_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t2_mse_imageset), LogLevels.LOG_INFO)

        dw_imagesets = scan_session.get_dw_image_sets()
        for dw_imageset in dw_imagesets:
            mu.log("Found DW: %s" % type(dw_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(dw_imageset), LogLevels.LOG_INFO)
        # give a visual break in the log
        mu.log("", LogLevels.LOG_INFO)

    # save the pdf report
    c.save()
    mu.log("------ FIN -------", LogLevels.LOG_INFO)



class ImageCatetory(IntEnum):
    GEOMETRY_3D = 1
    PROTON_DENSITY = 2
    T1_VIR = 3
    T1_VFA = 4
    T2_MSE = 5
    T2STAR_ME = 6
    DW = 7
    ADC = 8
    SECONDARY = 9
    UNKNOWN = 10
IMAGE_SECONDARY_STR = "secondary"       # this is when an image may have multiple secondary (i.e. not the magnitude image)
IMAGE_UNKNOWN_STR = "unknown"
# Category details dictionary
IMAGE_CAT_STR_AND_DICOM_DICT = OrderedDict()
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.GEOMETRY_3D] = ("geom", None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.PROTON_DENSITY] = ("pd", None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.T1_VIR] = ("t1_vir", ["InversionTime", "RepetitionTime"])
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.T1_VFA] = ("t1_vfa", ["FlipAngle", "RepetitionTime"])
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.T2_MSE] = ("t2_mse", None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.T2STAR_ME] = ("t2star_me", None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.DW] = ("dw", None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.ADC] = ("adc", None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.SECONDARY] = (IMAGE_SECONDARY_STR, None)
IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.UNKNOWN] = (IMAGE_UNKNOWN_STR, None)
# Dictionary of strings for labeling of different categories (i.e dataframes entries, foldername prefixes)
IMAGE_CAT_STR_DICT = OrderedDict()
# Dictionary of important DICOM fields (the variable ones) for identifying sets
IMAGE_CAT_DICOM_TAG_DICT = OrderedDict()
# construct from settings dicitonary
for key, (cat_str, dicom_tag_list) in IMAGE_CAT_STR_AND_DICOM_DICT.items():
    IMAGE_CAT_STR_DICT[key] = cat_str
    IMAGE_CAT_DICOM_TAG_DICT[key] = dicom_tag_list



""" 
ScanSessionAbstract
Take a dataframe with dicom image metadata (filepath and metadata for each image slice) and
applies logical filters to extract the desired image sequences for phantom quantification
the filtering functions are to be implemented in concrete subclasses and return
simpleITK image(s) and associated scan parameters (i.e. echo times, inversion times)

Helper functions to filter the MRI meta-data are implemented in this class to be 
used by each concrete sub-class (i.e. ScanSessionSiemensSkyra, ScanSessionPhilipsIngenia etc.)
"""
class ScanSessionAbstract(ABC):
    def __init__(self, dicom_dir, force_geometry_imageset=None):
        assert os.path.isdir(dicom_dir), "ScanSessionAbstract::init(): " \
                                         "dicom_dir must be a valid directory : %s" % dicom_dir
        # image and imageSet lists to populate
        self.geom_image_list = None
        self.pd_image_list = None
        self.vir_imageset_list = None
        self.vfa_imageset_list = None
        self.t2_imageset_list = None
        self.t2star_imageset_list = None
        self.dw_imageset_list = None
        # parameter map lists
        self.adc_map_list = None

        # force the scan session to use another type of image as the geometric image for ROI detection
        self.force_geometry_imageset = force_geometry_imageset

        # search the dicom directory and strip tags to populate a metadata dataframe
        self.dicom_searcher = mu.DICOMSearch(dicom_dir)
        self.meta_data_df = self.dicom_searcher.get_df()
        assert not self.meta_data_df.empty, "ScanSessionAbstract::init(): " \
                                            "no valid dicom files found in directory : %s" % dicom_dir
        # order the sequence list by date and time
        self.meta_data_df = self.meta_data_df.sort_values(['SeriesDate', 'SeriesTime'], ascending=[True, True])

        # -----------------------------------------------------------------
        # label the series with a category
        # -----------------------------------------------------------------
        self.meta_data_df["Category"] = IMAGE_UNKNOWN_STR
        category_list = IMAGE_CAT_STR_DICT.values()
        variable_interest_list = IMAGE_CAT_DICOM_TAG_DICT.values()
        series_pd_idx_category_list = [self.get_geometric_series_UIDs(),
                                       self.get_proton_density_series_UIDs(),
                                       self.get_t1_vir_series_UIDs(),
                                       self.get_t1_vfa_series_UIDs(),
                                       self.get_t2_series_UIDs(),
                                       self.get_t2star_series_UIDs(),
                                       self.get_dw_series_UIDs(),
                                       self.get_adc_series_UIDs()]

        for category_name, pd_index_list in zip(category_list, series_pd_idx_category_list):
            if pd_index_list is not None:
                self.meta_data_df.loc[pd_index_list, "Category"] = category_name
        # mark geometric images
        self.meta_data_df["IsGeometric"] = False
        if self.force_geometry_imageset is None:
            geom_str = IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.GEOMETRY_3D][0]
            self.meta_data_df.loc[self.meta_data_df["Category"] == geom_str, "IsGeometric"] = True
        else:
            force_cat_str = IMAGE_CAT_STR_AND_DICOM_DICT[self.force_geometry_imageset][0]
            self.meta_data_df.loc[self.meta_data_df["Category"] == force_cat_str, "IsGeometric"] = True

        # -----------------------------------------------------------------
        # Match each image  to a geometry image
        # - first give a unique label to each geometry label
        # - loop over the image in acquisition order and link with the last geom image taken
        # -----------------------------------------------------------------
        self.meta_data_df["ReferenceGeometryImage"] = ""
        self.meta_data_df["ImageSet"] = ""
        # --------------------------------------------------------
        # - group the image sets together and label numerically (i.e. geometry_0, geometry_1)
        df = self.meta_data_df.drop_duplicates(subset=["SeriesInstanceUID"])
        geo_category_name, geo_dicom_tags_of_interest = IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.GEOMETRY_3D]
        category_idx = 0
        for idx, r in df.iterrows():
            if r.Category == geo_category_name:
                set_name = mu.key_fmt % (geo_category_name, category_idx)
                # apply the label to the main dataframe
                self.meta_data_df.loc[
                    self.meta_data_df["SeriesInstanceUID"].isin([r.SeriesInstanceUID]), "ImageSet"] = set_name
                category_idx = category_idx + 1

        # -----------------------------------------------------------------
        # split the categories into sets
        # -----------------------------------------------------------------
        # group the image sets together and label numerically (i.e. t1_vir_0, t1_vir_0)
        df = self.meta_data_df.copy(deep=True)
        df['Category'].replace(IMAGE_UNKNOWN_STR, np.nan, inplace=True)
        df.dropna(subset=['Category'], inplace=True)
        df = df.drop_duplicates(subset=["SeriesInstanceUID"])
        for category_name, variables_of_interest in zip(category_list, variable_interest_list):
            # skip geometry images as they have already been labeled
            if (not (category_name == geo_category_name)) and (not (category_name == IMAGE_UNKNOWN_STR)):
                # iterate over the rows of the ordered dataframe
                # compare the category variable of interest (i.e. flip ange in T1_VFA) & the reference geometry
                # if the current row is a duplicate / exists in the current group then create a new group
                group_seriesUID_dict = OrderedDict()
                current_group = OrderedDict()
                category_idx = 0
                for idx, r in df.iterrows():
                    # only consider rows of the current category
                    if category_name == r.Category:
                        set_name = mu.key_fmt % (category_name, category_idx)
                        # if its not based on a variable then every image will be a unique on in the category (i.e. PD/T2 series)
                        if variables_of_interest is None:
                            group_seriesUID_dict[set_name] = [r.SeriesInstanceUID]
                            category_idx = category_idx + 1
                        else:
                            # Duplicates based on the reference geometry and
                            # variables of interest (flip angle, inversion time etc.)
                            # build the key
                            match_list = []
                            for v in variables_of_interest:
                                match_list.append(r[v])
                            match_key = tuple(match_list)
                            # check if its a duplicate of something in the current group
                            if match_key in current_group.keys():
                                # store a list of seriesUIDs from the current group
                                group_seriesUID_dict[set_name] = list(current_group.values())
                                category_idx = category_idx + 1
                                current_group.clear()
                                # start a new group with the duplicate
                                current_group[match_key] = r.SeriesInstanceUID
                            else:
                                # add it to the current group
                                current_group[match_key] = r.SeriesInstanceUID
                # save the last group (no duplicate event to trigger save in loop)
                set_name = mu.key_fmt % (category_name, category_idx)
                group_seriesUID_dict[set_name] = list(current_group.values())
                # apply the label to the main dataframe
                for set_name, seriesUID_list in group_seriesUID_dict.items():
                    # update so the "set_name" is based on category_name also
                    self.meta_data_df.loc[self.meta_data_df["SeriesInstanceUID"].isin(seriesUID_list) & self.meta_data_df["Category"].str.match(category_name), "ImageSet"] = set_name
                    # also label any secondary images
                    self.meta_data_df.loc[self.meta_data_df["SeriesInstanceUID"].isin(seriesUID_list) & self.meta_data_df["Category"].str.match(IMAGE_UNKNOWN_STR),  "Category"] = IMAGE_SECONDARY_STR
                    self.meta_data_df.loc[self.meta_data_df["SeriesInstanceUID"].isin(seriesUID_list) & self.meta_data_df["Category"].str.match(IMAGE_SECONDARY_STR),  "ImageSet"] = set_name


        # label the series which will be used for geometry
        self.meta_data_df["GeomSet"] = ""
        df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
        # mark the images which are tagged as geometric (to handle forced override)
        category_idx = 0
        for idx, r in df.iterrows():
            if r.IsGeometric:
                set_name = mu.key_fmt % ("g", category_idx)
                # apply the label to the main dataframe
                self.meta_data_df.loc[
                    self.meta_data_df["SeriesInstanceUID"].isin([r.SeriesInstanceUID]), "GeomSet"] = set_name
                category_idx = category_idx + 1
        # --------------------------------------------------------
        # - loop over the image in acquisition order and link with the last geom image taken
        df = self.meta_data_df.drop_duplicates(subset=["SeriesInstanceUID"])
        geom_options = pd.unique(df["GeomSet"]).tolist()
        if '' in geom_options:
            geom_options.remove('')
        if len(geom_options):
            current_geom = geom_options[0]
            for idx, r in df.iterrows():
                if r.IsGeometric and (r.GeomSet != ''):
                    current_geom = r.GeomSet
                # apply the reference geometry link to the main dataframe
                self.meta_data_df.loc[self.meta_data_df["SeriesInstanceUID"].isin([r.SeriesInstanceUID]), "ReferenceGeometryImage"] = current_geom

        # link ADC maps to an DWI image series if possible
        self.meta_data_df["ReferenceImageSet"] = ""
        # - loop over the image in acquisition order and link with the last geom image taken
        df = self.meta_data_df.drop_duplicates(subset=["SeriesInstanceUID"])
        dwi_str = IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.DW][0]
        adc_str = IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.ADC][0]
        dwi_options = pd.unique(df[df["Category"] == dwi_str]["ImageSet"]).tolist()
        if len(dwi_options):
            current_dwi = dwi_options[0]
            for idx, r in df.iterrows():
                if r.Category == dwi_str:
                    current_dwi = r.ImageSet
                # apply the reference dwi link to the main dataframe
                if r.Category == adc_str:
                    self.meta_data_df.loc[self.meta_data_df["SeriesInstanceUID"].isin(
                        [r.SeriesInstanceUID]), "ReferenceImageSet"] = current_dwi

        # output the categorisation, set grouping & reference images
        self.log_meta_dataframe()

        # todo: remove (here for debugging new scansession classes)
        self.meta_data_df.to_csv("check_labeling.csv")

    @abstractmethod
    def get_geometric_series_UIDs(self):
        return None
    @abstractmethod
    def get_proton_density_series_UIDs(self):
        return None
    @abstractmethod
    def get_t1_vir_series_UIDs(self):
        return None
    @abstractmethod
    def get_t1_vfa_series_UIDs(self):
        return None
    @abstractmethod
    def get_t2_series_UIDs(self):
        return None
    @abstractmethod
    def get_t2star_series_UIDs(self):
        return None
    @abstractmethod
    def get_dw_series_UIDs(self):
        return None
    @abstractmethod
    def get_adc_series_UIDs(self):
        return None

    # output the categorisation, set grouping & reference images (via the log)
    def log_meta_dataframe(self, show_unknown=True):
        def mid_truncate(s, n):
            s = str(s)
            if len(s) > n:
                chunk_n = int(np.floor(n/2)) - 2
                s = s[:chunk_n] + "..." + s[-chunk_n:]
            return s

        df = self.meta_data_df.copy(deep=True)
        if show_unknown is False:
            df['Category'].replace(IMAGE_UNKNOWN_STR, np.nan, inplace=True)
            df.dropna(subset=['Category'], inplace=True)
        df = df.drop_duplicates(subset=["SeriesInstanceUID", "Category"])
        table_width = 182 + 16
        mu.log("=" * table_width, LogLevels.LOG_INFO)
        mu.log(
            "| DATE    | TIME    | DESCRIPTION                    | CATEGORY      | IMAGE SET     | GEOM SET  | REF GEOM.     | REF IMSET.    | SERIES_UID                                                        |",
            LogLevels.LOG_INFO)
        mu.log("=" * table_width, LogLevels.LOG_INFO)
        cur_image_set = ""
        for idx, r in df.iterrows():
            if (r.ImageSet == "") or ((r.ImageSet != cur_image_set) and (not (r.Category in [IMAGE_UNKNOWN_STR]))):
                mu.log("-" * table_width, LogLevels.LOG_INFO)
                cur_image_set = r.ImageSet
            series_descrip = mid_truncate(r.SeriesDescription, 30)
            mu.log("| %s | %6s | %30s | %13s | %13s | %9s | %13s | %13s | %65s |" %
                   (r.SeriesDate, str(r.SeriesTime).split(".")[0], series_descrip,
                    r.Category, r.ImageSet, r.GeomSet, r.ReferenceGeometryImage, r.ReferenceImageSet, r.SeriesInstanceUID),
                   LogLevels.LOG_INFO)
        mu.log("=" * table_width, LogLevels.LOG_INFO)

    # same output as log but to pdf
    def write_pdf_summary_page(self, c):
        df = self.meta_data_df.copy(deep=True)
        df['Category'].replace('', np.nan, inplace=True)
        df.dropna(subset=['Category'], inplace=True)
        df = df.drop_duplicates(subset=["SeriesInstanceUID", "Category"])
        table_width = 179
        pdf = mu.PDFSettings()
        c.setFont(pdf.font_name, pdf.small_font_size)  # set to a fixed width font
        c.drawString(pdf.left_margin + pdf.page_width/3.,
                     pdf.page_height - pdf.top_margin,
                     "Image Sorting : Summary")
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - pdf.small_line_width,
                     "=" * table_width)
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - 2*pdf.small_line_width,
                     "| DATE    | TIME    | DESCRIPTION                    | CATEGORY      | IMAGE SET     | GEOM SET  | REF GEOM. | SERIES_UID                                                        |")
        c.drawString(pdf.left_margin, pdf.page_height - pdf.top_margin - 3*pdf.small_line_width,
                     "=" * table_width)
        cur_image_set = ""
        line_offset = 0
        for line_dx, (idx, r) in enumerate(df.iterrows()):
            if (r.ImageSet == "") or ((r.ImageSet != cur_image_set) and (not (r.Category in [IMAGE_UNKNOWN_STR]))):
                c.drawString(pdf.left_margin,
                             pdf.page_height - pdf.top_margin - pdf.small_line_width * (line_dx + 4 + line_offset),
                             "-" * table_width)
                line_offset = line_offset + 1
                cur_image_set = r.ImageSet
            series_descrip = r.SeriesDescription
            if len(series_descrip) > 30:
                series_descrip = series_descrip[0:27] + "..."
            c.drawString(pdf.left_margin,
                         pdf.page_height - pdf.top_margin - pdf.small_line_width*(line_dx + 4 + line_offset),
                         "| %s | %6s | %30s | %13s | %13s | %9s | %9s | %65s |" %
                         (r.SeriesDate, str(r.SeriesTime).split(".")[0], series_descrip,
                          r.Category, r.ImageSet, r.GeomSet, r.ReferenceGeometryImage, r.SeriesInstanceUID))
        c.drawString(pdf.left_margin,
                     pdf.page_height - pdf.top_margin - pdf.small_line_width*(line_dx + 5 + line_offset),
                     "=" * table_width)
        c.showPage()                             # new page

    def get_geometric_images(self):
        if self.geom_image_list is None:
            mu.log("ScanSessionAbstract::get_geometric_images(): "
                   "loading geometric images from disk...", LogLevels.LOG_INFO)
            df = self.meta_data_df.drop_duplicates(subset=["GeomSet"])
            geom_options = pd.unique(df["GeomSet"]).tolist()
            if '' in geom_options:
                geom_options.remove('')
            geomset_series_uid_list = df[df["GeomSet"].isin(geom_options)].SeriesInstanceUID
            geom_image_list = []
            for series_instance_uid, geoset_label in zip(geomset_series_uid_list, geom_options):
                image_set_df = self.meta_data_df[self.meta_data_df.SeriesInstanceUID == series_instance_uid].copy()

                # check if it is a geometric image (not diffusion, T1 or T2, etc...)
                image_set_cat = pd.unique(image_set_df["Category"]).tolist()[0]
                if (image_set_cat == IMAGE_CAT_STR_AND_DICOM_DICT[ImageCatetory.DW][0]):
                    dw_imagesets = self.get_dw_image_sets(ignore_geometric_images=True)
                    for dw_imset in dw_imagesets:
                        #does this set match the one we want to use as geometric image
                        if series_instance_uid == dw_imset.series_instance_UIDs[0]:
                            #create an ImageGeometric from diffusion image
                            geom_image_list.append(ImageGeometric(geoset_label,
                                                                  dw_imset.image_list[0], #use b = 0
                                                                  series_instance_uid,
                                                                  bits_allocated = None,
                                                                  bits_stored = None,
                                                                  rescale_slope = None,
                                                                  rescale_intercept = None))
                else:
                    image_set_df.rename(columns={"SeriesInstanceUID": "SeriesUID"}, inplace=True)
                    rval = self._load_image_from_df(image_set_df, series_descrip=geoset_label)
                    im, bits_allocated, bits_stored, rescale_slope, rescale_intercept, scale_slope, scale_intercept = rval
                    geom_image_list.append(ImageGeometric(geoset_label,
                                                          im,
                                                          series_instance_uid,
                                                          bits_allocated, bits_stored,
                                                          rescale_slope, rescale_intercept,
                                                          scale_slope, scale_intercept))
            self.geom_image_list = geom_image_list
        return self.geom_image_list

    def get_proton_density_images(self):
        if self.pd_image_list is None:
            # construct the proton density images
            mu.log("ScanSessionAbstract::get_proton_density_images(): "
                   "loading proton density images from disk...", LogLevels.LOG_INFO)
            imageset_data_dict = self._get_imageset_data_from_df(ImageCatetory.PROTON_DENSITY)
            pd_image_list = []
            for imageset_name, (image_and_metadata_list, ref_geom_image) in imageset_data_dict.items():
                pd_images, metadata_list, other_list = zip(*image_and_metadata_list)
                series_instance_uid = other_list[0]["SeriesInstanceUID"]
                bits_allocated = other_list[0]["BitsAllocated"]
                bits_stored = other_list[0]["BitsStored"]
                rescale_slope = other_list[0]["RescaleSlope"]
                rescale_intercept = other_list[0]["RescaleIntercept"]
                assert len(
                    pd_images) == 1, "ScanSessionAbstract::get_proton_density_images() -there should only be one pd image in each set, found %d" % len(
                    pd_images)
                # create the imageset object and append to return list
                pd_image_list.append(ImageProtonDensity(imageset_name,
                                                        pd_images[0],
                                                        ref_geom_image,
                                                        series_instance_uid,
                                                        bits_allocated, bits_stored,
                                                        rescale_slope, rescale_intercept))
            self.pd_image_list = pd_image_list
        return self.pd_image_list

    def get_t1_vir_image_sets(self):
        if self.vir_imageset_list is None:
            # construct the T1 imagesets
            mu.log("ScanSessionAbstract::get_t1_vir_image_sets(): "
                   "loading T1 VIR image sets from disk...", LogLevels.LOG_INFO)
            imageset_data_dict = self._get_imageset_data_from_df(ImageCatetory.T1_VIR)
            self.vir_imageset_list = self._get_image_sets(imageset_data_dict, ImageCatetory.T1_VIR, n_expected_dcm_tags=2)
        return self.vir_imageset_list

    def get_t1_vfa_image_sets(self):
        if self.vfa_imageset_list is None:
            # construct the T1 imagesets
            mu.log("ScanSessionAbstract::get_t1_vfa_image_sets(): "
                   "loading T1 VFA image sets from disk...", LogLevels.LOG_INFO)
            imageset_data_dict = self._get_imageset_data_from_df(ImageCatetory.T1_VFA)
            self.vfa_imageset_list = self._get_image_sets(imageset_data_dict, ImageCatetory.T1_VFA, n_expected_dcm_tags=2)
        return self.vfa_imageset_list

    def get_t2_mse_image_sets(self):
        if self.t2_imageset_list is None:
            mu.log("ScanSession::get_t2_image_sets(): "
                   "loading T2 image sets from disk...", LogLevels.LOG_INFO)
            self.t2_imageset_list = self._get_echo_image_sets(ImageCatetory.T2_MSE)
        return self.t2_imageset_list

    def get_t2star_image_sets(self):
        if self.t2star_imageset_list is None:
            mu.log("ScanSession::get_t2star_image_sets(): "
                   "loading T2star image sets from disk...", LogLevels.LOG_INFO)
            self.t2star_imageset_list = self._get_echo_image_sets(ImageCatetory.T2STAR_ME)
        return self.t2star_imageset_list

    def get_dw_image_sets(self, ignore_geometric_images=False):
        # if DW imageset has been loaded without geometric images linked then trigger a reload
        if (self.dw_imageset_list is not None) and (not ignore_geometric_images):
            dw_imageset_has_all_geom_loaded = True
            for dw_image in self.dw_imageset_list:
                if dw_image.get_geometry_image() is None:
                    dw_imageset_has_all_geom_loaded = False
            if not dw_imageset_has_all_geom_loaded:
                self.dw_imageset_list = None
        # Load the DW imageset
        if self.dw_imageset_list is None:
            mu.log("ScanSession::get_dw_image_sets(): "
                   "loading DW image sets from disk...", LogLevels.LOG_INFO)
            self.dw_imageset_list = self._get_dw_image_sets(ImageCatetory.DW, ignore_geometric_images)
        return self.dw_imageset_list

    def _get_image_sets(self, imageset_data_dict, category, n_expected_dcm_tags):
        # acquisition timestamp
        study_date, study_time = self.get_study_date_time()
        imageset_list = []
        for imageset_name, (image_and_metadata_list, ref_geom_image) in imageset_data_dict.items():
            images, metadata_list, other_list = zip(*image_and_metadata_list)
            series_instance_uids = [x["SeriesInstanceUID"] for x in other_list]
            bits_allocated = other_list[0]["BitsAllocated"]
            bits_stored = other_list[0]["BitsStored"]
            rescale_slope_list     = [x["RescaleSlope"] for x in other_list]
            rescale_intercept_list = [x["RescaleIntercept"] for x in other_list]
            scale_slope_list     = [x["ScaleSlope"] for x in other_list]
            scale_intercept_list = [x["ScaleIntercept"] for x in other_list]
            # scanner details
            scanner_make = other_list[0]["Manufacturer"]
            scanner_model = other_list[0]["ManufacturerModelName"]
            scanner_sn = other_list[0]["DeviceSerialNumber"]
            scanner_field_strength = other_list[0]["MagneticFieldStrength"]
            # pull out the metadata (inversion recovery times, flip angles, echo times etc.)
            assert len(metadata_list[0]) == n_expected_dcm_tags, \
                "ScanSessionAbstract::_get_image_sets() - %s should have %d dicom parameters per image, found %d" % (
                category, len(metadata_list[0]), n_expected_dcm_tags)
            # handle the different types of image sets
            image_set = None
            if category == ImageCatetory.T1_VFA:
                flip_angle_list = [x[0] for x in metadata_list]
                repetition_time_list = [x[1] for x in metadata_list]
                # sort the images by the inversion recovery time
                flip_and_image_list = [(fa, tr, im, suid, rs, ri, ss, si) for im, fa, tr, suid, rs, ri, ss, si in
                                       zip(images, flip_angle_list, repetition_time_list, series_instance_uids,
                                           rescale_slope_list, rescale_intercept_list,
                                           scale_slope_list, scale_intercept_list)]
                flip_and_image_list.sort(key=lambda x: x[0])
                flip_angle_list, repetition_time_list, images, series_instance_uids,\
                    rescale_slope_list, rescale_intercept_list, scale_slope_list, scale_intercept_list = zip(*flip_and_image_list)
                image_set = ImageSetT1VFA(imageset_name,
                                          images,
                                          flip_angle_list,
                                          repetition_time_list,
                                          ref_geom_image,
                                          series_instance_uids,
                                          bits_allocated, bits_stored,
                                          rescale_slope_list, rescale_intercept_list,
                                          scale_slope_list, scale_intercept_list,
                                          scanner_make, scanner_model, scanner_sn, scanner_field_strength,
                                          study_date, study_time)
            elif category == ImageCatetory.T1_VIR:
                inversion_time_list = [x[0] for x in metadata_list]
                repetition_time_list = [x[1] for x in metadata_list]
                # sort the images by the inversion recovery time
                tir_time_and_image_list = [(tir, tr, im, suid, rs, ri, ss, si) for im, tir, tr, suid, rs, ri, ss, si in
                                           zip(images, inversion_time_list, repetition_time_list,
                                               series_instance_uids,
                                               rescale_slope_list, rescale_intercept_list,
                                               scale_slope_list, scale_intercept_list)]
                tir_time_and_image_list.sort(key=lambda x: x[0])
                inversion_time_list, repetition_time_list, images, series_instance_uids,\
                    rescale_slope_list, rescale_intercept_list, scale_slope_list, scale_intercept_list = zip(*tir_time_and_image_list)
                # create the imageset object and append to return list
                image_set = ImageSetT1VIR(imageset_name,
                                          images,
                                          inversion_time_list,
                                          repetition_time_list,
                                          ref_geom_image,
                                          series_instance_uids,
                                          bits_allocated, bits_stored,
                                          rescale_slope_list, rescale_intercept_list,
                                          scale_slope_list, scale_intercept_list,
                                          scanner_make, scanner_model, scanner_sn, scanner_field_strength,
                                          study_date, study_time)
            else:
                assert False, "ScanSessionAbstract::_get_image_sets() is only designed to be called by T1-VFA and T1-VIR models, not (%s) imageset category" % category
            # create the imageset object and append to return list
            if image_set is not None:
                imageset_list.append(image_set)
        return imageset_list

    def _get_echo_image_sets(self, category):
        imageset_list = []
        # acquisition timestamp
        study_date, study_time = self.get_study_date_time()
        # geometric images for linking
        geom_images = self.get_geometric_images()
        # get image sets from the dataframe
        category_name = IMAGE_CAT_STR_DICT[category]
        cat_df = self.meta_data_df[self.meta_data_df.Category == category_name]
        imageset_names = cat_df.drop_duplicates(subset=["ImageSet"]).ImageSet
        # loop over the T2 star sets
        for set_name in imageset_names:
            df_t2 = cat_df[cat_df.ImageSet == set_name]
            df_t2 = df_t2.sort_values(by=["EchoTime"])

            # scanner details
            scanner_make = df_t2.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
            scanner_model = df_t2.drop_duplicates(subset=["ManufacturerModelName"]).ManufacturerModelName.iloc[0]
            scanner_sn = df_t2.drop_duplicates(subset=["DeviceSerialNumber"]).DeviceSerialNumber.iloc[0]
            scanner_field_strength = df_t2.drop_duplicates(subset=["MagneticFieldStrength"]).MagneticFieldStrength.iloc[0]

            # sort the echos into groups (to handle multiple slices)
            image_echo_time_list = []
            for index, row in df_t2.iterrows():
                d_vec = [float(row["EchoTime"]), float(row["RepetitionTime"]),
                         row["SeriesInstanceUID"], row["SOPInstanceUID"], row["ImageFilePath"], row["SliceLocation"],
                         int(row["BitsAllocated"]), int(row["BitsStored"]), row["RescaleSlope"], row["RescaleIntercept"]]
                if scanner_make == "Philips":
                    d_vec.append(row["ScaleSlope"])
                    d_vec.append(row["ScaleIntercept"])
                image_echo_time_list.append(d_vec)
            image_echo_time_list.sort(key=lambda x: x[0])
            # get the unique echo list
            echo_time_list = [x[0] for x in image_echo_time_list]
            repetition_time_list = [x[1] for x in image_echo_time_list]
            series_instance_uids = [x[2] for x in image_echo_time_list]
            # load up each of the echo images
            unique_echo_list = pd.unique(echo_time_list)
            col_list = ["EchoTime", "RepetitionTime", "SeriesUID", "SOPInstanceUID",
                        "ImageFilePath", "SliceLocation",
                        "BitsAllocated", "BitsStored", "RescaleSlope", "RescaleIntercept"]
            if scanner_make == "Philips":
                col_list.append("ScaleSlope")
                col_list.append("ScaleIntercept")
            df_echo_images = pd.DataFrame(
                columns=col_list,
                data=image_echo_time_list)
            image_list = []
            image_info_list = []
            for echo_time in unique_echo_list:
                df_image = df_echo_images[df_echo_images["EchoTime"] == echo_time].copy()
                df_image.loc[:, "Manufacturer"] = scanner_make
                rval = self._load_image_from_df(df_image, series_descrip="%s-%s" % (set_name, echo_time))
                im, bits_allocated, bits_stored, rescale_slope, rescale_intercept, scale_slope, scale_intercept = rval
                image_list.append(im)
                image_info_list.append((bits_allocated, bits_stored,
                                        rescale_slope, rescale_intercept,
                                        scale_slope, scale_intercept))

            # include any associated geometry image
            ref_geom_image = None
            image_set_df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
            ref_geom_label = image_set_df[image_set_df.ImageSet == set_name].ReferenceGeometryImage.iloc[0]
            for g_im in geom_images:
                if g_im.get_label() == ref_geom_label:
                    ref_geom_image = g_im
            bits_allocated_list    = [x[0] for x in image_info_list]
            bits_stored_list       = [x[1] for x in image_info_list]
            rescale_slope_list     = [x[2] for x in image_info_list]
            rescale_intercept_list = [x[3] for x in image_info_list]
            scale_slope_list       = [x[4] for x in image_info_list]
            scale_intercept_list   = [x[5] for x in image_info_list]
            # create the appropriate ImageSet based on category
            if category == ImageCatetory.T2_MSE:
                imageset_list.append(ImageSetT2MSE(set_name,
                                                   image_list,
                                                   unique_echo_list,
                                                   repetition_time_list,
                                                   ref_geom_image,
                                                   series_instance_uids,
                                                   bits_allocated_list[0], bits_stored_list[0], # assume same across echos
                                                   rescale_slope_list, rescale_intercept_list,
                                                   scale_slope_list, scale_intercept_list,
                                                   scanner_make, scanner_model, scanner_sn,
                                                   scanner_field_strength,
                                                   study_date, study_time))
            elif category == ImageCatetory.T2STAR_ME:
                imageset_list.append(ImageSetT2Star(set_name,
                                                    image_list,
                                                    unique_echo_list,
                                                    repetition_time_list,
                                                    ref_geom_image,
                                                    series_instance_uids,
                                                    bits_allocated_list[0], bits_stored_list[0], # assume same across echos
                                                    rescale_slope_list, rescale_intercept_list,
                                                    scale_slope_list, scale_intercept_list,
                                                    scanner_make, scanner_model, scanner_sn,
                                                    scanner_field_strength,
                                                    study_date, study_time))
            else:
                assert False, "ScanSessionAbstract::_get_echo_image_sets() is only designed to be called by T2_MSE and T2STAR_ME models, not (%s) imageset category" % category
        return imageset_list

    def _get_dw_image_sets(self, category, ignore_geometric_images=False):
        imageset_list = []
        # acquisition timestamp
        study_date, study_time = self.get_study_date_time()
        # geometric images for linking
        if not ignore_geometric_images:
            geom_images = self.get_geometric_images()
        # get ADC maps for linking
        adc_maps = self.get_adc_maps()
        # get image sets from the dataframe
        category_name = IMAGE_CAT_STR_DICT[category]
        cat_df = self.meta_data_df[self.meta_data_df.Category == category_name]
        imageset_names = cat_df.drop_duplicates(subset=["ImageSet"]).ImageSet
        # loop over the bval sets
        for set_name in imageset_names:
            df_diff = cat_df[cat_df.ImageSet == set_name]
            df_diff = df_diff.sort_values(by=["DiffusionBValue"])
            df_diff["DiffusionBValue"] = df_diff["DiffusionBValue"] * 1e-6 #convert b values from s/mm^2 to s/um^2 to match reference ADC units

            # scanner details
            scanner_make = df_diff.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
            scanner_model = df_diff.drop_duplicates(subset=["ManufacturerModelName"]).ManufacturerModelName.iloc[0]
            scanner_sn = df_diff.drop_duplicates(subset=["DeviceSerialNumber"]).DeviceSerialNumber.iloc[0]
            scanner_field_strength = df_diff.drop_duplicates(subset=["MagneticFieldStrength"]).MagneticFieldStrength.iloc[0]

            # sort the echos into groups (to handle multiple slices)
            image_b_val_list = []
            for index, row in df_diff.iterrows():
                d_vec = [float(row["DiffusionBValue"]), float(row["RepetitionTime"]),
                         row["SeriesInstanceUID"], row["SOPInstanceUID"],
                         row["ImageFilePath"], row["SliceLocation"],
                         int(row["BitsAllocated"]), int(row["BitsStored"]), row["RescaleSlope"], row["RescaleIntercept"]]
                if scanner_make == "Philips":
                    d_vec.append(row["ScaleSlope"])
                    d_vec.append(row["ScaleIntercept"])
                image_b_val_list.append(d_vec)
            image_b_val_list.sort(key=lambda x: x[0])
            # get the unique echo list
            b_value_list = [x[0] for x in image_b_val_list]
            repetition_time_list = [x[1] for x in image_b_val_list]
            series_instance_uids = [x[2] for x in image_b_val_list]
            # load up each of the echo images
            unique_bval_list = pd.unique(b_value_list)
            col_list = ["DiffusionBValue", "RepetitionTime", "SeriesUID", "SOPInstanceUID", "ImageFilePath", "SliceLocation",
                        "BitsAllocated", "BitsStored", "RescaleSlope", "RescaleIntercept"]
            if scanner_make == "Philips":
                col_list.append("ScaleSlope")
                col_list.append("ScaleIntercept")
            df_bval_images = pd.DataFrame(
                columns=col_list,
                data=image_b_val_list)
            image_list = []
            image_info_list = []
            for bval in unique_bval_list:
                df_b = df_bval_images[df_bval_images["DiffusionBValue"] == bval].copy()
                df_b.loc[:, "Manufacturer"] = scanner_make
                rval = self._load_image_from_df(df_b, series_descrip="%s-%s" % (set_name, bval))
                im, bits_allocated, bits_stored, rescale_slope, rescale_intercept, scale_slope, scale_intercept = rval
                image_list.append(im)
                image_info_list.append((bits_allocated, bits_stored,
                                        rescale_slope, rescale_intercept,
                                        scale_slope, scale_intercept))

            # include any associated geometry image
            ref_geom_image = None
            image_set_df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
            ref_geom_label = image_set_df[image_set_df.ImageSet == set_name].ReferenceGeometryImage.iloc[0]
            if not ignore_geometric_images:
                for g_im in geom_images:
                    if g_im.get_label() == ref_geom_label:
                        ref_geom_image = g_im
            bits_allocated_list    = [x[0] for x in image_info_list]
            bits_stored_list       = [x[1] for x in image_info_list]
            rescale_slope_list     = [x[2] for x in image_info_list]
            rescale_intercept_list = [x[3] for x in image_info_list]
            scale_slope_list       = [x[4] for x in image_info_list]
            scale_intercept_list   = [x[5] for x in image_info_list]
            # create the appropriate ImageSet based on category
            if category == ImageCatetory.DW:
                diffusion_imset = ImageSetDW(set_name,
                                             image_list,
                                             unique_bval_list,
                                             repetition_time_list,
                                             ref_geom_image,
                                             series_instance_uids,
                                             bits_allocated_list[0], bits_stored_list[0],  # assume same across echos
                                             rescale_slope_list, rescale_intercept_list,
                                             scale_slope_list, scale_intercept_list,
                                             scanner_make, scanner_model, scanner_sn,
                                             scanner_field_strength,
                                             study_date, study_time)
                # link any available ADC maps
                if adc_maps is not None:
                    for adc_map in adc_maps:
                        if(adc_map.reference_imageset_label == set_name):
                            diffusion_imset.add_derived_parameter_map(adc_map)
                imageset_list.append(diffusion_imset)
            else:
                assert False, "ScanSessionAbstract::_get_dw_image_sets() is only designed to be called by DWI models, not (%s) imageset category" % category
        return imageset_list

    def get_adc_maps(self):
        return self._get_parameter_maps(ImageCatetory.ADC)

    def _get_parameter_maps(self, category):
        map_loaded = False
        if (category == ImageCatetory.ADC) and (self.adc_map_list is not None):
            map_loaded = True

        if not map_loaded:
            mu.log("ScanSessionAbstract::_get_parameter_maps(): "
                   "loading parametric [%s] map from disk..." % IMAGE_CAT_STR_AND_DICOM_DICT[category][0], LogLevels.LOG_INFO)

            # get parameter map names from the dataframe
            pmap_cat_name = IMAGE_CAT_STR_DICT[category]
            pmap_cat_df = self.meta_data_df[self.meta_data_df.Category == pmap_cat_name]
            pmap_labels = pmap_cat_df.drop_duplicates(subset=["ImageSet"]).ImageSet
            # get the associated seriesUIDs
            df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
            pmap_series_uid_list = df[df["ImageSet"].isin(pmap_labels)].SeriesInstanceUID
            pmap_reference_imageset_list = df[df["ImageSet"].isin(pmap_labels)].ReferenceImageSet

            param_map_list = []
            for series_instance_uid, pmap_label, pmap_ref_imset in zip(pmap_series_uid_list, pmap_labels, pmap_reference_imageset_list):
                pmap_df = self.meta_data_df[self.meta_data_df.SeriesInstanceUID == series_instance_uid].copy()
                pmap_df.rename(columns={"SeriesInstanceUID": "SeriesUID"}, inplace=True)
                rval = self._load_image_from_df(pmap_df, series_descrip=pmap_label)
                sitk_im, bits_allocated, bits_stored, rescale_slope, rescale_intercept, scale_slope, scale_intercept = rval
                # get the date and time ('SeriesDate', 'SeriesTime')
                date_aqcuired = pmap_df['SeriesDate'].unique().tolist()[0]
                time_aqcuired = pmap_df['SeriesTime'].unique().tolist()[0]
                # create a corresponding object
                pmap = None
                if (category == ImageCatetory.ADC):
                    pmap = ADCMap(pmap_label,
                                  sitk_im,
                                  series_instance_uid,
                                  reference_imageset_label=pmap_ref_imset,
                                  bits_allocated=bits_allocated, bits_stored=bits_stored,
                                  rescale_slope=rescale_slope, rescale_intercept=rescale_intercept,
                                  scale_slope=scale_slope, scale_intercept=scale_intercept,
                                  date_acquired=date_aqcuired, time_acquired=time_aqcuired)

                # add it to the parameter map list
                param_map_list.append(pmap)

            # assign it to the correct class member
            if (category == ImageCatetory.ADC):
                self.adc_map_list = param_map_list

        # return the appropriate map
        if (category == ImageCatetory.ADC):
            return self.adc_map_list
        return None


    # helper function to load an image from a dataframe with some expected columns:
    # - Manufacturor, BitsAllocated, BitsStored, RescaleIntercept, ImageFilePath, SliceLocation, SeriesUID
    # additionally, if the scanner is a Phillips then also need to provide
    # - ScaleSlope, ScaleIntercept
    def _load_image_from_df(self, im_df, series_descrip=None):
        # drop any duplicates
        im_df.drop_duplicates(subset=['SeriesUID', 'SOPInstanceUID'], keep='first', inplace=True)

        # get the series UID and manufacturor
        series_uid = im_df.SeriesUID.values[0]
        scanner_make = im_df.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
        # get the datatype details
        bits_allocated = im_df.drop_duplicates(subset=["BitsAllocated"]).BitsAllocated.iloc[0]
        bits_stored = im_df.drop_duplicates(subset=["BitsStored"]).BitsStored.iloc[0]
        rescale_slope = im_df.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.iloc[0]
        rescale_intercept = im_df.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.iloc[0]
        assert im_df.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.shape[0] == 1, \
            "ScanSessionAbstract::_load_image_from_df(%s): multiple RescaleIntercept values (i.e. per slice) for a single series not supported by MRBIAS" \
            % (series_descrip)
        assert im_df.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.shape[0] == 1, \
            "ScanSessionAbstract::_load_image_from_df(%s): multiple RescaleSlope values (i.e. per slice) for a single series not supported by MRBIAS" \
            % (series_descrip)
        # get all the slices
        files_sorting = []
        for index, row in im_df.iterrows():
            files_sorting.append((row["ImageFilePath"], row["SliceLocation"]))
        # sort the slices by slice location before reading by sitk
        files_sorting.sort(key=lambda x: x[1])
        files_sorted = [x[0] for x in files_sorting]

        # handle manufacturor specific images
        scale_slope = None
        scale_intercept = None
        if scanner_make == "Philips":
            scale_slope = im_df.drop_duplicates(subset=["ScaleSlope"]).ScaleSlope.iloc[0]
            scale_intercept = im_df.drop_duplicates(subset=["ScaleIntercept"]).ScaleIntercept.iloc[0]
            assert im_df.drop_duplicates(subset=["ScaleIntercept"]).ScaleIntercept.shape[0] == 1, \
                "ScanSessionAbstract::_load_image_from_df(%s): multiple ScaleIntercept values (i.e. per slice) for a single series not supported by MRBIAS" \
                % (series_descrip)
            assert im_df.drop_duplicates(subset=["ScaleSlope"]).ScaleSlope.shape[0] == 1, \
                "ScanSessionAbstract::_load_image_from_df(%s): multiple ScaleSlope values (i.e. per slice) for a single series not supported by MRBIAS" \
                % (series_descrip)
        r_val = mu.load_image_from_filelist(files_sorted, series_uid,
                                            rescale_slope, rescale_intercept,
                                            scale_slope, scale_intercept,
                                            series_descrp=series_descrip,
                                            philips_scaling=(scanner_make == "Philips"))
        im, rescale_slope, rescale_intercept, scale_slope, scale_intercept = r_val
        return im, bits_allocated, bits_stored, rescale_slope, rescale_intercept, scale_slope, scale_intercept


    def get_spin_echo_series(self, df=None):
        if df is None:
            df = self.meta_data_df
        return df[df.ScanningSequence.str.contains("SE", na=False)]

    def get_gradient_echo_series(self, df=None):
        if df is None:
            df = self.meta_data_df
        return df[df.ScanningSequence.str.contains("GR", na=False)]

    def get_inversion_recovery_series(self, df=None):
        if df is None:
            df = self.meta_data_df
        return df[df.ScanningSequence.str.contains("IR", na=False)]

    def get_2D_series(self, df=None):
        if df is None:
            df = self.meta_data_df
        return df[df.MRAcquisitionType=="2D"]

    def get_3D_series(self, df=None):
        if df is None:
            df = self.meta_data_df
        return df[df.MRAcquisitionType=="3D"]

    # get the date and time of the first image
    def get_study_date_time(self):
        df = self.meta_data_df.copy(deep=True)
        df['Category'].replace('', np.nan, inplace=True)
        df.dropna(subset=['Category'], inplace=True)
        df = df.drop_duplicates(subset=["SeriesInstanceUID"])
        return df.StudyDate.iloc[0], df.StudyTime.iloc[0]

    # general helper function to get image sets and associated metadata from the ordered/grouped metadataframe
    # and include any referenced geometric images (ImageGeometric)
    def _get_imageset_data_from_df(self, image_category):
        geom_images = []
        # !! careful editing this (could lead to recursion bug : self.get_geometric_images() calls ...
        #                                                        ...       _get_imageset_data_from_df
        if image_category != ImageCatetory.GEOMETRY_3D:
            # load all geometry images for linking
            geom_images = self.get_geometric_images()
        # get image sets from the dataframe
        category_name, category_dicom_tag_list = IMAGE_CAT_STR_AND_DICOM_DICT[image_category]
        cat_df = self.meta_data_df[self.meta_data_df.Category == category_name]
        imageset_names = cat_df.drop_duplicates(subset=["ImageSet"]).ImageSet
        imageset_data_dict = OrderedDict()
        for set_name in imageset_names:
            image_set_df = cat_df[cat_df.ImageSet == set_name]
            set_series_uids = image_set_df.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
             # get the scanner details
            scanner_make = image_set_df.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
            scanner_model = image_set_df.drop_duplicates(subset=["ManufacturerModelName"]).ManufacturerModelName.iloc[0]
            scanner_sn = image_set_df.drop_duplicates(subset=["DeviceSerialNumber"]).DeviceSerialNumber.iloc[0]
            scanner_field_strength = image_set_df.drop_duplicates(subset=["MagneticFieldStrength"]).MagneticFieldStrength.iloc[0]

            # loop over the image slices
            image_and_metadata_list = []
            for series_uid in set_series_uids:
                # get the image details
                image_df = image_set_df[image_set_df.SeriesInstanceUID == series_uid].copy()
                # take the meta data from the first slice file/row
                first_row = image_df.iloc[0]
                dicom_meta_list = []
                if category_dicom_tag_list is not None:
                    for dcm_tag in category_dicom_tag_list:
                        dicom_meta_list.append(first_row[dcm_tag])
                # load the image
                image_df.rename(columns={"SeriesInstanceUID": "SeriesUID"}, inplace=True)
                rval = self._load_image_from_df(image_df, series_descrip=set_name)
                im, bits_allocated, bits_stored, rescale_slope, rescale_intercept, scale_slope, scale_intercept = rval

                # group the image and metadata together
                image_and_metadata_list.append((im,  # image
                                                tuple(dicom_meta_list),  # image set parameters (i.e. flip angle)
                                                {"SeriesInstanceUID": series_uid,
                                                 "BitsAllocated": bits_allocated,
                                                 "BitsStored": bits_stored,
                                                 "RescaleSlope": rescale_slope,
                                                 "RescaleIntercept": rescale_intercept,
                                                 "ScaleSlope" : scale_slope,
                                                 "ScaleIntercept": scale_intercept,
                                                 "Manufacturer": scanner_make,
                                                 "ManufacturerModelName": scanner_model,
                                                 "DeviceSerialNumber": scanner_sn,
                                                 "MagneticFieldStrength": scanner_field_strength},
                                                ))  # other parameters of interest
            # include any associated geometry image
            ref_geom_image = None
            image_set_df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
            ref_geom_label = image_set_df[image_set_df.ImageSet == set_name].ReferenceGeometryImage.iloc[0]
            for g_im in geom_images:
                if g_im.get_label() == ref_geom_label:
                    ref_geom_image = g_im
            imageset_data_dict[set_name] = (image_and_metadata_list, ref_geom_image)
        return imageset_data_dict



class SystemSessionAbstract(ScanSessionAbstract):
    def __init__(self, dicom_dir, force_geometry_imageset=None):
        super().__init__(dicom_dir, force_geometry_imageset)

    def get_dw_series_UIDs(self):
        return None

    def get_adc_series_UIDs(self):
        return None

class DiffusionSessionAbstract(ScanSessionAbstract):
    def __init__(self, dicom_dir, force_geometry_imageset=None):
        super().__init__(dicom_dir, force_geometry_imageset)

    def get_proton_density_series_UIDs(self):
        return None

    def get_t1_vir_series_UIDs(self):
        return None

    def get_t1_vfa_series_UIDs(self):
        return None

    def get_t2_series_UIDs(self):
        return None

    def get_t2star_series_UIDs(self):
        return None

#CONCRETE CLASSES

#TEMPLATE SYSTEM PHANTOM CLASS FOR NEW ADDITIONS
class SystemSessionTemplate(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        return None

    def get_t1_vir_series_UIDs(self):
        return None

    def get_t1_vfa_series_UIDs(self):
        return None

    def get_t2_series_UIDs(self):
        return None

    def get_proton_density_series_UIDs(self):
        return None

    def get_t2star_series_UIDs(self):
        return None


#TEMPLATE DIFFUSION PHANTOM CLASS FOR NEW ADDITIONS
class DiffusionSessionTemplate(DiffusionSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        return None

    def get_dw_series_UIDs(self):
        return None

    def get_adc_series_UIDs(self):
        return None



class DiffusionSessionPhilipsIngeniaAmbitionX(DiffusionSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        df_ge_3D = super().get_3D_series()
        df_ge_3D_1mm = df_ge_3D[df_ge_3D.SliceThickness == 1]
        df_ge_3D_1mm_psn = df_ge_3D_1mm[df_ge_3D_1mm["SequenceName"].str.match(r"(?=.*\bT1FFE\b)") == True]
        return df_ge_3D_1mm_psn.index

    def get_dw_series_UIDs(self):
        df_2D = super().get_2D_series()
        df_2D_thick = df_2D[df_2D.SliceThickness == 4]
        df_2D_psn = df_2D_thick[df_2D_thick["SequenceName"].str.match(r"(?=.*\bDwiSE\b)") == True]
        df_2D_psn = df_2D_psn.sort_values(["DiffusionBValue"])
        return df_2D_psn.index

    def get_adc_series_UIDs(self):
        return None


class DiffusionSessionSiemensSkyra(DiffusionSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir, force_geometry_imageset=ImageCatetory.DW)

    def get_geometric_series_UIDs(self):
        return None

    def get_dw_series_UIDs(self):
        df_2D = super().get_2D_series()
        df_2D_tracew = df_2D[df_2D["ImageType"].str.contains("'TRACEW'")]
        df_2D_tracew = df_2D_tracew.sort_values(["DiffusionBValue"])
        return df_2D_tracew.index

    def get_adc_series_UIDs(self):
        df_2D = super().get_2D_series()
        df_2D_tracew = df_2D[df_2D["ImageType"].str.contains("'ADC'")]
        return df_2D_tracew.index

class SystemSessionAVLPhilipsMarlinNoGeo(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir, force_geometry_imageset=ImageCatetory.T1_VFA)

    def get_geometric_series_UIDs(self):
        return None

    def get_t1_vir_series_UIDs(self):
        df_se_2D = super().get_2D_series()
        df_se_ir_2D = super().get_inversion_recovery_series(df_se_2D)
        df_se_ir_2D_MIR = df_se_ir_2D[df_se_ir_2D.ImageType.str.contains("M_IR", na=False)]
        df_se_ir_2D_MIR = df_se_ir_2D_MIR.sort_values(by=["InversionTime"])
        return df_se_ir_2D_MIR.index

    def get_t1_vfa_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_vfa = df_ge_3D[df_ge_3D.ImageType.str.contains("M_FFE", na=False)]
        df_ge_3D_vfa = df_ge_3D_vfa[df_ge_3D_vfa.SeriesDescription.str.contains("tFA", na=False)]
        df_ge_3D_vfa = df_ge_3D_vfa[df_ge_3D_vfa.SliceThickness <= 10]
        df_ge_3D_vfa = df_ge_3D_vfa.sort_values(by=["FlipAngle"])
        return df_ge_3D_vfa.index

    def get_t2_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        return df_se_2D.index

    def get_proton_density_series_UIDs(self):
        return None

    def get_t2star_series_UIDs(self):
        return None


class SystemSessionPhilipsIngeniaAmbitionX(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        df_ge_3D = super().get_3D_series()
        df_ge_3D_1mm = df_ge_3D[df_ge_3D.SliceThickness == 1]
        df_ge_3D_1mm_psn = df_ge_3D_1mm[df_ge_3D_1mm["SequenceName"].str.match(r"(?=.*\bT1FFE\b)") == True]
        return df_ge_3D_1mm_psn.index

    def get_t1_vir_series_UIDs(self):
        df_2D = super().get_2D_series()
        df_2D_thick = df_2D[df_2D.SliceThickness == 6]
        df_2D_psn = df_2D_thick[df_2D_thick["SequenceName"].str.match(r"(?=.*\bTIR\b)") == True]
        df_2D_psn = df_2D_psn[df_2D_psn["ImageType"].str.match(r"(?=.*\bM_IR\b)") == True]
        df_2D_psn = df_2D_psn.sort_values(["InversionTime", "RescaleIntercept"], ascending= [True,False])
        return df_2D_psn.index

    def get_t1_vfa_series_UIDs(self):
        df_3D = super().get_3D_series()
        df_3D_3mm = df_3D[df_3D.SliceThickness == 3]
        df_pulse = df_3D_3mm[df_3D_3mm["SequenceName"].str.match(r"(?=.*\bT1FFE\b)") == True]
        df_pulse = df_pulse.sort_values(by=["FlipAngle"])
        return df_pulse.index

    def get_t2_series_UIDs(self):
        df_2D = super().get_2D_series()
        df_2D_6mm = df_2D[df_2D.SliceThickness == 6]
        df_t2 = df_2D_6mm[df_2D_6mm["SequenceName"].str.match(r"(?=.*\bTSE\b)") == True]
        return df_t2.index

    def get_proton_density_series_UIDs(self):
        df_2D = super().get_2D_series()
        df_2D_6mm = df_2D[df_2D.SliceThickness == 6]
        df_pd = df_2D_6mm[df_2D_6mm["SequenceName"].str.match(r"(?=.*\bSE\b)") == True]
        return df_pd.index

    def get_t2star_series_UIDs(self):
        return None


class SystemSessionSiemensSkyra(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_1mm = df_ge_3D[df_ge_3D.SliceThickness == 1]
        #return df_ge_3D_1mm.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID.values.tolist()
        return df_ge_3D_1mm.index

    def get_t1_vir_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        df_se_ir_2D = super().get_inversion_recovery_series(df_se_2D)
        df_se_ir_2D = df_se_ir_2D.sort_values(by=["InversionTime"])
        #return df_se_ir_2D.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_se_ir_2D.index

    def get_t1_vfa_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_gt1mm = df_ge_3D[df_ge_3D.SliceThickness > 1]
        df_ge_3D_gt1mm = df_ge_3D_gt1mm.sort_values(by=["FlipAngle"])
        #return df_ge_3D_gt1mm.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_ge_3D_gt1mm.index

    def get_t2_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        # regex expecting a 2 digit integer
        df_t2 = df_se_2D[df_se_2D["SequenceName"].str.match(r"^\*se2d([0-9]){2}") == True]
        #return df_t2.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_t2.index

    def get_proton_density_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        # regex expecting a 2 digit integer
        df_pd = df_se_2D[df_se_2D["SequenceName"].str.match(r"^\*se2d1") == True]
        #return df_pd.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_pd.index

    def get_t2star_series_UIDs(self):
        return None




class SystemSessionPhilipsMarlin(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_1mm = df_ge_3D[np.isclose(df_ge_3D.SliceThickness, 0.98)]
        #return df_ge_3D_1mm.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID.values.tolist()
        return df_ge_3D_1mm.index

    def get_t1_vir_series_UIDs(self):
        df_se_2D = super().get_2D_series()
        df_se_ir_2D = super().get_inversion_recovery_series(df_se_2D)
        df_se_ir_2D_MIR = df_se_ir_2D[df_se_ir_2D.ImageType.str.contains("M_IR", na=False)]
        df_se_ir_2D_MIR = df_se_ir_2D_MIR.sort_values(by=["InversionTime"])
        #return df_se_ir_2D_MIR.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_se_ir_2D_MIR.index

    def get_t1_vfa_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_gt1mm = df_ge_3D[df_ge_3D.SliceThickness > 1]
        df_ge_3D_gt1mm = df_ge_3D_gt1mm.sort_values(by=["FlipAngle"])
        #return df_ge_3D_gt1mm.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_ge_3D_gt1mm.index

    def get_t2_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        df_t2 = df_se_2D[df_se_2D["SequenceVariant"].str.contains("SS", na=False)]
        #return df_t2.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_t2.index

    def get_proton_density_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        # regex expecting a 2 digit integer
        df_pd = df_se_2D[df_se_2D["SequenceName"].str.match(r"^\*se2d1") == True]
        #return df_pd.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
        return df_pd.index

    def get_t2star_series_UIDs(self):
        return None


class SystemSessionAucklandCAM(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir, force_geometry_imageset=ImageCatetory.T1_VFA)

    def get_geometric_series_UIDs(self):
        return None

    def get_t1_vir_series_UIDs(self):
        return None

    def get_t1_vfa_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_scanOpt = df_ge_3D[df_ge_3D["ScanOptions"].str.match(r"^PER") == True]
        df_ge_3D_scanOpt_orig = df_ge_3D_scanOpt[
            df_ge_3D_scanOpt["ImageType"].str.contains("'ORIGINAL'", regex=False) == True]
        # order by flip angle
        df_ge_3D_scanOpt_orig = df_ge_3D_scanOpt_orig.sort_values(by=["FlipAngle"])
        return df_ge_3D_scanOpt_orig.index

    def get_t2_series_UIDs(self):
        return None

    def get_proton_density_series_UIDs(self):
        return None

    def get_t2star_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_2D = super().get_2D_series(df_ge)
        df_ge_3D_multiEcho = df_ge_2D[df_ge_2D["SequenceName"].str.match(r"^\*fl2d12") == True]
        df_ge_3D_multiEcho_orig = df_ge_3D_multiEcho[df_ge_3D_multiEcho["ImageType"].str.contains("'ORIGINAL'", regex=False) == True]
        return df_ge_3D_multiEcho_orig.index


class SystemSessionSiemensSkyraErin(SystemSessionAbstract):
    def __init__(self, dicom_dir):
        super().__init__(dicom_dir)

    def get_geometric_series_UIDs(self):
        df_se = super().get_spin_echo_series()
        df_se_2D = super().get_2D_series(df_se)
        df_se_2D_lt4mm = df_se_2D[df_se_2D.SliceThickness < 4]
        return df_se_2D_lt4mm.index

    def get_t1_vir_series_UIDs(self):
        return None

    def get_t1_vfa_series_UIDs(self):
        df_ge = super().get_gradient_echo_series()
        df_ge_3D = super().get_3D_series(df_ge)
        df_ge_3D_orig = df_ge_3D[df_ge_3D["ImageType"].str.contains("'ORIGINAL'", regex=False) == True]
        df_ge_3D_orig_noWater = df_ge_3D_orig[df_ge_3D_orig["ImageType"].str.contains("'WATER'", regex=False) == False]
        df_ge_3D_orig_noWater = df_ge_3D_orig_noWater.sort_values(by=["FlipAngle"])
        return df_ge_3D_orig_noWater.index

    def get_t2_series_UIDs(self):
        return None

    def get_proton_density_series_UIDs(self):
        return None

    def get_t2star_series_UIDs(self):
        return None









if __name__ == "__main__":
    main()