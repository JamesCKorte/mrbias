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
import tempfile

import os
import shutil
from collections import OrderedDict
from enum import IntEnum

import pydicom as dcm

import pandas as pd
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
from mrbias.image_sets import ImageGeometric, ImageProtonDensity, ImageSetT1VIR, ImageSetT1VFA, ImageSetT2MSE, ImageSetT2Star, ImageSetDW

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
    dcm_dir_d = "/Users/stanleynorris/Desktop/MRBIAS/new_data/Intial_SystemAndDiffusionDataset/DWI_Phantom/Images"
    dcm_dir_c = "/Users/stanleynorris/Desktop/MRBIAS/new_data/Intial_SystemAndDiffusionDataset/System_Phantom_with_CalibreAnalysis/Images"
    test_dcm_dir_list = [dcm_dir_a, dcm_dir_b, dcm_dir_c, dcm_dir_d]
    test_ss_config_list = ["SiemensSkyra", "SiemensSkyra", "PhilipsIngeniaAmbitionX", "DiffPhilipsIngeniaAmbitionX"]

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

        if ss_type == "DiffPhilipsIngeniaAmbitionX":
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
    SECONDARY = 8
    UNKNOWN = 9
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

        # force the scan session to use another type of image as the geometric image for ROI detection
        self.force_geometry_imageset = force_geometry_imageset

        # search the dicom directory and strip tags to populate a metadata dataframe
        self.dicom_searcher = DICOMSearch(dicom_dir)
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
                                       self.get_dw_series_UIDs()]

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
        geom_options.remove('')
        if len(geom_options):
            current_geom = geom_options[0]
            for idx, r in df.iterrows():
                if r.IsGeometric and (r.GeomSet != ''):
                    current_geom = r.GeomSet
                # apply the reference geometry link to the main dataframe
                self.meta_data_df.loc[self.meta_data_df["SeriesInstanceUID"].isin([r.SeriesInstanceUID]), "ReferenceGeometryImage"] = current_geom

        # output the categorisation, set grouping & reference images
        self.log_meta_dataframe()

        # todo: remove (here for debugging new scansession classes)
        #self.meta_data_df.to_csv("check_labeling.csv")

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

    # output the categorisation, set grouping & reference images (via the log)
    def log_meta_dataframe(self, show_unknown=True):
        df = self.meta_data_df.copy(deep=True)
        if show_unknown is False:
            df['Category'].replace(IMAGE_UNKNOWN_STR, np.nan, inplace=True)
            df.dropna(subset=['Category'], inplace=True)
        df = df.drop_duplicates(subset=["SeriesInstanceUID", "Category"])
        table_width = 182
        mu.log("=" * table_width, LogLevels.LOG_INFO)
        mu.log(
            "| DATE    | TIME    | DESCRIPTION                    | CATEGORY      | IMAGE SET     | GEOM SET  | REF GEOM.     | SERIES_UID                                                        |",
            LogLevels.LOG_INFO)
        mu.log("=" * table_width, LogLevels.LOG_INFO)
        cur_image_set = ""
        for idx, r in df.iterrows():
            if (r.ImageSet == "") or ((r.ImageSet != cur_image_set) and (not (r.Category in [IMAGE_UNKNOWN_STR]))):
                mu.log("-" * table_width, LogLevels.LOG_INFO)
                cur_image_set = r.ImageSet
            mu.log("| %s | %6s | %30s | %13s | %13s | %9s | %13s | %65s |" %
                   (r.SeriesDate, str(r.SeriesTime).split(".")[0], r.SeriesDescription,
                    r.Category, r.ImageSet, r.GeomSet, r.ReferenceGeometryImage, r.SeriesInstanceUID),
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
            geom_options.remove('')
            geomset_series_uid_list = df[df["GeomSet"].isin(geom_options)].SeriesInstanceUID
            geom_image_list = []
            for series_instance_uid, geoset_label in zip(geomset_series_uid_list, geom_options):
                image_set_df = self.meta_data_df[self.meta_data_df.SeriesInstanceUID == series_instance_uid]
                set_series_uids = image_set_df.drop_duplicates(subset=["SeriesInstanceUID"]).SeriesInstanceUID
                # get the datatype details
                bits_allocated = image_set_df.drop_duplicates(subset=["BitsAllocated"]).BitsAllocated.iloc[0]
                bits_stored = image_set_df.drop_duplicates(subset=["BitsStored"]).BitsStored.iloc[0]
                rescale_slope = image_set_df.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.iloc[0]
                rescale_intercept = image_set_df.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.iloc[0]
                # get all the slices
                files_sorting = []
                for index, row in image_set_df.iterrows():
                    files_sorting.append((row["ImageFilePath"], row["SliceLocation"]))
                # sort the slices by slice location before reading by sitk
                files_sorting.sort(key=lambda x: x[1])
                files_sorted = [x[0] for x in files_sorting]
                im = self._load_image_from_filelist(files_sorted, series_instance_uid,
                                                     rescale_slope, rescale_intercept,
                                                     row.SeriesDescription)
                geom_image_list.append(ImageGeometric(geoset_label,
                                                      im,
                                                      series_instance_uid,
                                                      bits_allocated, bits_stored,
                                                      rescale_slope, rescale_intercept))
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
            self.t2star_imageset_list = self.__get_echo_image_sets(ImageCatetory.T2STAR_ME)
        return self.t2star_imageset_list


    def _get_image_sets(self, imageset_data_dict, category, n_expected_dcm_tags):
        # acquisition timestamp
        study_date, study_time = self.get_study_date_time()
        imageset_list = []
        for imageset_name, (image_and_metadata_list, ref_geom_image) in imageset_data_dict.items():
            images, metadata_list, other_list = zip(*image_and_metadata_list)
            series_instance_uids = [x["SeriesInstanceUID"] for x in other_list]
            bits_allocated = other_list[0]["BitsAllocated"]
            bits_stored = other_list[0]["BitsStored"]
            rescale_slope = other_list[0]["RescaleSlope"]
            rescale_intercept = other_list[0]["RescaleIntercept"]
            # scanner details
            scanner_make = other_list[0]["Manufacturer"]
            scanner_model = other_list[0]["ManufacturerModelName"]
            scanner_sn = other_list[0]["DeviceSerialNumber"]
            scanner_field_strength = other_list[0]["MagneticFieldStrength"]
            # pull out the metadata (inversion recovery times, flip angles, echo times etc.)
            assert len(metadata_list[0]) == n_expected_dcm_tags, \
                "ScanSessionAbstract::_get_image_sets() - %s should have %d dicom parameters per image, found %d" % (
                category, len(
                    metadata_list[0]), n_expected_dcm_tags)
            # handle the different types of image sets
            image_set = None
            if category == ImageCatetory.T1_VFA:
                flip_angle_list = [x[0] for x in metadata_list]
                repetition_time_list = [x[1] for x in metadata_list]
                # sort the images by the inversion recovery time
                flip_and_image_list = [(fa, tr, im, suid) for im, fa, tr, suid in
                                       zip(images, flip_angle_list, repetition_time_list, series_instance_uids)]
                flip_and_image_list.sort(key=lambda x: x[0])
                flip_angle_list, repetition_time_list, images, series_instance_uids = zip(*flip_and_image_list)
                image_set = ImageSetT1VFA(imageset_name,
                                          images,
                                          flip_angle_list,
                                          repetition_time_list,
                                          ref_geom_image,
                                          series_instance_uids,
                                          bits_allocated, bits_stored,
                                          rescale_slope, rescale_intercept,
                                          scanner_make, scanner_model, scanner_sn, scanner_field_strength,
                                          study_date, study_time)
            elif category == ImageCatetory.T1_VIR:
                inversion_time_list = [x[0] for x in metadata_list]
                repetition_time_list = [x[1] for x in metadata_list]
                # sort the images by the inversion recovery time
                tir_time_and_image_list = [(tir, tr, im, suid) for im, tir, tr, suid in
                                           zip(images, inversion_time_list, repetition_time_list,
                                               series_instance_uids)]
                tir_time_and_image_list.sort(key=lambda x: x[0])
                inversion_time_list, repetition_time_list, images, series_instance_uids = zip(
                    *tir_time_and_image_list)
                # create the imageset object and append to return list
                image_set = ImageSetT1VIR(imageset_name,
                                          images,
                                          inversion_time_list,
                                          repetition_time_list,
                                          ref_geom_image,
                                          series_instance_uids,
                                          bits_allocated, bits_stored,
                                          rescale_slope, rescale_intercept,
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
            # get the datatype details
            bits_allocated = df_t2.drop_duplicates(subset=["BitsAllocated"]).BitsAllocated.iloc[0]
            bits_stored = df_t2.drop_duplicates(subset=["BitsStored"]).BitsStored.iloc[0]
            rescale_slope = df_t2.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.iloc[0]
            rescale_intercept = df_t2.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.iloc[0]
            # scanner details
            scanner_make = df_t2.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
            scanner_model = df_t2.drop_duplicates(subset=["ManufacturerModelName"]).ManufacturerModelName.iloc[0]
            scanner_sn = df_t2.drop_duplicates(subset=["DeviceSerialNumber"]).DeviceSerialNumber.iloc[0]
            scanner_field_strength = df_t2.drop_duplicates(subset=["MagneticFieldStrength"]).MagneticFieldStrength.iloc[0]

            # sort the echos into groups (to handle multiple slices)
            image_echo_time_list = []
            for index, row in df_t2.iterrows():
                image_echo_time_list.append((float(row["EchoTime"]), float(row["RepetitionTime"]),
                                             row["SeriesInstanceUID"], row["ImageFilePath"], row["SliceLocation"]))
            image_echo_time_list.sort(key=lambda x: x[0])
            echo_time_list, repetition_time_list, series_instance_uids, image_filepaths, slice_locations = zip(
                *image_echo_time_list)
            # load up each of the echo images
            unique_echo_list = pd.unique(echo_time_list)
            df_echo_images = pd.DataFrame(
                columns=["EchoTime", "RepetitionTime", "SeriesUID", "ImageFilePath", "SliceLocation"],
                data=image_echo_time_list)
            image_list = []
            for echo_time in unique_echo_list:
                series_uid = df_echo_images[df_echo_images["EchoTime"] == echo_time].SeriesUID.values[0]
                image_file_path_list = df_echo_images[df_echo_images["EchoTime"] == echo_time].ImageFilePath
                image_slice_location_list = df_echo_images[df_echo_images["EchoTime"] == echo_time].SliceLocation
                files_sorting = [(fp, slice_loc) for fp, slice_loc in
                                 zip(image_file_path_list, image_slice_location_list)]
                # sort the slices by slice location before reading by sitk
                files_sorting.sort(key=lambda x: x[1])
                files_sorted = [x[0] for x in files_sorting]

                im = self._load_image_from_filelist(files_sorted, series_uid,
                                                    rescale_slope, rescale_intercept)
                image_list.append(im)

            # include any associated geometry image
            ref_geom_image = None
            image_set_df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
            ref_geom_label = image_set_df[image_set_df.ImageSet == set_name].ReferenceGeometryImage.iloc[0]
            for g_im in geom_images:
                if g_im.get_label() == ref_geom_label:
                    ref_geom_image = g_im
            # create the appropriate ImageSet based on category
            if category == ImageCatetory.T2_MSE:
                imageset_list.append(ImageSetT2MSE(set_name,
                                                   image_list,
                                                   unique_echo_list,
                                                   repetition_time_list,
                                                   ref_geom_image,
                                                   series_instance_uids,
                                                   bits_allocated, bits_stored,
                                                   rescale_slope, rescale_intercept,
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
                                                    bits_allocated, bits_stored,
                                                    rescale_slope, rescale_intercept,
                                                    scanner_make, scanner_model, scanner_sn,
                                                    scanner_field_strength,
                                                    study_date, study_time))
            else:
                assert False, "ScanSessionAbstract::_get_echo_image_sets() is only designed to be called by T1-VFA and T1-VIR models, not (%s) imageset category" % category
        return imageset_list

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
            # get the datatype details
            bits_allocated = image_set_df.drop_duplicates(subset=["BitsAllocated"]).BitsAllocated.iloc[0]
            bits_stored = image_set_df.drop_duplicates(subset=["BitsStored"]).BitsStored.iloc[0]
            rescale_slope = image_set_df.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.iloc[0]
            rescale_intercept = image_set_df.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.iloc[0]
            # get the scanner details
            scanner_make = image_set_df.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
            scanner_model = image_set_df.drop_duplicates(subset=["ManufacturerModelName"]).ManufacturerModelName.iloc[0]
            scanner_sn = image_set_df.drop_duplicates(subset=["DeviceSerialNumber"]).DeviceSerialNumber.iloc[0]
            scanner_field_strength = image_set_df.drop_duplicates(subset=["MagneticFieldStrength"]).MagneticFieldStrength.iloc[0]

            # loop over the image slices
            image_and_metadata_list = []
            for series_uid in set_series_uids:
                files_sorting = []
                for index, row in image_set_df[image_set_df.SeriesInstanceUID == series_uid].iterrows():
                    files_sorting.append((row["ImageFilePath"], row["SliceLocation"]))
                # take the meta data from the last slice file/row
                dicom_meta_list = []
                if category_dicom_tag_list is not None:
                    for dcm_tag in category_dicom_tag_list:
                        dicom_meta_list.append(row[dcm_tag])
                # sort the slices by slice location before reading by sitk
                files_sorting.sort(key=lambda x: x[1])
                files_sorted = [x[0] for x in files_sorting]

                im = self._load_image_from_filelist(files_sorted, series_uid,
                                                     rescale_slope, rescale_intercept,
                                                     row.SeriesDescription)

                # group the image and metadata together
                image_and_metadata_list.append((im,  # image
                                                tuple(dicom_meta_list),  # image set parameters (i.e. flip angle)
                                                {"SeriesInstanceUID": row.SeriesInstanceUID,
                                                 "BitsAllocated": bits_allocated,
                                                 "BitsStored": bits_stored,
                                                 "RescaleSlope": rescale_slope,
                                                 "RescaleIntercept": rescale_intercept,
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

    def _load_image_from_filelist(self, files_sorted, series_uid,
                                   rescale_slope, rescale_intercept,
                                   series_descrp=None):
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
            im = reader.Execute()
            # undo the scaling to get the raw values
            assert isinstance(rescale_slope, float) and isinstance(rescale_intercept, float), \
                "Scale[%s]/Intercept[%s] not floats" % (str(rescale_slope), str(rescale_intercept))
            if np.isnan(rescale_slope) or np.isnan(rescale_intercept):
                mu.log("ScanSession::_load_image_from_filelist(): Scale[%s]/Intercept[%s] are invalid"
                       " setting them to [m=1.0, c=0.0]" % (str(rescale_slope), str(rescale_intercept)),
                       LogLevels.LOG_WARNING)
                rescale_slope = 1.0
                rescale_intercept = 0.0
            im = self._rescale_image_to_raw(im, rescale_slope, rescale_intercept,
                                             series_decript=series_descrp)
        return im

    def _rescale_image_to_raw(self, im, rescale_slope, rescale_intercept, series_decript=None):
        if not (np.isclose(rescale_slope, 1.0) and np.isclose(rescale_intercept, 0.0)):
            mu.log("ScanSession::_get_imageset_data_from_df(%s): rescaling image back to raw data" %
                   series_decript, LogLevels.LOG_INFO)
            im_arr = sitk.GetArrayFromImage(im)
            im_arr = (im_arr - rescale_intercept) / rescale_slope
            im_arr = im_arr.astype(np.uint16)
            im_raw = sitk.GetImageFromArray(im_arr)
            im_raw.SetOrigin(im.GetOrigin())
            im_raw.SetSpacing(im.GetSpacing())
            im_raw.SetDirection(im.GetDirection())
            return im_raw
        return im

class ScanSessionTemplate(ScanSessionAbstract):
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

    def get_dw_series_UIDs(self):
        return None

class SystemSessionAbstract(ScanSessionAbstract):
    def __init__(self, dicom_dir, force_geometry_imageset=None):
        super().__init__(dicom_dir, force_geometry_imageset)

    def get_dw_series_UIDs(self):
        return None

class DiffusionSessionAbstract(ScanSessionAbstract):
    def __init__(self, dicom_dir, force_geometry_imageset=None):
        super().__init__(dicom_dir, force_geometry_imageset)

    def get_dw_image_sets(self):
        if self.dw_imageset_list is None:
            mu.log("ScanSession::get_dw_image_sets(): "
                   "loading DW image sets from disk...", LogLevels.LOG_INFO)
            self.dw_imageset_list = self.__get_dw_image_sets(ImageCatetory.DW)
        return self.dw_imageset_list

    def __get_dw_image_sets(self, category):
        imageset_list = []
        # acquisition timestamp
        study_date, study_time = self.get_study_date_time()
        # geometric images for linking
        geom_images = self.get_geometric_images()
        # get image sets from the dataframe
        category_name = IMAGE_CAT_STR_DICT[category]
        cat_df = self.meta_data_df[self.meta_data_df.Category == category_name]
        imageset_names = cat_df.drop_duplicates(subset=["ImageSet"]).ImageSet
        # loop over the bval sets
        for set_name in imageset_names:
            df_b = cat_df[cat_df.ImageSet == set_name]
            df_b = df_b.sort_values(by=["DiffusionBValue"])
            # get the datatype details
            bits_allocated = df_b.drop_duplicates(subset=["BitsAllocated"]).BitsAllocated.iloc[0]
            bits_stored = df_b.drop_duplicates(subset=["BitsStored"]).BitsStored.iloc[0]
            rescale_slope = df_b.drop_duplicates(subset=["RescaleSlope"]).RescaleSlope.iloc[0]
            rescale_intercept = df_b.drop_duplicates(subset=["RescaleIntercept"]).RescaleIntercept.iloc[0]
            # scanner details
            scanner_make = df_b.drop_duplicates(subset=["Manufacturer"]).Manufacturer.iloc[0]
            scanner_model = df_b.drop_duplicates(subset=["ManufacturerModelName"]).ManufacturerModelName.iloc[0]
            scanner_sn = df_b.drop_duplicates(subset=["DeviceSerialNumber"]).DeviceSerialNumber.iloc[0]
            scanner_field_strength = df_b.drop_duplicates(subset=["MagneticFieldStrength"]).MagneticFieldStrength.iloc[0]

            # sort the echos into groups (to handle multiple slices)
            image_b_val_list = []
            for index, row in df_b.iterrows():
                image_b_val_list.append((float(row["DiffusionBValue"]), float(row["RepetitionTime"]),
                                             row["SeriesInstanceUID"], row["ImageFilePath"], row["SliceLocation"]))
            image_b_val_list.sort(key=lambda x: x[0])
            b_value_list, repetition_time_list, series_instance_uids, image_filepaths, slice_locations = zip(
                *image_b_val_list)
            # load up each of the echo images
            unique_bval_list = pd.unique(b_value_list)
            df_bval_images = pd.DataFrame(
                columns=["DiffusionBValue", "RepetitionTime", "SeriesUID", "ImageFilePath", "SliceLocation"],
                data=image_b_val_list)
            image_list = []
            for bval in unique_bval_list:
                series_uid = df_bval_images[df_bval_images["DiffusionBValue"] == bval].SeriesUID.values[0]
                image_file_path_list = df_bval_images[df_bval_images["DiffusionBValue"] == bval].ImageFilePath
                image_slice_location_list = df_bval_images[df_bval_images["DiffusionBValue"] == bval].SliceLocation
                files_sorting = [(fp, slice_loc) for fp, slice_loc in
                                 zip(image_file_path_list, image_slice_location_list)]
                # sort the slices by slice location before reading by sitk
                files_sorting.sort(key=lambda x: x[1])
                files_sorted = [x[0] for x in files_sorting]

                im = self._load_image_from_filelist(files_sorted, series_uid,
                                                        rescale_slope, rescale_intercept)
                image_list.append(im)

            # include any associated geometry image
            ref_geom_image = None
            image_set_df = self.meta_data_df.drop_duplicates(subset=["ImageSet"])
            ref_geom_label = image_set_df[image_set_df.ImageSet == set_name].ReferenceGeometryImage.iloc[0]
            for g_im in geom_images:
                if g_im.get_label() == ref_geom_label:
                    ref_geom_image = g_im
            # create the appropriate ImageSet based on category
            if category == ImageCatetory.DW:
                imageset_list.append(ImageSetDW(set_name,
                                                   image_list,
                                                   unique_bval_list,
                                                   repetition_time_list,
                                                   ref_geom_image,
                                                   series_instance_uids,
                                                   bits_allocated, bits_stored,
                                                   rescale_slope, rescale_intercept,
                                                   scanner_make, scanner_model, scanner_sn,
                                                   scanner_field_strength,
                                                   study_date, study_time))
            else:
                assert False, "ScanSessionAbstract::_get_dw_image_sets() is only designed to be called by DWI models, not (%s) imageset category" % category
        return imageset_list

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
        df_2D_psn = df_2D_psn.sort_values(["DiffusionBValue", "RescaleIntercept"], ascending= [True,False])
        return df_2D_psn.index



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


class SystemSessionAucklandCAM(ScanSessionAbstract):
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


class ScanSessionSiemensSkyraErin(ScanSessionAbstract):
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






# Which tags are important?
# ============================================
# ScanningSequence : is used to determine if a spin-echo or a gradient echo sequence
# MRAcquisitionType : is used to determine 2D/3D
class DICOMSearch(object):
    def __init__(self, target_dir, read_single_file=False, output_csv_filename=None):
        mu.log("DICOMSearch::init(): searching target DCM dir: %s" % target_dir,
               LogLevels.LOG_INFO)
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
                pass # print(e, filepath)
            except AttributeError as e:
                pass # print(e, filepath)
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
                mu.log("'DICOMSearch::__init()__: skipping non-image file :%s" % fpath,
                       LogLevels.LOG_WARNING)
        mu.log('DICOMSearch::init():  search complete!  %d image sets with Unique IDs found' %
               len(dicom_dict.keys()),
               LogLevels.LOG_INFO)
        # create a pandas dataframe from selected DICOM metadata
        special_tag = dcm.tag.Tag(0x2001, 0x1020)
        dicom_data = []
        column_meta_names = ['ImageFilePath', 'ImageType', 'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
                             'StudyDate', 'StudyTime', 'StudyDescription', 'StudyInstanceUID',
                             'InstitutionName', 'InstitutionAddress', 'InstitutionalDepartmentName',
                             'Modality', 'Manufacturer', 'ManufacturerModelName', 'DeviceSerialNumber',
                             'SeriesDate', 'SeriesTime', 'SeriesDescription', 'ProtocolName',
                             'SeriesInstanceUID', 'SeriesNumber', 'AcquisitionDate', 'AcquisitionTime',
                             'BitsAllocated', 'BitsStored', 'ScanningSequence', 'ScanOptions',
                             'RescaleSlope', 'RescaleIntercept',
                             'SequenceVariant', 'MRAcquisitionType',
                             'SliceThickness', 'FlipAngle',
                             'EchoTime', 'EchoNumbers', 'RepetitionTime', 'PixelBandwidth',
                             'NumberOfPhaseEncodingSteps', 'PercentSampling', 'SliceLocation',
                             "SequenceName", "MagneticFieldStrength", "InversionTime", "DiffusionBValue"]
        alternatives_dict = {"SequenceName" : [special_tag, "PulseSequenceName"] }
                             # "ScanningSequence": ["EchoPulseSequence"], #[EchoPulseSequence"],
                             # "MRAcquisitionType": ["VolumetricProperties"],
                             # "AcquisitionDate": ["InstanceCreationDate", "ContentDate"],
                             # "AcquisitionTime": ["InstanceCreationTime", "ContentTime"]}
        for UID, ds_filepaths in dicom_dict.items():
            mu.log('         Parsing DCM file (%s) / SeriesInstanceUID: %s' %
                   (ds_filepaths[0][0].SeriesDescription, UID),
                   LogLevels.LOG_INFO)
            for ds, filepath in ds_filepaths:
                data_row = [filepath]
                available_tags = ds.dir()
                if special_tag in ds.keys():
                    available_tags.append(special_tag)
                for tag_name in column_meta_names[1:]: # skip the "ImageFilePath"
                    if tag_name in available_tags:
                        data_row.append(ds[tag_name].value)
                    else:
                        # search for alternatives
                        alt_found = False
                        if tag_name in alternatives_dict.keys():
                            for alt_tag_name in alternatives_dict[tag_name]:
                                if alt_tag_name in available_tags:
                                    mu.log("DICOMSearch::__init__(): missing dicom tag (%s) in file (%s)" %
                                           (tag_name, filepath), LogLevels.LOG_WARNING)
                                    mu.log("DICOMSearch::__init__():   ...   using alternative tag (%s) with value (%s)" %
                                           (alt_tag_name, ds[alt_tag_name]), LogLevels.LOG_WARNING)
                                    data_row.append(ds[alt_tag_name].value)
                                    alt_found = True
                                    break
                        if not alt_found:
                            data_row.append(None)  # if doesn't exist so data field is left blank
                            if not (tag_name in ["InversionTime", "RescaleSlope", "RescaleIntercept", "PulseSequenceName", "SequenceName"]):
                                mu.log("DICOMSearch::__init__(): unable to locate dicom tag (%s) in file (%s)" %
                                       (tag_name, filepath), LogLevels.LOG_WARNING)
                                #     for d in available_tags:
                                #         print("----> ", d, "  : ",  ds[d])
                                #     assert False
                dicom_data.append(data_row)
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
        mu.log('DICOMSearch::save_df() to %s' % df_filename,
               LogLevels.LOG_INFO)
        self.df.to_csv(df_filename)



if __name__ == "__main__":
    main()