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
07-July-2023  :               (James Korte) : Initial code to help users add new scanner protocols
"""

#################################################################################
# Set the following two variables:
# -----------------------------------------------------------------------------
# - DICOM_DIRECTORY:     the directory of dicom images you want to search
# - OUTPUT_CSV_FILENAME: the file to write the dicom metadata to for viewing
#
#################################################################################
DICOM_DIRECTORY = "I:\JK\MR-BIAS\Data_From_Hayley\Phantom_scans\Phantom_scans"
OUTPUT_CSV_FILENAME = "dicom_metadata.csv"




import numpy as np
import pandas as pd

# Code to add the parent directory to allow importing mrbias core modules
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
import mrbias.scan_session as ss



# # search the dicom directory and strip metadata to write in the output csv file
# scan_session = ss.DICOMSearch(target_dir=DICOM_DIRECTORY,
#                               output_csv_filename=OUTPUT_CSV_FILENAME)

# # sort the data to show a summary to help define sorting rules
# df = scan_session.get_df()
df = pd.read_csv(OUTPUT_CSV_FILENAME)
df_sum = df.drop_duplicates(subset=['SeriesInstanceUID'],
                            keep='last').reset_index(drop = True)
dicom_summary_tag_list = ['SeriesDescription', 'ProtocolName', 'SequenceName', 'ScanningSequence', 'ScanOptions', 'SequenceVariant', 'ImageType',
                          'MRAcquisitionType', 'SliceThickness', 'FlipAngle', 'EchoTime', 'RepetitionTime', 'InversionTime']
# reorder the columns so the useful ones are together at the start
col_name_list = df_sum.columns.values.tolist()
for d_tag in dicom_summary_tag_list:
    col_name_list.remove(d_tag)
df_sum = df_sum[dicom_summary_tag_list + col_name_list]
df_sum.to_csv("dicom_metadata_summary.csv")

param_str_list = []
for r_dx, r in df_sum.iterrows():
    p_list = []
    for param_name in dicom_summary_tag_list:
        p_list.append(str(r[param_name]))
    param_str_list.append(p_list)
param_str_list.append(dicom_summary_tag_list)
# calculate the max width for each column
param_arr = np.array(param_str_list)
n_cols = param_arr.shape[1]
col_widths = np.zeros(n_cols)
for col_dx in range(n_cols):
    cols_vals = param_arr[:, col_dx]
    for v in cols_vals:
        if len(v) > col_widths[col_dx]:
            col_widths[col_dx] = len(v)
param_str_list.pop() # remove the header information from the table contents
# output the summary table
table_width = int(np.sum(col_widths)) + n_cols*3 + 1
print("="*table_width)
p_str = "| "
for p_val, c_width in zip(dicom_summary_tag_list, col_widths):
    format_str = "%" + "%ds | " % c_width
    p_str += format_str % p_val
print(p_str)
print("="*table_width)
for p_list in param_str_list:
    p_str = "| "
    for p_val, c_width in zip(p_list, col_widths):
        format_str = "%" + "%ds | " % c_width
        p_str += format_str % p_val
    print(p_str)
print("="*table_width)



