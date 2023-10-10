*Authors: Stanley Norris &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Modified: 10/10/2023*

# Tutorial: Adding a new scanner or acquisition protocol
This tutorial will show you how to add a new scanner to the software, using the example of adding images taken with a [Philips Ingenia Ambition 1.5T X machine](https://www.philips.com.au/healthcare/product/HC781356/ingenia-ambition-excel-in-your-daily-mr-services-helium-free). We made these modifications for our collaborators from the Oliva Newton-John Cancer Wellness and Research Centre at Austin Health, who were interested in applying MRBIAS to their [diffusion phantom](https://qmri.com/product/diffusion-phantom/).

## Which parts of the software perform the image sorting?
The main script in the image sorting process is "scan_session.py", in the "mrbias" directory. In order to extend MRBIAS to a new scanner, you will need:
-	A folder of dicom images taken with your scanner, for example "new_data/Images"
-	A new configuration file in the "config" directory, for example "example_config_glen.yaml"
-	A new test script, for example "mr_bias_example_glen.py"

and you will make changes to the following files:
-	"mrbias/mrbias.py"
-	"mrbias/scan_session.py"

## The rough steps to adding the scanner
Ultimately, it is up to you to decide which properties of the images you will use to distinguish different scanning sequences. The problem with adding a new scanner to the software is that different scanners will produce image metadata in different ways, and this part is also important when extending the software to use completely different image sequences e.g., diffusion wieghted images. The rough steps are:
- Decide which DICOM metadata to use to distinguish image sets (e.g., geometric, T1, T2, T2* image sets)
- Check if this metadata is already extracted by MRBIAS
- If there is metadata that you need to use in the image sorting, which doesn't appear in the check, you will need to modify "mrbias/scan_session.py"
- Update the main MRBIAS pipeline

### Deciding which metadata to use to distinguish image sets
In the example below, the image sets are sufficiently distinguished by the following metadata DICOM tags:
- MRAcquisitionType
- SliceThickness
- ScanningTechnique
- ImageType

By using software such as ImageJ or 3DSlicer, you can inspect the metadata of each image set and deduce which tags will be used for sorting.

### Checking if MRBIAS already extracts the relevant metadata
You can check if the tags you are using are already extracted by "mrbias/scan_session.py" by running the script "utils/add_new_scanner_protocol.py". You will simply need to modify this script to point to your data directory.

```python
#################################################################################
# INSTRUCTIONS
#################################################################################
# Set the following two variables:
# -----------------------------------------------------------------------------
# - DICOM_DIRECTORY:     the directory of dicom images you want to search
# - OUTPUT_CSV_FILENAME: the file to write the dicom metadata to for viewing
#
# Check the results:
# -------------------------------------------------------------------------------
#   The script will output a comma seperated data file with the dicom metadata
#################################################################################
DICOM_DIRECTORY = os.path.join(base_dir, "new_data", "Images")
OUTPUT_CSV_FILENAME = "dicom_metadata.csv"
#################################################################################


```

You can now check to see if "mrbias/scan_session.py" extracts your tags by manually inspecting the output of this script in "utils/dicom_metadata.csv". If all the DICOM tags you want to use for sorting appear in this file, then skip the next step.

### Modify ScanSession to extract the relevant metadata
Unfortunately, the ScanningTechnique tag is not extracted by MRBIAS. Normally, you could simply modify "mrbias/scan_session.py", by adding the corresponding tag to the column_meta_names list:
```python
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
                             "SequenceName", "MagneticFieldStrength", "InversionTime", "ScanningTechnique", "DiffusionBValue"]
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
                for tag_name in column_meta_names[1:]: # skip the "ImageFilePath"
```
However, after this change is made the tag still does not appear in the "utils/dicom_metadata.csv" output. The reason for this is that MRBIAS is normally using this command to extract metadata
```python
available_tags = ds.dir()
```
But the ScanningTechnique tag is not extracted by this command. So, this special tag must be dealt with differently by adding special_tag
```python
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
```
Now, if the "utils/dicom_metadata.csv" output columns show all the relevant metadata tags, you are ready to move on to the next step.

### 
