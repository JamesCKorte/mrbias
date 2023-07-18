*Authors: James Korte &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Modified: 18/07/2023*

# Tutorial: Adding a new ROI detection template

So you want to add a new image for ROI detection?  This tutorial will show you how to do this using the example of adding the a template for the [system lite phantom or essential system phantom](https://qmri.com/product/essential-system/) (which is a simplified version of the [premium system phantom](https://qmri.com/product/ismrm-nist-premium-system-phantom/)). We made these modifications for our collaborators from the Auckland Bioengineering Institute at the University of Auckland, who wanted to try out MRBIAS on thier system lite phantom images.

## Where do the ROI detection templates live?
Each ROI detection template is stored in the “mrbias/roi_detection_templates” directory, with each template having its own subdirectory. For example, MR-BIAS comes with a premade ROI detection template for the system phantom which is in the folder “mrbias/roi_detection_templates/siemens_skyra_3T”. The templates require two things:
-	A sub-folder of dicom images (with a folder name "dicom")
-	ROI specification files which define the location and shape of the regions of interest
    - default_T1_rois.yaml
    - default_T2_rois.yaml
    
### The ROI specification files
These files specify the shape, location and name of each ROI and are written in the YAML format. For each ROI there is a block of code, starting with a label (i.e. t1_roi_1) and then followed by it’s attributes, such as its central coordinates (i.e. [129, 76, 73]) and shape details (such as “sphere” and “radius”). At this point the only ROI type is a sphere, feel free to contact us if you need another shape.
```yaml
t1_roi_1:
  roi_type: "sphere"
  roi_radius_mm: 5.0
  ctr_vox_coords: [129, 76, 73]

t1_roi_2:
  roi_type: "sphere"
  roi_radius_mm: 5.0
  ctr_vox_coords: [159,  86, 73]

t1_roi_3:
  roi_type: "sphere"
  roi_radius_mm: 5.0
  ctr_vox_coords: [177, 112, 72]
```

## How to define your own regions of interest
To define your regions of interest, open your dicom image in a viewing software such as [ImageJ](https://imagej.net/ij/download.html) or [FIJI](https://imagej.net/software/fiji/downloads). You can then navigate through the image and hover over the centre of a region of interest, such as the spherical region in our example and read a coordinate from the software. In FIJI you can see this as the voxel in the round brackets, in the screenshot you can read the location as [101, 71, 20]
 
Which will be entered into our T1_roi.yaml file as
```yaml
t1_roi_1:
  roi_type: "sphere"
  roi_radius_mm: 5.0
  ctr_vox_coords: [101, 71, 20]
```
 
If you now run our helper script from the “utils/add_new_roi_template.py” then you can check that your coordinates and the radius of the sphere are set correctly
 
And once all 15 ROIs have been added we have the following. 
 
### Matching the labels to the correct phantom ROI
It is important to follow a naming convention for the ROIs, as the software uses these to lookup the reference values which have been taken from manufacturers manuals. These will be used in percent bias calculations. For the system phantoms (premium and lite versions) we use the naming conventions visible in the screenshots (i.e. t1_roi_1 to t1_roi_14) 

## Testing your new ROI template
Once you are happy with the location of your ROIs, it is a good time to see if the template will work with your dataset. That is, if you are able to load up dataset with a geometric image and other image sets (i.e. a T1-VFA dataset) and accurately detect ROIs in the image set. To help with this we have written another helper script “utils/check_new_roi_template.py”
You will need to set two global variables, and modify the line of code which selects the correct scan session class for your dataset
```python
#################################################################################
# INSTRUCTIONS
#################################################################################
#
# Set the following variables:
# -----------------------------------------------------------------------------
# - ROI_TEMPLATE_NAME: the name of the sub-directory you have put the new ROI template in
#                    : this is expected to be a sub-directory of mrbias/roi_detection_templates
# - DICOM_DIRECTORY:     the directory of dicom images you want to detect ROIs on
#
#
# Create your new ScanSession<YourClassName> class for testing:
# --------------------------------------------------------------------------------
# i.e. scan_session = ss.ScanSessionYourClassName(DICOM_DIRECTORY)
#
#
# Check the results:
# -------------------------------------------------------------------------------
#   The script will log information to the terminal while it processes the data
# you can read this to check things are working correctly.
#   There is also a pdf generated ("check_roi_detect.pdf") which you can
# visually inspect to check the ROIs are in the correct locations
#################################################################################
ROI_TEMPLATE_NAME = "siemens_skyra_3p0T"
DICOM_DIRECTORY = os.path.join(base_dir, "data", "mrbias_testset_B")
scan_session = ss.ScanSessionSiemensSkyra(DICOM_DIRECTORY)
registration_method = roi_detect.RegistrationOptions.TWOSTAGE_MSMEGS_CORELGD # default
#################################################################################
```
 


### Adding the new template as an option for MRBIAS
A ROI detection template is selected in the configuration file
```python
# ====================================================================================================================
# ROI detection options
# ====================================================================================================================
roi_detection:
  template_name: "systemlite_siemens_vida_3p0T"   # options include [siemens_skyra_3p0T, systemlite_siemens_vida_3p0T]
  save_mask_images: False                         # future feature (not implemented)
```
This field (“template_name”) is used by the main MRBIAS program to select the desired template. We also need to add this to that program “mrbias/mri_bias.py”
```python
# ===================================================================================================
# Dectect the ROIs on each geometry image (only ones used in a fit)
# ===================================================================================================
mu.log("-" * 100, LogLevels.LOG_INFO)
mu.log("MR-BIAS::analyse() : Detect the ROIs on each geometry image ...", LogLevels.LOG_INFO)
mu.log("-" * 100, LogLevels.LOG_INFO)
roi_template = self.conf.get_roi_template()
roi_reg_method = self.conf.get_roi_registration_method()
roi_is_partial_fov = self.conf.get_roi_registration_partial_fov()
roi_template_dir = None
if roi_template == "siemens_skyra_3p0T":
    roi_template_dir = os.path.join(mu.reference_template_directory(), "siemens_skyra_3p0T")
elif roi_template == "systemlite_siemens_vida_3p0T":
    roi_template_dir = os.path.join(mu.reference_template_directory(), "systemlite_siemens_vida_3p0T")
# ... add others
if roi_template is None:
    mu.log("MR-BIAS::analyse(): skipping analysis as unknown 'roi_template' defined for ROI detection",
           LogLevels.LOG_ERROR)
    return None
```



