*Authors: James Korte &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Modified: 12/03/2025*

# Tutorial: How to analyse a relaxometry dataset
This tutorial demonstrates how to estimate T<sub>1</sub> and T<sub>2</sub> relaxation times from MR images of a ISMRM/NIST system phantom. The tutorial assumes that you already have MR-BIAS installed and working on your system, [installation instructions can be found here](./how_to_install.md). The tutorial has the following major sections
- [Details of relaxometry datasets](./basic_analysis_relaxometry.md#details-of-relaxometry-datasets)
- [Run the analysis](./basic_analysis_relaxometry.md#run-the-analysis)
- [How does the automated analysis work?](./basic_analysis_relaxometry.md#how-does-the-automated-analysis-work)
- [Expected output from analysis](./basic_analysis_relaxometry.md#expected-output-from-analysis)
- [Running the software on your own data](./basic_analysis_relaxometry.md#running-the-software-on-your-own-data)
<br><br>

## Details of relaxometry datasets
The DICOM datasets used in this tutorial are included in the MR-BIAS repository ([mrbias_testset_A](https://github.com/JamesCKorte/mrbias/tree/main/data/mrbias_testset_A), [mrbias_testset_B](https://github.com/JamesCKorte/mrbias/tree/main/data/mrbias_testset_B)). The two datasets were acquired on a 3T Siemens scanner with the system phantom at room temperature, and include the following images:
* A geometric image, which is used for region of interest (ROI) detection 
* A series of images with variable flip angles (VFA), for quantification of T<sub>1</sub> relaxation times
* A series of images with variable inversion recovery times (VIR), for quantification of T<sub>1</sub> relaxation times
* A series of images taken at multiple spin-echo times (MSE), for quantification of T<sub>2</sub> relaxation times

## Run the analysis
We have provided an example script to analyse the relaxometry data. To run this example script, navigate to your cloned version of the MR-BIAS repository, then into the "examples" folder, and launch the script with the following commands:
```
cd examples
python relaxometry_example_1.py
```
You should see some text scrolling in your terminal/console window when you run the second command. This is a log of the analysis and provides some realtime feedback of the analysis as it is running. Each line of the log is tagged to let you know the context of the message, for example:
* [INFO] provides general information
* [WARNING] the software has found something slightly out of the ordinary but has been able to continue
* [ERROR] the software has run into something obviously incorrect, missing, and that it may be unable to continue 

The relaxometry_example_1.py script will sequentially analyse the two datasets, this will take a few minutes depending on your computer specs. While the analysis is running, you can learn a bit more about MR-BIAS, how it processes the images, estimates relaxometry times, and how to check the analysis has run successfully.

## How does the automated analysis work?
The automated analysis is described in detail in [the original MR-BIAS publication](https://doi.org/10.1088/1361-6560/acbcbb) and in [a presentation at the ESMRMB MRITogether conference (2023)](https://www.youtube.com/watch?v=QgFzDnjO4Jw&list=PLeDygc8TN_J65c0jM0ms__diTMylbEk9l&index=14&t=18m14s). The major steps in the analysis (Figure 1) include image sorting, ROI detection, model fitting, and reporting the results. The input to the analysis is a configuration file (.yaml) and a directory of DICOM images to analyse. The output of the analysis is a textual log file, a visual summary of the analysis (.pdf), and a comma separated value file (.csv) of the estimated T<sub>1</sub> and T<sub>2</sub> times. We will provide some guidance in the following sections on how to interpret the log file and the PDF report to check the major steps of the analysis.


<figure>
<img src="./assets/MRBIAS_workflow.png" height="800" alt="A diagram of the MR-BIAS automated workflow. Shows the main steps of the workflow which include image sorting, ROI detection, model fitting, and the reporting of results.">
  <figcaption><b>Figure 1:</b>  <i>The MR-BIAS automated workflow requires two inputs; a directory of images (DICOM format) to analyse and a
configuration file (.yaml format) to control the image analysis. Images are sorted into geometric images for ROI detection and T<sub>1</sub> and
T<sub>2</sub> image sets for fitting of relaxation models. The software has three outputs; a text based log of the analysis, a PDF formatted visual
summary and a comma separated value file of the model fitting results. This figure is taken from the original MR-BIAS publication (https://doi.org/10.1088/1361-6560/acbcbb)</i></figcaption>
</figure>


### Image sorting
To find the datasets required for relaxometry analysis the software scans the DICOM directory and extracts image metadata. The extracted meta-data is used to categorise the images into types, such as a geometric image for ROI detection, of a T<sub>1</sub> VFA series for relaxation time estimation. The results of this automatic image sorting are summarised in the logfile, the PDF report, and printed to the terminal during execution of the script.

A snippet of the log printed to the terminal (Figure 2) shows a summary of the image sorting for [mrbias_testset_A](https://github.com/JamesCKorte/mrbias/tree/main/data/mrbias_testset_A). The summary table is ordered by series number and displays other information such as the date, time, and description for each series. Each series is detected as a specific category (geometric, t1_vir, t1_vfa, t2_mse) and images of the same category are grouped together into image sets, for example all the different flip angle images of a VFA image set. The reference geometry (REF. GEOM) shows which geometry image will be used for ROI detection for each image series. The series UID is also provided for each series in case you want to manually verify the correct images are being analysed.

<figure>
  <img src="./assets/relaxometry_testsetA_sorting_log.PNG" alt="A snippet from the log printed during the terminal during MR-BIAS analysis, showing a summary of the detected images and their respective categories.">
  <figcaption><b>Figure 2:</b>  <i>The image sorting summary logged to the terminal during execution of MR-BIAS. Each image series is detected as a category and related image series are also grouped into imagesets (i.e. all the flip angle images of a VFA dataset). A similar table is also available in the PDF report.</i></figcaption>
</figure>

### ROI detection

The PDF report includes a visual summary of the ROI detection (Figure 3). The visual summary includes the template image and template ROIs for both T<sub>1</sub> and T<sub>2</sub> regions of the system phantom. There is a checkerboard display for the unregistered and registered geometric image. In the provided example (Figure 3) you can see in the unregistered geometric image that we have deliberately rotated the phantom by a large angle for testing purposes. In the registered images, it is clear that the ROI detection has been able to correct the phantom rotation. Finally, the template ROIs have been transformed onto the target image (Figure 3, right), visual inspection of these allows us to conclude that the ROI detection looks reasonable as the coloured circles are located within the circular regions of the phantom.
<figure>
  <img src="./assets/roi_detection_sysphan_summary.PNG" alt="TODO">
  <figcaption><b>Figure 3:</b>  <i>a visual summary of the ROI detection performed on a system phantom. The figure includes (top row) T<sub>1</sub> images and (bottom row) T<sub>2</sub> images. The (left) template used for ROI detection consists of an image and associated ROIs. The (central) unregistered and registered target images are shown in a checkerboard with the template image to assess the rigid registration. The (right) template ROIs transformed onto the target geometric image are used to visually assess the quality of the ROI detection.</i></figcaption>
</figure>
<br><br>

The ROI detection is performed on the geometric image, and the result of that process (a rigid transform) is used to transform the template ROIs onto each image used for model fitting. To check the accuracy of ROI detection, a summary for each image set is provided in the PDF report. An example of this is provided for a variable flip angle image set (Figure 4). A zoomed view is shown for each ROI, multiple planes (axial and sagittal) are shown for 3D image sets to help visually assess the quality of the ROI detection.
<figure>
  <img src="./assets/roi_detection_sysphan_detail.PNG" alt="TODO">
  <figcaption><b>Figure 4:</b>  <i>A summary of detected ROIs as transformed onto a variable flip angle image. The target image (greyscale) has an overlay of the detected (colours) regions of interest.</i></figcaption>
</figure>

### Model fitting

The software will fit a signal model, or in this case a model of relaxation, to the measured signal in each region of interest across all images in an imageset. For example a T<sub>1</sub> relaxation time has been estimated in each ROI of the variable flip angle imageset using a two parameter signal model. A summary table of the estimated relaxation times is provided in the PDF report (Figure 5) and is also output in a datafile (.csv) if further analysis is desired. The signal equation is detailed in the PDF report (Figure 5) with a description of the parameters, their initial values, and their bounds during the optimisation based curve fitting.
<figure>
  <img src="./assets/model_fitting_vfa_table.PNG" alt="TODO">
  <figcaption><b>Figure 5:</b>  <i>A summary table of the estimated parameters of a T<sub>1</sub> relaxation model on a variable flip angle (VFA) image set. The ground truth NMR reference values for T<sub>1</sub> relaxation times, as provided by the phantom manufacturor are show in the column 'T1_ref'. The signal equation is detailed below the summary table, including descriptions of the free parameters, thier initial values, and thier bounds during optimisation based curve fitting.</i></figcaption>
</figure>
<br><br>

The curves resulting from the model fit are plotted against the measured data (Figure 6) to allow visual inspection of the quality of the curve fit. Here we can see the (coloured line) curve fit is reasonable in comparison to the (coloured dots) measured data from the images. The (black dots) are measurement points which we have manually exlcuded, in this case images with a flip angle of 15 degrees. The (coloured vertical bars on the dots) standard deviation of the measurement within an ROI for each measurement point are also shown.
<figure>
  <img src="./assets/model_fitting_vfa_detail.PNG" alt="TODO">
  <figcaption><b>Figure 6:</b>  <i>A graphical summary of the signal models as fit to the measured image data for a T<sub>1</sub> variable flip angle (VFA) image set</i></figcaption>
</figure>
<br><br>

A summary of %bias, the deviation of the measured T<sub>1</sub> values in comparison to the NMR reference values provided by the phantom manufacturer (Figure 7) is provided in the PDF report. This provides a quick visual summary of accuracy of estimated T<sub>1</sub> values across all ROIs in the phantom.
<figure>
  <img src="./assets/model_fitting_vfa_bias.PNG" alt="TODO">
  <figcaption><b>Figure 7:</b>  <i>The percentage error between the estimated T<sub>1</sub> values and the NMR reference values provided by the phantom manufacturer, or %bias, for each ROI in the system phantom. In this dataset the %bias is generally less than 25%, the vial with the shortest relaxation time (t1_roi_14) is an outlier with a %bias > 200%.</i></figcaption>
</figure>



## Expected output from analysis
Once the example analysis script has completed running on your computer, you should have two new folders in the directory "mrbias/examples/output". Both folders should start with "PMCC_SIEMENS-Skyra-3p0T..." and will have a structure similar to that shown in Figure 8. The output folder includes a PDF report (mri_bias.pdf), a logfile of the analysis (mri_bias.txt), a datamap (data_map_000.csv) to link sequence details to their related model fit directory, and directory for each model fit (t1_vfa..., t1_vir..., t2_mse...). 

<figure>
  <img src="./assets/relaxometry_testsetA_output_directory.PNG" alt="TODO">
  <figcaption><b>Figure 8:</b>  <i>The expected output directory structure after analysis of mrbias_testset_B.</i></figcaption>
</figure>
<br><br>

In each model fit directory we expect two comma separated value (.csv) files to be present (Figure 9). The "model_fit_summary.csv" provides summary metrics for each ROI such as the estimated T<sub>1</sub> relaxation time and the reference values. The "voxel_data.csv" file contains the measurement data for each voxel in an ROI, for example the variable flip angle curve for every voxel. The voxel data in this file is the processed DICOM data, which means it has had all configured pre-processing steps applied (i.e. normalisation, clipping detection, averaging).
<figure>
  <img src="./assets/relaxometry_testsetA_VFA_directory.PNG" alt="TODO">
  <figcaption><b>Figure 9:</b>  <i>The expected files in each model fit directory include a summary of the estimated parameters (model_fit_summary.csv) and the measured data to which the model was fit (voxel_data.csv).</i></figcaption>
</figure>
<br><br>

We recommend that you review the PDF reports generated on your computer, and check that the output for mrbias_testset_B (in the output folder PMCC_SIEMENS-Skyra-3p0T-46069_20200706-201432) is comparable to the figures in this tutorial.

## Running the software on your own data
To run MR-BIAS on your own dataset we suggest making a copy of the "relaxometry_example_1.py" script (below) and naming it "my_relaxometry_analysis_script.py". You will then need to modify a few key lines to point the software at your data, set the correct image sorting settings, and define your experimental conditions.
```python
# import required mrbias modules
from mrbias import MRBIAS

# specify the configuration file to control the analysis
configuration_filename = os.path.join("..", os.path.join("config", "relaxometry_example_1_config.yaml"))
# specific the dicom directories to analyse
dicom_directory_a = "..\data\mrbias_testset_A"
dicom_directory_b = "..\data\mrbias_testset_B"

# create a MRBIAS analysis object
mrb = MRBIAS(configuration_filename, write_to_screen=True)
# run the analysis (output will be created in the "output_directory" specified in the configuration file)
mrb.analyse(dicom_directory_a)
# run analysis on a second dicom directory (with the same analysis settings)
mrb.analyse(dicom_directory_b)
```

### Let the software know where to find your dataset
To get the software to analyse a dataset on your computer, you will need to update the 'dicom_directory' variables to the directories that contain your DICOM datasets.
```python
dicom_directory_a = "..\path_to_your_dicom_directory"
dicom_directory_b = "..\path_to_your_second_dicom_directory"
```
### Choose an appropriate image sorting option
The configuration file which controls how MR-BIAS will analyse these datasets is currently specified by the following statement. 
```python
configuration_filename = os.path.join("..", os.path.join("config", "relaxometry_example_1_config.yaml"))
```
This configuration file is setup to categorise images acquired on a Siemens 3T scanner with sequence parameters as recommended in the Calibre MRI system phantom manual. If your data was acquired on another scanner from a different manufacturer or with a different set of sequence parameters, then the image sorting may not categorise your DICOM data correctly. If that is the case, there are a number of pre-defined image sorting options which may match your data (Table 1).


|   Image Sorting Setting  |     Scan Session            |     Phantom     | Manufacturer | Field strength | Comments                                                                             |
|:------------------------:|:---------------------------:|:---------------:|:------------:|:--------------:|:-------------------------------------------------------------------------------------|
|    siemens_skyra_3p0T    |  sys_siemens_3T_skyra.py    | System Phantom  |   Siemens    |     3.0 T      | Protocols with sequence settings as recommended in the Calibre System Phantom manual |        
|   philips_marlin_1p5T    |  sys_philips_1p5T_marlin.py | System Phantom  |   Philips    |     1.5 T      | Protocols with sequence settings as recommended in the Calibre System Phantom manual |

<b>Table 1:</b>  <i>The options for pre-defined image sorting of the ISMRM/NIST system phantom.</i>

To select another option, such as 'philips_marlin_1p5T', we suggest you make a copy of 'relaxometry_example_1_config.yaml' and call it something like 'my_config.yaml'. You will need to update your script to use this new configuration file with the following statement:
```python
configuration_filename = os.path.join("..", os.path.join("config", "my_config.yaml"))
```
In your file 'my_config.yaml' you will then need to update the block of settings which are responsible for file sorting
```
# ====================================================================================================================
# DICOM sorting options
# ====================================================================================================================
dicom_sorting:
  scan_protocol : "siemens_skyra_3p0T"   
  scan_protocol_options: ["siemens_skyra_3p0T", "philips_marlin_1p5T"]
```
to use the 'philips_marlin_1p5T' option update the 'scan_protocol' line to be the following
```
scan_protocol : "philips_marlin_1p5T" 
```
If the available sorting options are not suitable for your dataset, then please refer to our tutorial on [adding a new scanner or acquisition protocol](./adding_a_new_scanner.md) or [contact us](./contact.md) for assistance.

### Define the relevant experimental conditions
Some quantitative parameters, such as T<sub>1</sub> and T<sub>2</sub> relaxation times, are magnetic field strength and temperature dependant. To ensure the software is comparing your measurements to the correct measurement values, you will need to check or update the experimental conditions used during the acquisition of your image data. The field strength and temperature are defined in the following section of the configuration file
```
# ====================================================================================================================
# PHANTOM/EXPERIMENT configuration
# ====================================================================================================================
phantom_experiment:
  phantom_manufacturer: "caliber_mri" # options include [caliber_mri]
  phantom_type: "system_phantom"      # options include [system_phantom]
  phantom_serial_number: "130-0093"   # for CalibreMRI phantoms SN has format XXX-XXXX
  field_strength_tesla: 3.0           # options include [1.5, 3.0]
  temperature_celsius: 20.0           # will match to closest experimental reference data
```

### Run your analysis
Now that you have created and updated your own analysis script and configuration file, you can run the analysis with the following command
```
python my_relaxometry_analysis_script.py
```

## Concluding remarks
You have now run your first relaxometry analysis with MR-BIAS, if you had any issues with the example script please [contact us](./contact.md). We hope the tutorial has given you an overview of the basic workflow of the automated analysis, and how you can use the PDF report and the log files to check the analysis is valid, and to view the estimated relaxation rates. 

A tutorial which may now be of interest is [how to customise the analysis settings by writing your own configuration file](./writing_a_config_file.md). Alternatively, please refer to the [main tutorial page](./tutorials.md) for help on other topics.

<br> <br> <br> <br> <br> <br>

### Revision history

|     Date      |   Author    | Changes                                   |
|:-------------:|:-----------:|:------------------------------------------|
| 12 March 2025 | James Korte | Created introductory relaxometry tutorial |




