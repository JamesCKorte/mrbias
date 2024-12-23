*Authors: Arpita Dutta, James Korte &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Modified: 23/12/2024*

# Tutorial: Adding a new signal model for curve fitting

This tutorial will show you how to add a new signal model by using the example of a T2* model for the [system lite phantom or essential system phantom](https://qmri.com/product/essential-system/) (which is a simplified version of the [premium system phantom](https://qmri.com/product/ismrm-nist-premium-system-phantom/)). These modifications were made with our collaborators at the Auckland Bioengineering Institute, the University of Auckland. They wanted to try out MRBIAS on a series of T2* images of their system lite phantom.


## Where do the signal models live and how do I add a new one?
Each signal model is a class defined in a seperate file in the folder “mrbias/curve_fit_models”, for example a two parameter T1 relaxation model for variable flip angle (VFA) data is defined in the file "mrbias/curve_fit_models/t1_vfa_2param.py". Adding a new signal model involves three-steps
- Define a new curve fitting model 
- Include the new model as an option in the main progam "mrbias/mri_bias.py"  
- Add a region of interest (ROI) detection template if required by the new signal model


### Define a new curve fitting model
You will need to create a new file in the "mrbias/curve_fit_models" folder, in this example we copied an existing file "mrbias/curve_fit_models/t1_vfa_2param.py" and renamed it for our new T2* model "t2star_ge_2param.py". In each signal model file you need to modify the class that inherits from the base class "CurveFitAbstract", we renamed the existing class "T2SECurveFitAbstract3Param" to our new class named "T2StarCurveFitAbstract2Param". The "CurveFitAbstract" class handles the curve fitting procedure and reliest on your new class to define your signal model, including a mathematical function and it's associated variables. Here is the class that we added to define a two parameter the T2* model, where the two free parameters are M0 and T2Star, and the echo time, TE, is as measured on the MRI scanner. Note that the function 'get_initial_parameters' returns two initial values, one for M0 and one for T2Star, and the function'fit_parameter_bounds' functions returns bounds (an lower bound and an upper bound) for both the parameters, M0 and T2Star.

```python
class T2StarCurveFitAbstract2Param(CurveFitAbstract):
    def __init__(self, imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                 echo_exclusion_list=None, exclusion_label=None):
        self.eqn_param_map = OrderedDict()
        self.eqn_param_map["M0"] = ("Equilibrium magnetisation", "max(S(TE))", "0.0", "inf")
        self.eqn_param_map["T2Star"] = ("T2Star relaxation time", "TE_median", "0.0", "inf")
        self.eqn_param_map["TE"] = ("Echo time", "as measured", "-", "-")
        self.eqn_param_map["TE_median"] = ("The median measured echo time", "from signal", "-", "-")
        super().__init__(imageset, reference_phantom, initialisation_phantom, preprocessing_options,
                         echo_exclusion_list, exclusion_label)

    def get_model_name(self):
        return "T2StarCurveFit2param"

    def get_meas_parameter_name(self):
        return 'EchoTime'

    def get_symbol_of_interest(self):
        return 'T2Star'
    def get_ordered_parameter_symbols(self):
        return ['M0', 'T2Star']
    def get_meas_parameter_symbol(self):
        return 'TE'

    def fit_function(self, TE, M0, T2Star):
        # M0 = Weighted equilibrium magnetisation
        # T2Star = T2Star relaxation time
        # TE = echo time
        return M0 * np.exp(-TE/T2Star)

    def get_initial_parameters(self, roi_dx, voxel_dx):
        cf_roi = self.cf_rois[roi_dx]
        vox_series = cf_roi.get_voxel_series(voxel_dx)
        init_val = cf_roi.initialisation_value #self.initialisation_phantom.get_roi_by_dx(roi_dx)
        if cf_roi.is_normalised:
            return [1.,  init_val]
        else:
            return [np.max(vox_series),  init_val]

    def fit_parameter_bounds(self):
        return (0., 0.), (np.inf, np.inf)

    def estimate_cf_start_point(self, meas_vec, av_sig, init_val, cf_roi):
        try:
            # half_val = np.max(av_sig)*0.368
            # half_val_idx = np.argmin(np.abs(av_sig-half_val))
            # return meas_vec[half_val_idx]
            # use median
            return np.median(cf_roi.meas_var_vector)
        except:
            mu.log("T2StarCurveFitAbstract2Param::estimate_cf_start_point(): failed to estimate start point, using "
                   "default values of %.3f" % init_val, LogLevels.LOG_WARNING)
            return init_val

    def get_model_eqn_strs(self):
        eqn_strs = ["S(TE) = M0 * exp(-TE/T2Star)"]
        return eqn_strs

```

To include your new model in the curve fitting module, you also need to add a line to the "mrbias/curve_fitting.py" such as the following
```python
from mrbias.curve_fit_models.t2star_ge_2param import T2StarCurveFitAbstract2Param
```



### Incliude the new model in mrbias/mri_bias.py 

The main program ("mrbias/mri_bias.py") needs to be updated to include your new curve fitting class an an option when and analysing the MRI dataset. If we were adding another T1 variable flip angle (VFA) model, this would be a quick process, as the main program already includes code to load T1-VFA datasets and interpret T1-VFA options from the configuration file. In that simplified situation a few lines of code would be required to use your new model if it was selected in the configutation file, for example if a 4 parameter T1-VFA model had been added then the following would be added to the 'analyse' function of the MRBIAS class in the main program:
```python
def analyse(self, dicom_directory):
		.
		.
		.
    for t1_vir_model_str in t1_vir_model_list:
	mdl = None
	if t1_vir_model_str == "2_param":
	    mdl = curve_fit.T1VIRCurveFitAbstract2Param(imageset=t1_vir_imageset,
							reference_phantom=ref_phan,
							initialisation_phantom=init_phan,
							preprocessing_options=preproc_dict,
							inversion_exclusion_list=inversion_exclusion_list,
							exclusion_label=exclusion_label)
	elif t1_vir_model_str == "3_param":
	    mdl = curve_fit.T1VIRCurveFitAbstract3Param(imageset=t1_vir_imageset,
							reference_phantom=ref_phan,
							initialisation_phantom=init_phan,
							preprocessing_options=preproc_dict,
							inversion_exclusion_list=inversion_exclusion_list,
							exclusion_label=exclusion_label)
	elif t1_vir_model_str == "4_param":                                                                   # <-- ADDED
	    mdl = curve_fit.T1VIRCurveFitAbstract4Param(imageset=t1_vir_imageset,                             # <-- ADDED
							reference_phantom=ref_phan,                           # <-- ADDED
							initialisation_phantom=init_phan,                     # <-- ADDED
							preprocessing_options=preproc_dict,                   # <-- ADDED
							inversion_exclusion_list=inversion_exclusion_list,    # <-- ADDED
							exclusion_label=exclusion_label)                      # <-- ADDED
```
But as this is the first T2* dataset to be analysed we needed to update a few sections of the main program to
- Add code to load a T2* dataset
- Add code to parse T2* model fitting options from the configuration file (.yaml)
- Add code to load the T2* model you just created

Firstly, to read in the relevant T2* images, a new imageset is added to the 'analyse' function. The following highlights changes to the function to load T2* images into 't2_star_imagesets' and log some basic imaging details.
```python
def analyse(self, dicom_directory):
		.
		.
		.
        # if a valid scan protocol found load up relevant image sets
        geometric_images = []
        pd_images = []
        t1_vir_imagesets = []
        t1_vfa_imagesets = []
        t2_mse_imagesets = []
        t2_star_imagesets = []            # <--------------------------------------------------------------------------- ADDED
        if ss is not None:
            geometric_images = ss.get_geometric_images()
            #pd_images = ss.get_proton_density_images()
            t1_vir_imagesets = ss.get_t1_vir_image_sets()
            t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
            t2_mse_imagesets = ss.get_t2_mse_image_sets()
            t2_star_imagesets = ss.get_t2star_image_sets() # <----------------------------------------------------------- ADDED
            ss.write_pdf_summary_page(c)
        # log some basic details of the imagesets
        for t1_vir_imageset in t1_vir_imagesets:
            mu.log("Found T1(VIR): %s" % type(t1_vir_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t1_vir_imageset), LogLevels.LOG_INFO)
        for t1_vfa_imageset in t1_vfa_imagesets:
            mu.log("Found T1(VFA): %s" % type(t1_vfa_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t1_vfa_imageset), LogLevels.LOG_INFO)
        for t2_mse_imageset in t2_mse_imagesets:
            mu.log("Found T2(MSE): %s" % type(t2_mse_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t2_mse_imageset), LogLevels.LOG_INFO)
        for t2_star_imageset in t2_star_imagesets:  #       <----------------------------------------------------------- ADDED
            mu.log("Found T2(Star): %s" % type(t2_star_imageset), LogLevels.LOG_INFO) # <------------------------------- ADDED
            mu.log("\t\t%s" % str(t2_star_imageset), LogLevels.LOG_INFO) # <-------------------------------------------- ADDED
```
To linked any available geometric images, you need to include the new imageset (i.e. t2_star_imagesets) to the following section of code.
```python
geometric_images_linked = set()
        if ss is not None:
            # exclude any geometric images that are not reference in curve fit data
            mu.log("MR-BIAS::analyse(): Identify linked geometric images ...", LogLevels.LOG_INFO)
            for geometric_image in geometric_images:
                for fit_imagesets in [t1_vir_imagesets, t1_vfa_imagesets, t2_mse_imagesets, t2_star_imagesets]: <-- MODIFIED
                    for imageset in fit_imagesets:
                        g = imageset.get_geometry_image()
                        if g.get_label() == geometric_image.get_label():
                            geometric_images_linked.add(geometric_image)
                            mu.log("\tfound geometric image (%s) linked with with imageset (%s)" %
                                   (geometric_image.get_label(), imageset.get_set_label()),
                                   LogLevels.LOG_INFO)
                            mu.log("\tfound geometric image (%s) linked with with imageset.geoimage (%s)" %
                                   (repr(geometric_image), repr(imageset.get_geometry_image())),
                                   LogLevels.LOG_INFO)
```

Secondly, add in configuration options for T2* model fitting with the following changes to the class 'MRIBiasCurveFitConfig'
```python
class MRIBiasCurveFitConfig(MRIBIASConfiguration):
	.
	.
	.
    # T2 Star GE SETTINGS
    def __get_t2_star_ge(self, param_name, default_value):
        return self.__get_nestled("t2_star_ge_options", param_name, default_value)
    def __get_t2_star_ge(self):
        return self.__get_t2_star_ge("fitting_models", ["2_param"])
    def get_t2_star_ge_exclusion_list(self):
        return self.__get_t2_star_ge("echo_exclusion_list", None)
    def get_t2_star_ge_exclusion_label(self):
        return self.__get_t2_star_ge("echo_exclusion_label", "user_angle_excld")
    def get_t2_star_ge_2D_slice_offset_list(self):
        return self.__get_2D_slice_offset_list("t2_star_ge_options")
```

Lastly, using the configuration options you added, if a user selects a T2 star model fit, the following code is added to load up your new T2 star model with the user specified configuration options
```python
def analyse(self, dicom_directory):
		.
		.
		.
	t1vir_map_df = None
	t1vfa_map_df = None
	t2mse_map_df = None
	t2star_map_df = None    # <------------------------------- ADDED
	dw_map_df = None
	t1vir_map_d_arr = []
	t1vfa_map_d_arr = []
	t2mse_map_d_arr = []
	t2star_map_d_arr = []
	dw_map_d_arr = []       # <------------------------------- ADDED
		.
		.
		.
	# ----------------------------------------------------             #  <--- ADDED (the whole next chunk)
        # T2Star Gradient Echo
        for t2_star_imageset in t2_star_imagesets:
            t2_star_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t2_star_model_list = self.cf_config.get_t2_star_ge_models()
            echo_exclusion_list = self.cf_config.get_t2_star_ge_exclusion_list()
            exclusion_label = self.cf_config.get_t2_star_ge_exclusion_label()
            for t2_star_model_str in t2_star_model_list:
                mdl = None
                if t2_star_model_str == "2_param":
                    mdl = curve_fit.T2StarCurveFitAbstract2Param(imageset=t2_star_imageset,
                                                                 reference_phantom=ref_phan,
                                                                 initialisation_phantom=init_phan,
                                                                 preprocessing_options=preproc_dict,
                                                                 echo_exclusion_list=echo_exclusion_list,
                                                                 exclusion_label=exclusion_label)
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c, is_system=True,
                                                include_pmap_pages=include_roi_pmap_pages)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
                    # add data to the map to link dicom images to analysis folders
                    for series_uid, series_num, TE, TR in zip(t2_star_imageset.series_instance_UIDs,
                                                              t2_star_imageset.series_numbers,
                                                              t2_star_imageset.meas_var_list,
                                                              t2_star_imageset.repetition_time_list):
                        exclude_TE = TE in echo_exclusion_list
                        t2star_map_d_arr.append([t2_star_imageset.label, t2_star_model_str, series_num, TE, TR,
                                                 cf_normal, cf_averaging, cf_exclude, cf_percent_clipped_threshold,
                                                 exclude_TE, exclusion_label,
                                                 series_uid, d_dir])

        if len(t2star_map_d_arr):
            t2star_map_col_names = ["Label", "Model", "SeriesNumber",
                                    "%s (%s)" % (
                                        t2_star_imagesets[0].meas_var_name, t2_star_imagesets[0].meas_var_units),
                                    "RepetitionTime",
                                    "Normalise", "Average", "ExcludeClipped", "ClipPcntThreshold",
                                    "Excluded", "ExclusionLabel",
                                    "SeriesInstanceUID", "AnalysisDir"]
            t2star_map_df = pd.DataFrame(t2star_map_d_arr, columns=t2star_map_col_names)
		.
		.
		.
	# join the data mapping frames and save to disk
        map_vec = []
        for m in [t1vir_map_df, t1vfa_map_df, t2mse_map_df, t2star_map_df, dw_map_df]:   # <------------------------------- MODIFED
            if m is not None:
                map_vec.append(m)
        if len(map_vec):
            df_data_analysis_map = pd.concat([t1vir_map_df, t1vfa_map_df, t2mse_map_df, t2star_map_df, dw_map_df], # <----- MODIFED
                                             axis=0, join='outer')
            df_data_analysis_map.to_csv(data_map_filename)							
```


### Add a ROI detection template (if required)

When adding a new signal model type, you may also need to be able to detect different ROIs in the phantom. The templates for ROI detection are stored in the "mrbias/roi_detection_templates" folder, with each template residing in its designated phantom-scanner-specific subdirectory. The T2* model for system lite phantom, for example, uses "mrbias/roi_detection_templates/systemlite_siemens_vida_3p0T/default_T2_rois.yaml" template.

To add a new ROI template, please refer to 'mrbias/documentation/adding_a_roi_template' documentation.




## Testing your new Signal model 
Once you are happy with all the settings, it is a good time to see if the signal model works. You can analyse your dataset with the new model by calling MRBIAS, an example is provided in 'mrbias/examples/relaxometry_auckland_cam.py'. In this code, ensure that the 'dicom_directory_a' correctly points to a directory of DICOM images which you want to analys, and the 'configuration_filename' points to an appropriate configuration. We have included an example configuration file in 'mrbias/config/relaxometry_auckland_cam_config.yaml', some details on how you can update the configuration file are discussed below. 

### Update the Config file

The 'mrbias/config' folder contains sample configuration files. Check that all of the settings in the config file are align for your requirments.  Here are a few mentioned fields which were updated to run the T2* model.

#### Update the output directory
```python
# ====================================================================================================================
# Global settings
# ====================================================================================================================
global:
  #output_directory: "Data/Output/" # use forward slashes in filepath
```
 
#### Correct ROI detection template is selected in the configuration file
For the system-lite phantom, it set to "systemlite_siemens_vida_3p0T" 


```python
# ====================================================================================================================
# ROI detection options
# ====================================================================================================================
roi_detection:
  template_name: "systemlite_siemens_vida_3p0T"   # options include [siemens_skyra_3p0T, systemlite_siemens_vida_3p0T]
  save_mask_images: False                         # future feature (not implemented)
```

#### Update the 'scan_potocol' field
```python
# ====================================================================================================================
# DICOM sorting options
# ====================================================================================================================
dicom_sorting:
  scan_protocol : "auckland_cam_3p0T"   # options include [siemens_skyra_3p0T, philips_marlin_1p5T, auckland_cam_3p0T]

```

#### Update "curve fitting"
To run T2* two parameter model, "fitting_models" under "t2_star_options" set to "2_param".

```python
# ====================================================================================================================
# Curve fitting options
# ====================================================================================================================
curve_fitting:
  # -----------------------------------------------------------------------------------------------------------------
  # PRE-PROCESSING
  # -----------------------------------------------------------------------------------------------------------------
  averaging: "voxels_in_ROI"              # options include [none, voxels_in_ROI]
  normalisation: "voxel_max"              # options include [none, voxel_max, roi_max]
  exclude: "clipped"                      # options include [none, clipped]
  percent_clipped_threshold: 100           # [0, 100] for parital clipping, the number of unclipped/clean voxels required across all measurements in the ROI
  # -----------------------------------------------------------------------------------------------------------------
  # MODEL OPTIONS
  # -----------------------------------------------------------------------------------------------------------------
  t2_star_options:
    fitting_models: ["2_param"] # options include: 3_param (a * np.exp(-t*r2star) + b) and 2_param (a * np.exp(-t*r2star))

```


## Concluding remarks
We hope this has been helpful for you to make your own signal model. If anything is unclear and you would like some further assistance, or if you would like to share your template for others to use, please reach out to the MR-BIAS team with the [contact details on the main page](../README.md). 


