*Authors: Arpita Dutta &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Modified: 18/07/2023*

# Tutorial: Adding a new signal model for curve fitting

This tutorial will show you how to add a new signal model by using the example of a T2* model for the [system lite phantom or essential system phantom](https://qmri.com/product/essential-system/) (which is a simplified version of the [premium system phantom](https://qmri.com/product/ismrm-nist-premium-system-phantom/)). We made these modifications for our collaborators at the Auckland Bioengineering Institute, the University of Auckland. They wanted to try out MRBIAS on their system lite phantom images.


## Where do the signal model function live?
Each signal model function is stored in the “mrbias/curve_fitting.py” python script. Adding a new signal model involves three-step processes. 
- Add ROI detection template for the new signal model
- Include the curve fitting function
- Update mrbias/mri_bias.py 


### The ROI detection template

For the new signal model, it should be able to detect ROIs in its corresponding MRI sequence. The templates for ROI detection are stored in the "mrbias/roi_detection_templates" folder, with each template residing in its designated phantom-scanner-specific subdirectory. The T2* model for system lite phantom, for example, uses "mrbias/roi_detection_templates/systemlite_siemens_vida_3p0T/default_T2_rois.yaml" template.

To add a new ROI template, please refer to 'mrbias/documentation/adding_a_roi_template' documentation.


### Include the curve fitting function
You will need to open the "mrbias/curve_fitting.py" script and add a new class for the new signal model. Here is the class that was added to fit the T2* model with two parameters. Also, ensure that the 'get_initial_parameters' and 'fit_parameter_bounds' functions return only two parameters when adding a two-parameter model and three parameters when working with a three-parameter model.

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

    def get_model_eqn_parameter_strs(self):
        eqn_param_strs = []
        for p_name, (descr, init_v, min_v, max_v) in self.eqn_param_map.items():
            eqn_param_strs.append((p_name, descr, init_v, min_v, max_v))
        return eqn_param_strs

```

You can easily copy-paste an existing signal model class and then update the fit function and inputs according to the model you intend to add.



### Update mrbias/mri_bias.py 

The "mrbias/mri_bias.py" script needs to be updated in order to read and analyse the original MR images.

Firstly, to read the relevant original images, you need to introduce a new imageset to the 'analyse' function. For example, all original T2* images are loaded into 't2_star_imagesets' and some basic imaging details are logged.

```python
def analyse(self, dicom_directory):
        .
		.
		.
		.
        # if a valid scan protocol found load up relevant image sets
        geometric_images = []
        pd_images = []
        t1_vir_imagesets = []
        t1_vfa_imagesets = []
        t2_mse_imagesets = []
        if ss is not None:
            geometric_images = ss.get_geometric_images()
            #pd_images = ss.get_proton_density_images()
            t1_vir_imagesets = ss.get_t1_vir_image_sets()
            t1_vfa_imagesets = ss.get_t1_vfa_image_sets()
            t2_mse_imagesets = ss.get_t2_mse_image_sets()
			t2_star_imagesets = ss.get_t2star_image_sets()
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
        for t2_star_imageset in t2_star_imagesets:
            mu.log("Found T2(Star): %s" % type(t2_star_imageset), LogLevels.LOG_INFO)
            mu.log("\t\t%s" % str(t2_star_imageset), LogLevels.LOG_INFO)
```

To check for linked geometric images, you need to include the new imageset (i.e. t2_star_imagesets) to the following section of code.

```python
geometric_images_linked = set()
        if ss is not None:
            # exclude any geometric images that are not reference in curve fit data
            mu.log("MR-BIAS::analyse(): Identify linked geometric images ...", LogLevels.LOG_INFO)
            for geometric_image in geometric_images:
                for fit_imagesets in [t1_vir_imagesets, t1_vfa_imagesets, t2_mse_imagesets,t2_star_imagesets]:
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

In the code below, you need to add the new curve fitting function so that it can call the appropriate curve fitting function by reading the configuration file.

```python
 # T2 Star
        for t2_star_imageset in t2_star_imagesets:
            t2_star_imageset.update_ROI_mask()  # trigger a mask update
            # get model options from configuration file
            t2_star_model_list = cf_config.get_t2_star_models()
            echo_exclusion_list = cf_config.get_t2_star_exclusion_list()
            exclusion_label = cf_config.get_t2_star_exclusion_label()
            for t2_star_model_str in t2_star_model_list:
                mdl = None
                if t2_star_model_str == "3_param":
                    mdl = curve_fit.T2StarCurveFitAbstract3Param(imageset=t2_star_imageset,
                                                                 reference_phantom=ref_phan,
                                                                 initialisation_phantom=init_phan,
                                                                 preprocessing_options=preproc_dict,
                                                                 echo_exclusion_list=echo_exclusion_list,
                                                                 exclusion_label=exclusion_label)
                if t2_star_model_str == "2_param":
                    mdl = curve_fit.T2StarCurveFitAbstract2Param(imageset=t2_star_imageset,
                                                                 reference_phantom=ref_phan,
                                                                 initialisation_phantom=init_phan,
                                                                 preprocessing_options=preproc_dict,
                                                                 echo_exclusion_list=echo_exclusion_list,
                                                                 exclusion_label=exclusion_label)
                    
                if mdl is not None:
                    # add summary page to pdf
                    mdl.write_pdf_summary_pages(c)
                    # write the data output
                    d_dir = os.path.join(out_dir, mdl.get_imset_model_preproc_name())
                    if not os.path.isdir(d_dir):
                        os.mkdir(d_dir)
                    mdl.write_data(data_dir=d_dir,
                                   write_voxel_data=cf_write_vox_data)
							
```

Lastly, you need to create functions for the new model (i.e. T2*) under 'class MRIBiasCurveFitConfig'.

```python
   def get_t2_star_models(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t2_star_options" in cf_config.keys():
                t2_star_opts = cf_config["t2_star_options"]
                if "fitting_models" in t2_star_opts.keys():
                    return t2_star_opts["fitting_models"]
        # not found, return a default value
        default_value = ["3_param"]
        mu.log("MR-BIASCurveFitConfig::t2_star_options(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value


    def get_t2_star_exclusion_list(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t2_star_options" in cf_config.keys():
                t2_star_opts = cf_config["t2_star_options"]
                if "echo_exclusion_list" in t2_star_opts.keys():
                    return t2_star_opts["echo_exclusion_list"]
        # not found, return a default value
        default_value = None
        mu.log("MR-BIASCurveFitConfig::get_t2_star_exclusion_list(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

    def get_t2_star_exclusion_label(self):
        cf_config = super().get_curve_fitting_config()
        if cf_config is not None:
            if "t2_star_options" in cf_config.keys():
                t2_star_opts = cf_config["t2_star_options"]
                if "echo_exclusion_label" in t2_star_opts.keys():
                    return t2_star_opts["echo_exclusion_label"]
        # not found, return a default value
        default_value = "user_angle_excld"
        mu.log("MR-BIASCurveFitConfig::get_t2_star_exclusion_label(): not found in configuration file, "
               "using default value : %s" % str(default_value), LogLevels.LOG_WARNING)
        return default_value

```python


### Testing your new Signal model 
Once you are happy with all the settings, it is a good time to see if the signal model works.  Now, you can initiate the model by calling MRBIAS. A sample code is provided in 'mrbias/mr_bias_auckland_cam_example.py'. In this code, ensure that the 'dicom_directory_a' correctly points to your input MRI DICOMs, and the 'configuration_filename' points to the appropriate configuration for run. Before executing the code, please make sure that the configuration is suitably updated in accordance with your requirements.

### Update the Config file

The 'mrbias/config' folder contains sample configuration files. Check that all of the settings in the config file are align for your requirments.  Here are a few mentioned fields which were updated to run the T2* model.

#### Update the output directory
```python
 ====================================================================================================================
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


