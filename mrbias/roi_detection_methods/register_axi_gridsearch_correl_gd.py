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
  21-September-2024  :               (James Korte) : Refactoring   MR-BIAS v1.0
"""
import os
import copy
import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy.signal as scisig # to find local minima in metric grid search



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
from mrbias.roi_detection_methods.detection_methods import RegistrationMethodAbstract, ROI_DETECT_SEED

class RegistrationAxialRotGridThenNBestGradDecnt(RegistrationMethodAbstract):
    def __init__(self, target_geo_im, fixed_im, roi_template,
                 centre_images=True, n_best=6):
        super().__init__(target_geo_im, fixed_im, roi_template)
        mu.log("RegistrationCorrelationGradientDescent::init()", LogLevels.LOG_INFO)
        self.centre_images = centre_images
        self.n_best = n_best
        self.grid_tx_list = None

    def register(self):
        # setup the storage for grid results
        self.grid_tx_list = []
        # cast images to float for registration metric evaluation
        moving = sitk.Cast(self.moving_im, sitk.sitkFloat32)
        fixed = sitk.Cast(self.fixed_im, sitk.sitkFloat32)
        # perform a grid search of set angles to identify potential starting points
        tx0, tx_combined = self.register_with_correl_grid_search(fixed, moving, centering_step=self.centre_images)
        # identify n_best initial transforms
        initial_tx_list = self.get_nbest_transforms_from_grid_search(tx0)
        # perform a rigid 3D image registration for each of the n_best starting positions
        final_tx_list = []
        for t_dx, tx1 in enumerate(initial_tx_list):
            # to remove effects of a partial field of view
            # resize the fixed image to the transformed moving image field of view
            mov_im_shape_mm = np.array(moving.GetSize()) * np.array(moving.GetSpacing())
            mov_im_height_mm = mov_im_shape_mm[2]
            # crop the fixed image to the same height
            size_arr = [fixed.GetSize()[0],
                        fixed.GetSize()[1],
                        int(np.ceil(mov_im_height_mm / fixed.GetSpacing()[2]))]
            index_arr = [0,
                         0,
                         (fixed.GetSize()[2] - size_arr[2]) // 2]
            fixed_crop = sitk.RegionOfInterest(fixed, size_arr, index_arr)

            moving_1 = sitk.Resample(moving, fixed_crop, tx1, sitk.sitkLinear)
            fine_tx = self.register_with_correlation_gradient_descent(fixed_crop, moving_1, centering_step=False)
            # create final combined transform and assess accuracy
            tx_combined = sitk.Transform()
            tx_combined.AddTransform(tx1)
            tx_combined.AddTransform(fine_tx)
            moving_2 = sitk.Resample(moving, fixed_crop, tx_combined, sitk.sitkLinear)
            # assess
            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsCorrelation()
            R.SetMetricSamplingStrategy(R.NONE) # use all samples on fixed image for assessment
            metric_2 = R.MetricEvaluate(fixed_crop, moving_2)
            # add it into the results list
            final_tx_list.append([t_dx, metric_2, tx_combined])
        # select the best performing combined transform (the lowest correlation cost function/metric)
        final_tx_list.sort(key=lambda a: a[1])
        best_dx, best_metric, best_tx = final_tx_list[0]
        return best_tx, best_metric # transform, metric

    def register_with_correl_grid_search(self, fixed, moving, centering_step):
        # set rotational centre (0,0,0) to be in the middle of the fixed image
        # we achieve this by setting the origin (which is in the bottom left corner) to half the image extent
        im_fov = np.array(fixed.GetSpacing()) * np.array(fixed.GetSize())
        new_origin = (-im_fov / 2).tolist()
        fix_im = mu.sitk_deepcopy(fixed)
        fix_im.SetOrigin(new_origin)

        # create translational transform between original and new rotational centre
        orig_origin = np.array(fixed.GetOrigin())
        tx0 = sitk.Euler3DTransform()
        tx0.SetTranslation(orig_origin - new_origin)

        # setup correlation registration on a fixed grid of angles
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingPercentage(percentage=0.8, seed=ROI_DETECT_SEED)
        R.SetMetricSamplingStrategy(R.REGULAR)
        # Note: Assumes the slice pack is aligned with an axial plan of the phantom (circles of  ROIs visible)
        # Rotations included in the search are a flip up/down and in-plane rotation at an angle increment of dInPlaneAngle
        # Order of parameters is ( x-rotation, y-rotation, z-rotation, x, y, z)
        dInPlaneAngle = 10.0
        R.SetOptimizerAsExhaustive([1,
                                    0,
                                    int(180.0 / dInPlaneAngle),  # every dInPlaneAngle degs
                                    0, 0, 0])
        R.SetOptimizerScales(
            [np.pi,
             0.,
             np.pi / float(180.0 / dInPlaneAngle),  # every dInPlaneAngle degs
             0.0, 0.0, 0.0])

        # centre as initial transform
        initial_transform = sitk.Euler3DTransform()
        if centering_step:
            initial_transform = sitk.CenteredTransformInitializer(fix_im, moving, sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.MOMENTS)
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        # hook up a logger
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.record_iteration_info(R))
        # register
        final_transform = R.Execute(fix_im, moving)
        # create composite transform which accounts for the initial rotation centre change as well
        # this is required so the mask can be correctly transformed to the target image
        tx_combined = sitk.Transform()
        tx_combined.AddTransform(final_transform)
        tx_combined.AddTransform(sitk.Transform.GetInverse(tx0))
        # return the initial shift transform (for use with recorded history), and also the final best transform if no further action wanted
        return tx0, tx_combined

    def record_iteration_info(self, method):
        rx, ry, rz, tx, ty, tz = method.GetOptimizerPosition()
        self.grid_tx_list.append([method.GetMetricValue(), np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz), tx, ty, tz])

    def get_nbest_transforms_from_grid_search(self, tx0):
        df = pd.DataFrame(self.grid_tx_list, columns=["Metric", "Flip", "NoRot", "InPlane", "tx", "ty", "tz"])
        min_tx_data = []
        for angle in [0, 180]:
            df_angle = df[df["Flip"] == angle]
            metric_arr = df_angle.Metric.values
            min_dx = scisig.argrelextrema(metric_arr, np.less)[0]  # find local minima
            for dx in min_dx:
                d_vec = [metric_arr[dx],
                         df_angle.Flip.values[dx],
                         df_angle.NoRot.values[dx],
                         df_angle.InPlane.values[dx],
                         df_angle.tx.values[dx],
                         df_angle.ty.values[dx],
                         df_angle.tz.values[dx]]
                min_tx_data.append(d_vec)
        # build a new data frame with transform at minima
        df_min = pd.DataFrame(min_tx_data, columns=["Metric", "Flip", "NoRot", "InPlane", "tx", "ty", "tz"])
        # get the n_best performing
        df_min = df_min.sort_values(by="Metric", ascending=True)
        df_min = df_min.iloc[0:self.n_best]
        # create SimpleITK transforms
        best_tx_list = []
        for r_dx, r in df_min.iterrows():
            trans = sitk.Euler3DTransform()
            trans.SetTranslation([r.tx, r.ty, r.tz])
            trans.SetRotation(np.deg2rad(r.Flip),
                              np.deg2rad(r.NoRot),
                              np.deg2rad(r.InPlane))
            tx_combined = sitk.Transform()
            tx_combined.AddTransform(trans)
            tx_combined.AddTransform(sitk.Transform.GetInverse(tx0))
            best_tx_list.append(tx_combined)
        return best_tx_list

    def register_with_correlation_gradient_descent(self, fixed, moving, centering_step):
        # setup the registration method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingPercentage(0.5, seed=ROI_DETECT_SEED)
        R.SetMetricSamplingStrategy(R.REGULAR)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                   numberOfIterations=1000,
                                                   minStep=0.000001)
        R.SetOptimizerScalesFromPhysicalShift()
        # centre as initial transform
        initial_transform = sitk.Euler3DTransform()
        if centering_step:
            initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.MOMENTS)
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        # register
        return R.Execute(fixed, moving)

