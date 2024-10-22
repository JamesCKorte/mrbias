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
from mrbias.roi_detection_methods.detection_methods import RegistrationMethodAbstract


class RegistrationMutualInformationGradientDescent(RegistrationMethodAbstract):
    def __init__(self, target_geo_im, fixed_im, roi_template,
                 centre_images=True):
        super().__init__(target_geo_im, fixed_im, roi_template)
        mu.log("RegistrationCorrelationGradientDescent::init()", LogLevels.LOG_INFO)
        self.centre_images = centre_images

    def register(self):
        moving = sitk.Cast(self.moving_im, sitk.sitkFloat32)
        fixed = sitk.Cast(self.fixed_im, sitk.sitkFloat32)
        # setup the registration method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingPercentage(0.25)
        R.SetMetricSamplingStrategy(R.RANDOM)
        # R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
        #                                            numberOfIterations=5000,
        #                                            minStep=0.000001)
        R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                        numberOfIterations=1000)
                                        #double convergenceMinimumValue=1e-6)
        R.SetOptimizerScalesFromPhysicalShift()
        # centre as initial transform
        initial_transform = sitk.Euler3DTransform()
        if self.centre_images:
            initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R.SetInitialTransform(initial_transform)
        R.SetInterpolator(sitk.sitkLinear)
        # hook up a logger
        R.AddCommand(sitk.sitkIterationEvent, lambda: self.print_iteration_info(R))
        # register
        final_transform = R.Execute(fixed, moving)
        return final_transform, R.GetMetricValue()

