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
from mrbias.image_sets import ImageGeometric
from mrbias.roi_detection_methods.detection_methods import RegistrationMethodAbstract
from mrbias.roi_detection_methods.detection_methods import DetectionOptions
from mrbias.roi_detection_methods.register_none import RegistrationNone
from mrbias.roi_detection_methods.register_correl_gd import RegistrationCorrelationGradientDescent
from mrbias.roi_detection_methods.register_mmi_gd import RegistrationMutualInformationGradientDescent
from mrbias.roi_detection_methods.register_mse_gridsearch import RegistrationMSEGridSearch
from mrbias.roi_detection_methods.register_axi_gridsearch_correl_gd import RegistrationAxialRotGridThenNBestGradDecnt


class RegistrationTwoStage(RegistrationMethodAbstract):
    def __init__(self, target_geo_im, fixed_im, roi_template,
                 stage_a_registration_method,#=RegistrationOptions.MSME_GRIDSEARCH,
                 stage_b_registration_method,#=RegistrationOptions.COREL_GRADIENTDESCENT,
                 partial_fov_a=False,
                 partial_fov_b=False):
        super().__init__(target_geo_im, fixed_im, roi_template)
        mu.log("RegistrationTwoStage::init()", LogLevels.LOG_INFO)
        self.stage_a_registration_method = stage_a_registration_method
        self.stage_b_registration_method = stage_b_registration_method
        self.stage_a_rego = RegistrationTwoStage.generate_registration_instance(stage_a_registration_method,
                                                                                fixed_im, target_geo_im,
                                                                                self.roi_template,
                                                                                partial_fov_a)
        self.stage_b_rego = None
        self.partial_fov_a = partial_fov_a
        self.partial_fov_b = partial_fov_b


    def register(self):
        # stage A registration
        trans_a, metric_a = self.stage_a_rego.register()
        trans_image_a = sitk.Resample(self.moving_im, self.fixed_im, trans_a, sitk.sitkLinear)
        trans_geo_im_a = ImageGeometric("", trans_image_a)
        # stage B registration
        self.stage_b_rego = RegistrationTwoStage.generate_registration_instance(self.stage_b_registration_method,
                                                                                self.fixed_im, trans_geo_im_a,
                                                                                self.roi_template,
                                                                                self.partial_fov_b)
        trans_b, metric_b = self.stage_b_rego.register()
        # compose the final transform
        combined_transform = None
        if hasattr(sitk, "CompositeTransform"):
            combined_transform = sitk.CompositeTransform(trans_a)
        elif hasattr(sitk, "Transform"):
            combined_transform = sitk.Transform(trans_a)
        else:
            mu.log("RegistrationTwoStage::register(): unable to locate simpleITK multi-stage transform", LogLevels.LOG_ERROR)
            assert False
        combined_transform.AddTransform(trans_b)
        return combined_transform, metric_b

    @staticmethod
    def generate_registration_instance(reg_method, fixed_geom_im, target_geo_im, roi_template, partial_fov=False):
        rego = None
        centre_images = not partial_fov
        if reg_method == DetectionOptions.NONE:
            rego = RegistrationNone(target_geo_im, fixed_geom_im, roi_template)
        elif reg_method == DetectionOptions.COREL_GRADIENTDESCENT:
            rego = RegistrationCorrelationGradientDescent(target_geo_im, fixed_geom_im, roi_template,
                                                          centre_images=centre_images)
        elif reg_method == DetectionOptions.MSE_GRIDSEARCH:
            rego = RegistrationMSEGridSearch(target_geo_im, fixed_geom_im, roi_template)
        elif reg_method == DetectionOptions.MMI_GRADIENTDESCENT:
            rego = RegistrationMutualInformationGradientDescent(target_geo_im, fixed_geom_im, roi_template,
                                                                centre_images=centre_images)
        elif reg_method == DetectionOptions.COREL_AXIGRID_NBEST_GRADIENTDESCENT:
            rego = RegistrationAxialRotGridThenNBestGradDecnt(target_geo_im, fixed_geom_im, roi_template,
                                                              centre_images=centre_images)
        assert rego is not None, "RegistrationTwoStage::generate_registration_instance(): " \
                                 "invalid registration method, %s" % str(reg_method)
        return rego

