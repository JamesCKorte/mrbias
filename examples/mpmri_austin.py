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
02-August-2021  :               (James Korte) : Initial code for MR-BIAS v0.0
  23-June-2022  :               (James Korte) : GitHub Release   MR-BIAS v1.0
"""
import os
# Code to add the parent directory to allow importing mrbias core modules
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
if str(root) not in sys.path:
    sys.path.insert(1, str(root))
# import required mrbias modules
from mrbias import MRBIAS


# specify the configuration file to control the analysis
configuration_filename_a = os.path.join("..", os.path.join("config", "system_austin_config.yaml"))
configuration_filename_b = os.path.join("..", os.path.join("config", "diffusion_austin_config.yaml"))

# specific the dicom directories to analyse
dicom_directory_a = r"add path to relaxometry data"
dicom_directory_b = r"add path to diffusion data"

# create MRBIAS analysis objects
mrb = MRBIAS(configuration_filename_a, write_to_screen=True)
mrb.analyse(dicom_directory_a)

mrb = MRBIAS(configuration_filename_b, write_to_screen=True)
mrb.analyse(dicom_directory_b)