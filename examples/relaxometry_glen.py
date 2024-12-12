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
  14-July-2023  :               (Stanley Norris) : Example added for collaborator
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
configuration_filename = os.path.join("..", os.path.join("config", "relaxometry_glen_config.yaml"))
# specific the dicom directories to analyse
dicom_directory_a = r"Glenn you will need to change this path to a data directory on your computer"

# create a MRBIAS analysis object
mrb = MRBIAS(configuration_filename, write_to_screen=True)
# run the analysis (output will be created in the "output_directory" specified in the configuration file)
mrb.analyse(dicom_directory_a)