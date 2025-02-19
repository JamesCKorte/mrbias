# Welcome to MR-BIAS!
Magnetic Resonance BIomarker Accreditation Software (MR-BIAS) is an automated tool for extracting quantitative MR parameters from NIST/ISMRM system phantom images. The software has been tested on images from a 3T Siemens Skyra, we are currently working with the community to add support for images from other scanners, for enquiries please contact us (james.korte@petermac.org). 

[![PyPI Downloads](https://img.shields.io/pypi/dm/mrbias.svg?label=PyPI%20downloads)](
https://pypi.org/project/mrbias/)
[![Relaxometry Paper](https://img.shields.io/badge/DOI-10.1088%2F1361--6560%2Facbcbb-blue)](
https://doi.org/10.1088/1361-6560/acbcbb)

### Documentation
We recently shared the motivation and basic usage details of the software during [a presentation at the ESMRMB MRITogether conference](https://www.youtube.com/watch?v=QgFzDnjO4Jw&list=PLeDygc8TN_J65c0jM0ms__diTMylbEk9l&index=14&t=18m14s). We also provide some basic instructions for using the software as follows, 
- [Installation](./README.md#installation)
- [Basic Usage](./README.md#usage)
- [Tutorials](./README.md#modifying-the-software-to-your-needs)
    - [Tutorial 1: adding a new scanner or acquisition protocol](./documentation/adding_a_new_scanner.md)
    - [Tutorial 2: adding a new region of interest template for a different phantom](./documentation/adding_a_roi_template.md)
    - [Tutorial 3: adding a new signal model for curve fitting data](./documentation/adding_a_new_model.md)

### Citation
If you find our software useful for your research, please reference the following:

*Korte, J.C., Chin, Z., Carr, M., Holloway, L. and Franich, R.D.*, 2023. [Magnetic resonance biomarker assessment software (MR-BIAS): an automated open-source tool for the ISMRM/NIST system phantom](https://iopscience.iop.org/article/10.1088/1361-6560/acbcbb). Physics in Medicine and Biology.


# Installation
The software can be downloaded from github with the following command
```
git clone https://github.com/JamesCKorte/mrbias.git
```
after downloading, change directory into the downloaded folder
```
cd mrbias
```
The software makes use of existing python packages which can be installed with the following command. This may take some time depending on which packages you already have installed in your python environment.
```
python -m pip install -r requirements.txt
```

### Minimum Requirements
We have tested the software with Python v3.7 and recommend installing into a virtual environment.

# Usage
A basic example and test data are included, to verify MR-BIAS is running on your system, please run the test script from the "examples" folder with the command
```
python relaxometry_example_1.py
```
The example program analyses two ISMRM/NIST system phantom image datasets and will generate two reports which can be found in the "mrbias\examples\output" folder. 

## Modifying the software to your needs
We are working on a number of tutorials to help users modify the software and contribute to this project. So far these include
- [Tutorial 1: adding a new scanner or acquisition protocol](./documentation/adding_a_new_scanner.md)
- [Tutorial 2: adding a new region of interest template for a different phantom](./documentation/adding_a_roi_template.md)
- [Tutorial 3: adding a new signal model for curve fitting data](./documentation/adding_a_new_model.md)








