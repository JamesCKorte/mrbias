# MR-BIAS
Magnetic Resonance BIomarker Accreditation Software (MR-BIAS) is an automated tool for extracting quantitative MR parameters from NIST/ISMRM system phantom images. The software has been tested on images from a 3T Siemens Skyra, we would like to add support for images from other scanners, for enquiries please contact us (james.korte@petermac.org). 
### Citation
A technical note describing the software is under review, a relevant citation will be added here upon publication.
### Minimum Requirements
We have tested the software with Python v3.7 and recommend installing into a virtual environment.

## Installation
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

## Usage
A basic example and test data are included, to verify MR-BIAS is running on your system, please run the example script with
```
python mr_bias_example_1.py
```
The example program analyses two ISMRM/NIST system phantom image datasets and will generate two reports which can be found in the "mrbias\output" folder. 








