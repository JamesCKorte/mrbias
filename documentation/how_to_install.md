*Authors: James Korte &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date Modified: 07/03/2025*

# Tutorial: How to install MR-BIAS
This tutorial details the steps required to install MR-BIAS. Firstly we provide the requirements of the software and what you need to already have running on your computer. Followed by how to download the software, install associated packages, and run the software to check everything is in order. 

## Requirements
To install and run MR-BIAS on your computer you will need to
- install git on your computer ([a guide for how to do this](https://github.com/git-guides/install-git))
- install python on your computer ([download from here](https://www.python.org/downloads/))

The MR-BIAS software has been tested with Python v3.7 we recommend [creating a virtual python environment](https://docs.python.org/3/library/venv.html) to install MR-BIAS into.

## Installation
Now that you meet the software requirements, you're ready to download MR-BIAS. The software is hosted on github and can be downloaded with the following git command
```
git clone https://github.com/JamesCKorte/mrbias.git
```
after cloning the repository to your computer, change directory into the downloaded folder
```
cd mrbias
```
The software makes use of existing python packages which can be installed with the following command. This may take some time depending on which packages you already have installed in your python environment.
```
python -m pip install -r requirements.txt
```


## Run the software
Now that you've got mrbias and some associated python packages installed, we will get you to run a basic example to check everything is working. The MR-BIAS repository which you cloned includes some example relaxometry and diffusion data. The repository also includes an example script to analyse the relaxometry data. To run this example script, navigate to the "examples" folder and launch the script with the following commands
```
cd examples
python relaxometry_example_1.py
```
The example program analyses two ISMRM/NIST system phantom image datasets and will generate two reports which can be found in the "mrbias\examples\output" folder. If everything runs smoothly, open up one of the PDF reports and check things look ok. For more information on the expected report contents for a relaxometry or diffusion analysis please refer to the
- [Relaxometry analysis tutorial](./basic_analysis_relaxometry.md)
- [Diffusion analysis tutorial](./basic_analysis_diffusion.md)

## Troubleshooting
If you have any issues when running the software please double check you have a suitable version of Python installed and have correctly installed all associated packages. If you are still having any issues then [please get in touch with us for assistance](./contact.md).







 <br> <br> <br> <br> <br> <br>

### Revision history

|     Date      |   Author    | Changes                       |
|:-------------:|:-----------:|:------------------------------|
| 07 March 2025 | James Korte | Created installation tutorial |




