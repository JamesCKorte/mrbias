from setuptools import setup

## NOTE TO SELF
##
## python setup.py sdist
## twine upload dist/*

setup(
   name='mrbias',
   version='1.0.2',
   description='Magnetic Resonance BIomarker Assessment Software (MR-BIAS) is an automated tool for extracting quantitative MR parameters from NIST/ISMRM system phantom images',
   license_files = ('LICENSE.txt',),
   author='James Korte',
   author_email='korte.james@gmail.com',
   url='https://github.com/JamesCKorte/mrbias',
   packages=['mrbias'],  #same as name
   package_data={"mrbias": ["roi_reference_values/caliber_system_phantom/*.csv",
                            "roi_reference_values/caliber_system_phantom/batch1_sn_lte_130-0041/*.csv",
                            "roi_reference_values/caliber_system_phantom/batch2_sn_gte_130-0042/*.csv",
                            "roi_reference_values/caliber_system_phantom/batch2p5_sn_gte_130-0133/*.csv",
                            "roi_detection_templates/siemens_skyra_3p0T/dicom/*",
                            "roi_detection_templates/siemens_skyra_3p0T/*.yaml"]},
   install_requires=['Pillow', 'lmfit', 'SimpleITK', 'numpy', 'pandas', 'scipy', 'pyyaml', 'pydicom', 'matplotlib', 'seaborn', 'reportlab'], #external packages as dependencies
)
