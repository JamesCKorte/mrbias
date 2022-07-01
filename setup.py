from setuptools import setup

setup(
   name='mrbias',
   version='1.0',
   description='Magnetic Resonance BIomarker Assessment Software (MR-BIAS) is an automated tool for extracting quantitative MR parameters from NIST/ISMRM system phantom images',
   author='James Korte',
   author_email='korte.james@gmail.com',
   url='https://github.com/JamesCKorte/mrbias',
   packages=['mrbias'],  #same as name
   install_requires=['Pillow', 'lmfit', 'SimpleITK', 'numpy', 'pandas', 'scipy', 'pyyaml', 'pydicom', 'matplotlib', 'seaborn', 'reportlab'], #external packages as dependencies
)