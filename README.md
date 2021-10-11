# Automated preprocessing of X-ray scattering images for contrast enhancement via trained neural network
Contrast enhancement pipeline for X-ray scattering images to automate preprocessing for linear clipping via trained neural network.

## Disclaimer

This repository mainly serves as a public archive for the code used in my Master Thesis and is not yet optimized for the use by other researchers. Future updates will improve the functionality and usability of the program.

## Main Dependencies
To be able to run the code, a python 3.7 installation with the dependencies of the file `requirements.txt` is required.

## Usage
This pipeline is for preprocessing of X-ray scattering images with bad contrasts.
Due to different experimental setups of diffraction patterns it is not possible to just find fixed parameters for classical contrast enhancement algorithms.

Here an approach with supervised learning is presented.

In this pipeline the linear clipping method is used.
For training many images had to be labeled with the values for the lower and the upper limit (normalized to 0,1).
This labeld dataset is not provided, but can be simple rebuilt with own or puplic datasets of X-ray scattering images (store it as .h5 file; according to read_data.py).

For testing the automated contrast enhancement follow the code in  

    main.py


Please feel free to contact me in case of problems, for questions or feedback. 
