# Preprocessing_xray_scattering_images
Preprocessing pipeline for xray scattering images to achieve the best possible results in feature detection

# Disclaimer

This repository mainly serves as a public archive for the code used in my Master Thesis and is not yet optimized for the use by other researchers. Future updates will improve the functionality and usability of the program.

# Main Dependencies
To be able to run the code, a python 3.7 installation with the following dependencies is required:
- tqdm
- torch
- numpy
- matplotlib
- mpld3
- datetime
- h5py
- pathlib

# Usage
This Pipeline is for preprocessing of xray scattering images with somehow bad contrasts.

The target is the best possible performance of feature detection. 
Because it is not possible to use a common function to enhance the images' quality (because of hot pixel and very various contrasts), an approach with supervised learning is made.

In this pipeline the linear clipping is used.
Therefore the images have to be labeled with the values for the lower and the upper limit.

Store it as an .h5 file (according to read_data.py).

Load the data with 

    read_data()

And create datasets of random patches of the experimental images (to get rid of the hot-pixel zones):

    data_loader = PatchLoader.from_data(data, batch_size, bins=bin_size)
With 3 different data:    
- train_loader
- validation_loader
- test_loader 
    
To load the model

    model = SimpleModel().cuda()
    
And to start the training process

    patch_trainer = PatchTrainer(model, train_loader, validation_loader,'path_to_save_the_model',  learning_rate)
    patch_trainer.train(num_of_epochs, num_of_plot_intervall)
