# Variational-Autoencoder-VAE

To Run the generating MNIST images Project, please follow the instructions.

Python 3.7.6

Virtual environment is suggested.


install the packages provided in requirements

-Requirements

import tensorflow as tf  ----> TENSORFLOW 1.15
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data ----> TO INSTALL THE MNIST DATASET


We install the version of the tensorflow 1.15, 

-main.py

In main file, we have training and model saver. Main imports the model and reading data functions from model.py

Data will be installed, when we run the main file to the same directory if it is not exist, else folder should be named as MNIST_data,
and should include:

t10k-images-idx3-ubyte.gz, 
t10k-labels-idx1-ubyte.gz, 
train-images-idx3-ubyte.gz, 
train-labels-idx1-ubyte.gz


To run the main file please use the following line from the terminal which directs to the project file.

python main.py

At the end of the training checkpoint file are extracted by tensorflow. During training, epoch, total loss, reconstruction loss and regularization 
will be printed to the console. Model is saved to the same directory into 3 pieces:

model.data-00000-of-00001
model.index
model.meta


-generator.py

In generator file, we call the saved model 

to call the generator file from the terminal, one may use the following line:

python generator.py

It will show and save the 100 generated images, named as generated_images.png ; in each run generated images will be different than each other.


There might be warnings in the beginning of both generator.py and main.py, if you have also tensorflow 2, or another version of python in your virtual environment. 
Please ignore them.

-model.py

There is nothing to do with the model file for training and image generation. It includes encoder and decoder functions
