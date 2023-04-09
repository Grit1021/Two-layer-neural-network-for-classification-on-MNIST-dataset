# Two-layer-neural-network-for-classification-on-MNIST-dataset

## Contents
- Tasks
   - Training
   - Hyperparameter tuning
   - Test
- Requirements
- Guide


## Tasks
Construct a two-layer neural network classifier including the three parts below.

### Training
- Activation functions
- Back propogation with loss and gradient computation
- Learning rate decay
- L2-regularization
- Optimizer SGD
- Model saving

### Hyperparameter Tuning
- Hidden layer size
- Learning rate 
- Regularization

### Test
Load the training model with tuned hyperparameters and obtain the classification accuracy.

Utilize the MNIST dataset, more information can be acceessed via http://yann.lecun.com/exdb/mnist/.

DO NOT use Pytorch,TensorFlow or any deep-learning python packages. NumPy is allowed.

Upload your code onto your github repository with instructions on training and testing process in a README file. The trained model should be uploaded onto a cloud drive.



## Requirements
1. Python 3
2. NumPy
3. Tqdm

## Guide
- Network.py

This is the Neural Network class. One can design and improve DNN models with it.

- plots.py

The function for plotting the training history which needs Matplotlib.

- train.py

It gives an example for training the model with Network.py and plotting the loss functions with plots.py. With a givn random seed, you can also obtain the result with 99.998% accuracy on the training dataset and 98.37% accuracy on the test dataset. In this python file, it also includes the functions for loading the dataset and saving the model.

- hp_search.ipynb

This Jupyter notebook gives a visualization of the training result and the whole process for hyperparameter searching. The weights of the first layer in the neural network can be visualized via PCA. 

- Model

A trained model with 98.37% accuracy on the test dataset is available at the [cloud drive](https://pan.baidu.com/s/1wIt9RMwZCdqEtjD0jDOigA?pwd=cvcv) with extracting password 'cvcv'.

Use the following code "nn = Network.load(r'mymodel.txt')" to load the model.


- Dataset

The MNIST dataset can be downloaded in the form of pkl.gz via the [link](https://academictorrents.com/details/323a0048d87ca79b68f12a6350a57776b6a3b7fb). It has been split into training set, validation set and test set with 50000,10000 and 10000 figures respectively.

- neural_network_pj1

It gives the 
