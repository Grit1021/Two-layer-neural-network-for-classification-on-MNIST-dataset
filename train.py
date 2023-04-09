'''
Date:3/18/2023
Author: Yuxin Li
'''
import numpy as np
from Network import Network # my Network class
from plots import plots       # self-designed plots functions
import pickle
import gzip
import time

# data loader
time_start = time.time()
def load_data(path):
    f = gzip.open(path, 'rb')  #open the zip file
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

path='mnist.pkl.gz'
# training_data  : (50000 * 784, 50000)
# validation_data: (10000 * 784, 10000)
# test_data      : (10000 * 784, 10000)
training_data , validation_data , test_data = load_data(path)
data_x = training_data[0]
data_y = np.eye(10)[training_data[1]]    # onehot encoding
valid_x = validation_data[0]
valid_y = np.eye(10)[validation_data[1]] # onehot encoding


# set up a neural network
np.random.seed(20230318)
nn = Network(hidden_size = [784,1000,10], acts = ['relu','sigmoid'], lr = 3e-2,
                regws = None, regbs = None)
# nn = Network(hidden_size = [784,1000,10], acts = ['sigmoid','sigmoid'], lr = 3e-2,
#                 regws = None, regbs = None)


# if you want to load your model, use the code here
# nn = Network.load(r'mymodel.txt')


# train the model
result = nn.fit(data_x, data_y, epochs = 20, batch_size = 40, loss_func = 'CE', 
                valid_x = valid_x, valid_y = valid_y, valid_freq = 1)


# read the training result
# losses = result['loss']
# accs   = result['acc'] 


# save your model here
nn.save(r'mymodel.txt')


# check the accuracy on training, validation and testing data
print('-'*30 + '\nAccuracy on Training Data')
nn.predict(data_x, data_y, batch_size = 40, verbose = True)

print('-'*30 + '\nAccuracy on Validation Data')
nn.predict(valid_x, valid_y)

print('-'*30 + '\nAccuracy on Testing Data')
nn.predict(test_data[0], np.eye(10)[test_data[1]])
print(end = '')


print(f'Finished within {time.time() - time_start} seconds')


# plot the loss and accuracy curves
plots(result)
