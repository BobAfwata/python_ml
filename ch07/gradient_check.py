# Python program to check the gradients during training
# Bob Afwata <bafwata@gmail.com>
# 29/3/2025

#import numpy and simpleConvNet modules
import numpy as np
from simple_convnet import SimpleConvNet

#instantiate the simpleconvultion network . Set the input to 10x10
#set the filter_num = 10, fiter_size to 3 and padding to 0,stride 1,hidden_size to 10 and output size of 10 ,
# instantiate the weight_init_std to 0.01.
network = SimpleConvNet(input_dim=(1,10, 10), 
                        conv_param = {'filter_num':10, 'filter_size':3, 'pad':0, 'stride':1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)

#generate 100 randonm  numbers and reshape the vector and array to be use for training and test.
X = np.random.rand(100).reshape((1, 1, 10, 10))
T = np.array([1]).reshape((1,1))

# calculate the numerical gradient of the network using the X and T data sets generated above
grad_num = network.numerical_gradient(X, T)
grad = network.gradient(X, T) # find the network gradient values.

# print the gradients calculated
for key, val in grad_num.items():
    print(key, np.abs(grad_num[key] - grad[key]).mean())