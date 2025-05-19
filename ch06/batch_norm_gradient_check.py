# Python Program to Check the gradients in a model using batches 
# Bob Afwata <bafwata@gmail.com>
# 27/3/2025

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist # import the MNIST data and Multilayer Module
from common.multi_layer_net_extend import MultiLayerNetExtend

# 데이터 읽기
# load the mnist data and divide it into training and test data 
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#instantiate the MltilayerNet Extended network with an input of 784, hidden layer of 100 and output size of 10 
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], output_size=10,
                              use_batchnorm=True)

#copy the vector x_train into x_batch and t_train into t_batch[]
x_batch = x_train[:1]
t_batch = t_train[:1]

# use backpagation to calculate the network gradient and numerical gradient.
grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_gradient(x_batch, t_batch)

#using the formulae to calculate the gradient and print the results.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
