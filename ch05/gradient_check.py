# import MNIST dataset,numpy 
#A program to check the gradients 
# Bob Afwata<bafwata@gmail.com>
import sys, os
sys.path.append(os.pardir)  # set the path for the files
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기 
# Load the training and test data.
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#initilize the neural network.
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
#Evaluate the gradients 
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
