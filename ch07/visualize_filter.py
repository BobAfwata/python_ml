# python program to visualize / display the filter 
# Bob Afwata <bafwata@gmail.com>
# 29/3/2025

#import numpy ,matplotlib and SimpleConvNet
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet

#function to deiplay the filter . set the nx to 8, margin of 3 and scale of 10
def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx)) # calculate the ny by converting them by rounding up the values and convering them to integers


    fig = plt.figure() #plot the figures with the left,right,bottom top paddings etc.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show() # show the plots

#instantiate the Simple Convolutional network 
network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params['W1'])  # read the weights from the network parameters 

# 학습된 가중치
network.load_params("params.pkl") # load the network parameters from the params.pkl pickle file
filter_show(network.params['W1']) # show the filter from the network parameters.
