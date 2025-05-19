# Python code to perform filtring on gray scale image
# Bob Afwata <bafwata@gmail.com>
# 29/3/2025
#import numpy ,matplotlib SimpleConvNet and Convolution.
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet # import SimpleConvolutio net
from matplotlib.image import imread # import imread to read the images
from common.layers import Convolution

#fucntion to show the filter perfomed on the image
def filter_show(filters, nx=4, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(show_num / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(show_num):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')

#instantiate SimpleConvNet , use an input of 28x28 pixel images,perfom filtering
# set fiter number to 30,filter size to 5 ,hidden layer of 100,output= 10 and weigh_init_tstandard of 0.01

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 학습된 가중치
# load the network parametrs file to be used
network.load_params("params.pkl")

filter_show(network.params['W1'], 16)

#read the gray scale image lena_gray.png from the datasets folder
img = imread('../dataset/lena_gray.png')
#reshape the image to fit our requirements.
img = img.reshape(1, 1, *img.shape)
# st the figure for the plot
fig = plt.figure()

w_idx = 1

for i in range(16):
    w = network.params['W1'][i] #|get the weights from the network parameters.
    b = 0  # network.params['b1'][i]

    w = w.reshape(1, *w.shape)
    #b = b.reshape(1, *b.shape)
    #perform convolution using the weights and biases
    conv_layer = Convolution(w, b) 
    out = conv_layer.forward(img) # set the output layer fater performing forward convolution on the image
    # reshape the output.
    out = out.reshape(out.shape[2], out.shape[3])
    
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
