#Python Module to implement simple Reccurrent Neural Network (RNN)
# Bob Afwata bafwata@gmail.com
# 29/4/2025

#import numpy and common time layers 
import sys
sys.path.append('..')
import numpy as np
from common.time_layers import *


#Simple RNN class definition 
class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size  #define the Simple RNN with vocabulary size,wordvector size and hidden size.
        rn = np.random.randn #randomy generate the numbers 

        # 가중치 초기화
        embed_W = (rn(V, D) / 100).astype('f') # create the word embeddings with 'f' as the delimiter or separator
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')  #define the affine layers 
        affine_b = np.zeros(V).astype('f')

        # 계층 생성 # put together the Time Ebdeddings ,RNN and Affine layers 
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss() #calculate the Softmax loss function.
        self.rnn_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):      # perform forward propagation from one layer to the next while calculating the loss function
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1): # perform back propagation from one layer to the next while calculating the loss function
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers): # for this case use reversed() function to navigate in opposite direction
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()
