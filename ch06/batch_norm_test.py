# Python program to perform batch training of a model using 
# Stochastic Gradient Descent Method (SDG) and Adam Modifiers.
# Bob Afwata <bafwata@gmail.com>
# 27/3/2025
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist # Load the MNIST data ,MUltilayer ,SGD and Adam optimizers
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam # import the SGD and Adam optimizers.

#load the mnist data and divide it into training and testing data.
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
#set the training batch vector to 1000 
x_train = x_train[:1000]
t_train = t_train[:1000]

#set the Maximum epoch to 20 and batch size of 100 ,learning rate of the model to 0.01
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

#function to train the model.
def __train(weight_init_std):
    #instantiate the batch normal MultilayerNet network , set the batch to normal.
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)  # use SGD optimizer
    
    train_acc_list = []  #create the training list 
    bn_train_acc_list = [] # create the batch normal training account list.
    
    iter_per_epoch = max(train_size / batch_size, 1) # use the max function to calculat the iterations per epoch.
    epoch_cnt = 0
    
    #use a for loop to generate the batch masks 
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size) #randomnly generate the batch masks.
        x_batch = x_train[batch_mask] # populate the x_batch vector with the x_train
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)  # calculate the network gradients for each network.
            optimizer.update(_network.params, grads)     # update the values of the optimizer.
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train) #calculate the network accuracy.
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 그래프 그리기==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    #diaplay all the 16 (4 x 4 ) subplots of the data .
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()   #show all the plots and subplots 
