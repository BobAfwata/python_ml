# Python Code to compare the results of using various optimizers during model training with mnist data
# Bob Afwata <bafwata@gmail.com>
# 27/3/2025
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0. MNIST 데이터 읽기==========
# Loading the mnist data and dividing it into traing data and test data.
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128    #set the batch size to 128
max_iterations = 2000 #set the number of iterations to 2000


# 1. 실험용 설정==========
#instantiate the various optimizers 
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {} # create an empty dictionary for the variuos networks
train_loss = {} # create an empty dictionary called train loss
for key in optimizers.keys():  #instantiate the various networks 
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size) #randomnly generate the btch mask.
    x_batch = x_train[batch_mask] # populate the x_batch and t_batch vectors with the training data.
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():    # use the varius optimizers and calculate the gradiens in each case.
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
#Dictionary for the varius markers to be used in graphing . This is to represenr the varius types of optimizers.
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"} 
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.title("Graph to Show the Results of Use of various Optimizers.")
plt.legend()
plt.show()
