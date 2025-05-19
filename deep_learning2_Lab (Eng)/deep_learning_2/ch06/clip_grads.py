# clipping the training gradients 
# Bob Afwata bafwata@gmail.com
# 29/4/2025

import numpy as np

#generate the random 3x3 data and moderate by multipying by 3
dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0 # setting the  maximum normalization size to 5

# function to clip the gradients 
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

#print the outputs of the gradients before and after the clipping of gradients operation.
print('before:', dW1.flatten())
clip_grads(grads, max_norm)
print('after:', dW1.flatten())
