# show additional dataset
# bob afwata
# 13/05/2025
import sys
sys.path.append('..')
from dataset import sequence

#loaad the text file with the words to be used as seed. set seed size to 1984
(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('../dataset/addition.txt', seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)
# (45000, 7) (45000, 5)
# (5000, 7) (5000, 5)

#train the model 
print(x_train[0])
print(t_train[0])
# [ 3  0  2  0  0 11  5]
# [ 6  0 11  7  5]

#join the words to create the sentense and print
print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))
# 71+118
# _189
