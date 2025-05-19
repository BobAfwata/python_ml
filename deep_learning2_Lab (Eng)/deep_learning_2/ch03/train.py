# python program to train a simple word model.
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

#set the parameters such as word size ,hidden,batch sizes and epoch time
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

# sample word text to be used 
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
#get the size of the vocabulary of words to be used.
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# craete the model 
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam() # use ADAM optimers 
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot() # plot the output of the traing 

#print the outputs of the predicted word 
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
