#Code to generate text using Reccurent neural network 
# Bob Afwata bafwata@gmail.com
# 11/05/2025
import sys
sys.path.append('..')
from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus) # get the size of the corpus


model = BetterRnnlmGen()
model.load_params('../ch06/BetterRnnlm.pkl') # extract and load the parameters from the pickle file

# start 문자와 skip 문자 설정
# provide the words to start with as a template and symbols and delimiters to separate the words.
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print(txt)


model.reset_state()

start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

#generate the words starting from the word ids
word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids  #join the words to form a sentence.
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')   #repalce the delimiters
print('-' * 50)
print(txt) # print the output sentense.
