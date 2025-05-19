# Python code to generate text 
# Bob Afwata bafwata@gmail.com
# 12/05/2025
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

#instantiate the Recurrent neural network
model = RnnlmGen()
model.load_params('../ch06/Rnnlm.pkl') #load thee parameters from the pickle file.

# start 문자와 skip 문자 설정
# set the starting word to "you" to be used as a template to generatee the sentence
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$'] # the delimiters or words to skip 
skip_ids = [word_to_id[w] for w in skip_words] # iterate through the words and get the ids of words to skip
# 문장 생성
word_ids = model.generate(start_id, skip_ids) #generate the words while skipping whats to be skipped.
txt = ' '.join([id_to_word[i] for i in word_ids]) # join the words to form the sentense
txt = txt.replace(' <eos>', '.\n')
print(txt) # print the final words .
