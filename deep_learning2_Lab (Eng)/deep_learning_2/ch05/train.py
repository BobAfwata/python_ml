# Python code to train a Simple RNN
#Bob Afwata bafwata@gmail.com
# 29/4/2025 

import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# 하이퍼파라미터 설정
#set the training parameters ie batch size,wordvector size,hidden sie max epoch etc.
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN의 은닉 상태 벡터의 원소 수
time_size = 5  # RNN을 펼치는 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train') #loa the training model.
corpus_size = 1000  # 테스트 데이터셋을 작게 설정
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # 입력
ts = corpus[1:]  # 출력（정답 레이블）

# 모델 생성
#instantiate the model using the SimpleRNNML module and vocabulary_size,wordvector size,hidden sizes
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)  # use Stochatic Gradient Descent SGD Optimizer
trainer = RnnlmTrainer(model, optimizer)  #instantiate the trainer 

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot() # plot the outputs of the training.
