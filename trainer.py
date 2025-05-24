import os
import time
import tensorflow as tf
from neg_sampler import WarpSampler
from SASRec import SASREC
from tqdm import tqdm
from utils import *
import keras
from recommenders.utils.timer import Timer

print(f"Tensorflow version: {tf.__version__}")
tf.get_logger().setLevel('ERROR')

num_epochs = 5
batch_size = 128
seed = 42

data_dir = './data'
dataset = 'ml-1m'

lr = 1e-3
maxlen = 50
num_blocks = 2
hidden_units = 100
num_heads = 1
dropout_rate = 0.1
l2_emb = 0.0
num_neg_test = 100
model_name = 'sasrec'

input_file = os.path.join(data_dir, f'{dataset}.txt')

data = SASRecDataSet(inputfile=input_file)
data.split()

num_steps = int(len(data.user_train / batch_size))
cc = 0.0
for u in data.user_train:
    cc += len(data.user_train[u])
print('%g Users and %g items' % (data.usernum, data.itemnum))
print('average sequence length: %.2f' % (cc / len(data.user_train)))

model = SASREC(item_num=data.itemnum,
               seq_max_len=maxlen,
               num_blocks=num_blocks,
               embedding_dim=hidden_units,
               attention_dim=hidden_units,
               attention_num_heads=num_heads,
               dropout_rate=dropout_rate,
               conv_dims=[100, 100],
               l2_reg=l2_emb,
               num_neg_test=num_neg_test
               )

sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)

with Timer() as train_time:
    t_test = model.train(data, sampler, num_epochs=num_epochs, batch_size=batch_size, lr=lr, val_epoch=6)

print('Time cost for training is {0:.2f} mins'.format(train_time.interval / 60.0))

res_syn = {"ndcg@10": t_test[0], "Hit@10": t_test[1]}
print(res_syn)


