import os
import time
import argparse
import tensorflow as tf
from neg_sampler import WarpSampler
from SASRec import SASRec as Model
from tqdm import tqdm
from utils import *
import keras


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) // args.batch_size  # 使用 // 进行整除
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

# 设置 GPU 选项
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_logical_device_configuration(physical_devices[0],
                                               [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)

# TensorFlow 2.x 中没有初始化变量的步骤
# 使用 tf.keras.Model 和 tf.Variable 时不需要显式的初始化步骤

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            with tf.GradientTape() as tape:
                auc, loss = model(u, seq, pos, neg, is_training=True)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer = keras.optimizers.Adam(learning_rate=args.lr)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('')
            print(f'epoch:{epoch}, time: {T:.2f}(s), valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), '
                  f'test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
except Exception as e:
    print(f"An error occurred: {e}")
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
