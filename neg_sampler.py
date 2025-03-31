import tensorflow as tf
import numpy as np


class NegativeSampler:
    def __init__(self, item_num, num_negatives=1, seed=None):
        self.item_num = item_num
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

    def sample(self, user_histories, pos_items):
        batch_size = len(pos_items)
        neg_items = []

        for i in range(batch_size):
            negatives = []
            while len(negatives) < self.num_negatives:
                sampled = self.rng.integers(1, self.item_num + 1)  # item 从 1 开始编号
                if sampled not in user_histories[i] and sampled != pos_items[i]:
                    negatives.append(sampled)
            neg_items.append(negatives)

        return tf.convert_to_tensor(neg_items, dtype=tf.int32)
