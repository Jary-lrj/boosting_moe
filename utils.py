import pandas as pd
import numpy as np
import tensorflow as tf
from neg_sampler import NegativeSampler


def build_sequence(data_path, length=51, num_negatives=5, item_num=22885):
    data = pd.read_csv(data_path)

    # 1. 预处理和排序
    data = data.sort_values(by=["userId", "timestamp"])
    data = data[["userId", "movieId", "rating"]]

    # 2. 过滤评分和活跃用户
    data = data[data["rating"] > 3]
    data = data.groupby("userId").filter(lambda x: len(x) >= 5)

    # 3. 编号 movieId，0 用作 padding
    movieId_list = data["movieId"].unique()
    movieId_map = {mid: idx + 1 for idx, mid in enumerate(movieId_list)}  # 从 1 开始，0 保留给 padding
    data["movieId"] = data["movieId"].map(movieId_map)

    # 4. 构建序列（每个用户一条，按时间顺序）
    user_sequence = []
    for userId, group in data.groupby("userId"):
        seq = group["movieId"].values[:length]
        user_sequence.append(seq)

    # 5. 用 0 在前方 padding，统一长度
    user_sequence = tf.keras.preprocessing.sequence.pad_sequences(user_sequence, maxlen=length, padding="pre", value=0)

    # 6. 构建输入和标签
    inputs = user_sequence[:, :-1]  # shape: (num_users, length - 1)
    labels = user_sequence[:, -1]  # shape: (num_users,)

    # 构建用户历史（用于负采样时排除）
    user_histories = [set(seq) - {0} for seq in inputs]  # 去掉 padding 0
    sampler = NegativeSampler(item_num=item_num - 1, num_negatives=num_negatives)  # -1 是因为 0 是 padding
    neg_samples = sampler.sample(user_histories, labels)  # shape: (batch_size, num_negatives)

    # 让inputs和labels成为张量
    inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return inputs, labels, neg_samples


if __name__ == "__main__":
    data_path = "./versions/1/rating.csv"
    inputs, labels = build_sequence(data_path)
