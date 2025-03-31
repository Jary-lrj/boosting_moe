import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SASRec import SASRec
import datetime
# data preprocessing


def build_sequence(data_path, length=51):
    data = pd.read_csv(data_path)
    # sort by userId and timestamp
    data = data.sort_values(by=["userId", "timestamp"])
    # only keep userId, movieId and rating
    columns = ["userId", "movieId", "rating"]
    data = data[columns]
    # only keep rating greater than 3
    data = data[data["rating"] > 3]
    # only keep user who have more than 5 records
    data = data.groupby("userId").filter(lambda x: len(x) >= 5)
    # map movieId to a new continuous id starting from 1
    movieId_list = data["movieId"].unique()
    movieId_map = dict(zip(movieId_list, range(1, len(movieId_list) + 1)))
    print(f"num items: {len(movieId_list) + 1}")
    data["movieId"] = data["movieId"].map(movieId_map)
    # build sequence,if length is longer than the user's record, fill the sequence with 0
    user_sequence = []
    for userId, group in data.groupby("userId"):
        user_sequence.append(group["movieId"].values[:length])
    # padding sequence as [0, 0, ... x, y]
    user_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        user_sequence, maxlen=length, padding="pre", value=0)

    # build label
    label = user_sequence[:, -1]
    # build input
    input = user_sequence[:, :-1]
    # build tf dataset

    return input, label


if __name__ == "__main__":
    data_path = "./versions/1/rating.csv"
    data = build_sequence(data_path)

    # parameters
    num_items = 22885
    embed_dim = 64
    num_heads = 1
    num_layers = 2

    model = SASRec(num_items, embed_dim, num_heads, num_layers)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 使用 tensorboard方便记录
    # 带时间戳的log_dir
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # build tf dataset
    input, label = data
    x_train, x_test, y_train, y_test = train_test_split(input,
                                                        label,
                                                        test_size=0.2,
                                                        random_state=42)
    model.fit(x_train,
              y_train,
              epochs=100,
              batch_size=64,
              callbacks=[tensorboard_callback],
              validation_split=0.2)

    # output precision, recall, F1-score in evaluate
    from sklearn.metrics import classification_report
    y_pred = model.predict(x_test)

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int)

    report = classification_report(y_test, y_pred_classes)
    print(report)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"loss: {loss}, accuracy: {accuracy}")
