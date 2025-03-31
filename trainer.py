from MoE import MoE
import tensorflow as tf
from BoostingMoE import BoostingMoE
from utils import build_sequence
import datetime

# data preprocessing

if __name__ == "__main__":

    # parameters
    num_items = 22885
    embed_dim = 64
    num_heads = 1
    num_layers = 2
    batch_size = 64

    # 构造数据集
    data_path = "./versions/1/rating.csv"
    inputs, pos_labels, neg_labels = build_sequence(data_path, num_negatives=5, item_num=num_items)

    # 将 inputs、neg_labels 合成 x，pos_labels 作为 y
    x = {"inputs": inputs, "neg_labels": neg_labels}
    y = pos_labels

    # 构建 Dataset：每个样本是 (x_dict, y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # 分割训练集、验证集、测试集
    total_size = len(inputs)
    valid_size = int(0.1 * total_size)
    test_size = int(0.1 * total_size)
    train_size = total_size - test_size - valid_size

    dataset = dataset.shuffle(buffer_size=total_size, seed=42)
    train_dataset = dataset.take(train_size).batch(batch_size)
    val_dataset = dataset.skip(train_size).take(valid_size).batch(batch_size)
    test_dataset = dataset.skip(train_size + valid_size).batch(batch_size)

    # 使用 tensorboard方便记录训练过程
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # 模型训练
    model = BoostingMoE(num_items, embed_dim, num_heads, num_layers, num_experts=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=[tensorboard_callback, early_stopping],
    )

    evaluate_metrics = model.evaluate(test_dataset)
    print(evaluate_metrics)

    predictions = model.predict(test_dataset)
    print(predictions)

    model.summary()
