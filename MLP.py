import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, num_items, embed_dim):
        super(MLP, self).__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim

        # 嵌入层：将item ID映射到嵌入空间
        self.item_embedding = tf.keras.layers.Embedding(input_dim=num_items, output_dim=embed_dim)
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.pred = tf.keras.layers.Dense(num_items, activation="softmax")

    def call(self, inputs, training=False):
        item_ids = inputs
        x = self.item_embedding(item_ids)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
