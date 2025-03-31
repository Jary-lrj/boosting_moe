import tensorflow as tf

# build SASRec model


class SASRec(tf.keras.Model):

    def __init__(self, num_items, embed_dim, num_heads, num_layers, dropout_rate=0.2):
        super(SASRec, self).__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 嵌入层：将item ID映射到嵌入空间
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embed_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            mask_zero=True,
        )

        self.transformer_blocks = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(num_items)

    def call(self, inputs, training=False):
        item_ids = inputs

        # Embedding lookup
        x = self.item_embedding(item_ids)

        # Add positional encoding
        seq_len = x.shape[1]
        position_embeddings = self.add_position_encoding(seq_len)
        x += position_embeddings

        # Transformer blocks
        for i in range(self.num_layers):
            attention_output = self.transformer_blocks[i](x, x)
            x = self.layer_norm(x + attention_output)
            x = self.dropout(x, training=training)

        # 最后一步：提取最后一个时间步
        last_step = x[:, -1, :]
        last_step = self.output_layer(last_step)
        return last_step

    def add_position_encoding(self, seq_len):
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
        dim = tf.range(self.embed_dim, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dim // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = position * angle_rates  # (seq_len, embed_dim)
        angle_rads = tf.where(tf.cast(dim % 2, tf.bool), tf.cos(angle_rads), tf.sin(angle_rads))
        pos_encoding = angle_rads[tf.newaxis, ...]
        return pos_encoding
