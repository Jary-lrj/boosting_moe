import tensorflow as tf
import keras


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True, scope="embedding"):
    with tf.name_scope(scope):
        lookup_table = keras.layers.Embedding(vocab_size, num_units, )

        if zero_pad:
            lookup_table.weights[0] = tf.concat([tf.zeros((1, num_units)), lookup_table.weights[0][1:, :]], axis=0)

        outputs = lookup_table(inputs)

        if scale:
            outputs *= (num_units ** 0.5)

    return outputs


def multihead_attention(q, k, num_units=None, num_heads=8, dropout_rate=0, is_training=True, causality=False):
    if num_units is None:
        num_units = q.shape[-1]

    attention_layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_units)

    if causality:
        mask = tf.linalg.band_part(tf.ones((q.shape[1], k.shape[1])), -1, 0)
        mask = tf.expand_dims(mask, axis=0)
    else:
        mask = None

    output = attention_layer(query=q, value=k, key=k, attention_mask=mask)
    return output


def feedforward(inputs, num_units=[2048, 512], dropout_rate=0.2, is_training=True):
    x = keras.layers.Dense(num_units[0], activation='relu')(inputs)
    x = keras.layers.Dropout(dropout_rate)(x, training=is_training)
    x = keras.layers.Dense(num_units[1])(x)
    x = keras.layers.Dropout(dropout_rate)(x, training=is_training)

    return x + inputs  # 残差连接


class SASRec(keras.Model):
    def __init__(self, usernum, itemnum, args):
        super(SASRec, self).__init__()
        self.item_embedding = keras.layers.Embedding(input_dim=itemnum + 1, output_dim=args.hidden_units)
        self.pos_embedding = keras.layers.Embedding(input_dim=args.maxlen, output_dim=args.hidden_units)
        self.dropout = keras.layers.Dropout(args.dropout_rate)

        self.blocks = []
        for _ in range(args.num_block):
            self.blocks.append(
                [keras.layers.LayerNormalization(),
                 multihead_attention,
                 keras.layers.LayerNormalization(),
                 feedforward
                 ]
            )

        self.final_norm = keras.layers.LayerNormalization()

    def call(self, input_seq, pos, neg, training):
        mask = tf.cast(tf.not_equal(input_seq, 0), tf.float32)
        mask = tf.expand_dims(mask, axis=0)

        seq_emb = self.item_embedding(input_seq)
        positions = tf.tile(tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0), [tf.shape(input_seq)[0], 1])
        pos_emb = self.pos_embedding(positions)
        seq = seq_emb + pos_emb

        seq = self.dropout(seq, training=training)
        seq *= mask

        for norm1, attention, norm2, ffn in self.blocks:
            seq = attention(norm1(seq), norm1(seq), num_units=self.args.hidden_units,
                            num_heads=self.args.num_heads, dropout_rate=self.args.dropout_rate,
                            is_training=training, causality=True)
            seq = ffn(norm2(seq), [self.args.hidden_units, self.args.hidden_units],
                      dropout_rate=self.args.dropout_rate, is_training=training)
            seq *= mask

        seq = self.final_norm(seq)

        seq_flat = tf.reshape(seq, [-1, self.args.hidden_units])
        pos = tf.reshape(pos, [-1])
        neg = tf.reshape(neg, [-1])

        pos_emb = tf.nn.embedding_lookup(self.item_embedding.weights[0], pos)
        neg_emb = tf.nn.embedding_lookup(self.item_embedding.weights[0], neg)

        pos_logits = tf.reduce_sum(pos_emb * seq_flat, axis=-1)
        neg_logits = tf.reduce_sum(neg_emb * seq_flat, axis=-1)

        istarget = tf.cast(tf.not_equal(pos, 0), tf.float32)
        loss = tf.reduce_sum(
            - tf.math.log(tf.nn.sigmoid(pos_logits) + 1e-24) * istarget -
            tf.math.log(1 - tf.nn.sigmoid(neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        auc = tf.reduce_sum(
            ((tf.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        return loss, auc

    def predict_logits(self, input_seq, candidate_items):
        seq_emb = self.item_embedding(input_seq)
        positions = tf.tile(tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0), [tf.shape(input_seq)[0], 1])
        pos_emb = self.pos_embedding(positions)
        seq = seq_emb + pos_emb

        seq = self.dropout(seq, training=False)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq *= mask

        for norm1, attention, norm2, ffn in self.blocks:
            seq = attention(norm1(seq), norm1(seq), num_units=self.args.hidden_units,
                            num_heads=self.args.num_heads, dropout_rate=self.args.dropout_rate,
                            is_training=False, causality=True)
            seq = ffn(norm2(seq), [self.args.hidden_units, self.args.hidden_units],
                      dropout_rate=self.args.dropout_rate, is_training=False)
            seq *= mask

        seq = self.final_norm(seq)
        seq_last = seq[:, -1, :]

        candidate_emb = tf.nn.embedding_lookup(self.item_embedding.weights[0], candidate_items)
        logits = tf.matmul(seq_last, candidate_emb, transpose_b=True)
        return logits
