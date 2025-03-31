import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# data preprocessing
def build_sequence(data_path, length=51):

    data = pd.read_csv(data_path).sort_values(by=["userId", "timestamp"])
    data = data.loc[(data["rating"] > 3), ["userId", "movieId"]]
    grouped_data = data.groupby("userId").filter(lambda x: len(x) >= 5)
    unique_movie_ids = grouped_data["movieId"].unique()
    movie_id_map = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movie_ids)}
    grouped_data["movieId"] = grouped_data["movieId"].map(movie_id_map)

    user_sequences = []
    for _, group in grouped_data.groupby("userId"):
        seq = group["movieId"].values[:length]
        if len(seq) < length:
            seq = np.pad(seq, (length - len(seq), 0), mode="constant")
        user_sequences.append(seq)

    user_sequences = np.array(user_sequences)
    inputs = user_sequences[:, :-1]
    labels = user_sequences[:, -1]

    return inputs, labels


# build SASRec model
class SASRecMoE(tf.keras.Model):
    def __init__(self, num_items, embed_dim, num_heads, num_layers, num_experts, dropout_rate=0.2):
        super(SASRecMoE, self).__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate

        self.item_embedding = tf.keras.layers.Embedding(input_dim=num_items, output_dim=embed_dim)

        # Define MoE-based components
        self.expert_layers = [tf.keras.layers.Dense(embed_dim, activation="gelu") for _ in range(num_experts)]
        self.gating_network = tf.keras.layers.Dense(num_experts, activation="softmax")
        self.transformer_blocks = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(num_items, activation="softmax")

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=False, use_embedding=False):

        if use_embedding:
            x = self.item_embedding(inputs)  # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        else:
            x = inputs  # 输入已经是 [batch_size, seq_len, embed_dim]

        seq_len = tf.shape(x)[1]
        position_embeddings = self.add_position_encoding(seq_len)
        x += position_embeddings

        gating_weights = self.gating_network(x)
        gating_weights = tf.reduce_mean(gating_weights, axis=1)

        expert_outputs = [expert_layer(x) for expert_layer in self.expert_layers]
        expert_outputs = tf.stack(expert_outputs, axis=1)
        weighted_expert_outputs = tf.reduce_sum(gating_weights[..., tf.newaxis, tf.newaxis] * expert_outputs, axis=1)

        # Transformer layers
        for layer in self.transformer_blocks:
            weighted_expert_outputs = layer(weighted_expert_outputs, weighted_expert_outputs, training=training)
            weighted_expert_outputs = self.dropout(weighted_expert_outputs)

        # 输出
        last_output = weighted_expert_outputs[:, -1, :]  # [batch_size, embed_dim]
        # output = self.output_layer(last_output)  # [batch_size, num_items]
        return last_output

    def add_position_encoding(self, seq_len):
        position = tf.range(0, seq_len, dtype=tf.float32)  # shape (seq_len,)
        position = position[:, tf.newaxis]  # shape (seq_len, 1)
        div_term = tf.exp(tf.range(0, self.embed_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.embed_dim))
        pe_even = tf.sin(position * div_term)
        pe_odd = tf.cos(position * div_term)
        pe = tf.concat([pe_even, pe_odd], axis=-1)  # shape (seq_len, embed_dim)
        return pe


if __name__ == "__main__":
    data_path = "./versions/1/rating.csv"
    data = build_sequence(data_path)

    # parameters
    num_items = 22885
    embed_dim = 64
    num_heads = 1
    num_layers = 2
    num_experts = 4
    dropout_rate = 0.2

    input_data, label_data = build_sequence(data_path, length=51)
    X_train, X_val, y_train, y_val = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(64, drop_remainder=True)
    val_dataset = val_dataset.batch(64)

    model = SASRecMoE(
        num_items=num_items,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        dropout_rate=dropout_rate,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = 0.0
        train_acc.reset_states()
        for X_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(X_batch, use_embedding=True, training=True)
                loss = loss_fn(y_batch, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss += loss
            train_acc.update_state(y_batch, predictions)

        # validation
        val_loss = 0.0
        val_acc.reset_states()
        for X_batch, y_batch in val_dataset:
            predictions = model(X_batch, use_embedding=True, training=False)
            loss = loss_fn(y_batch, predictions)
            val_loss += loss
            val_acc.update_state(y_batch, predictions)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        print(f"Train Loss:{train_loss:.4f}, Train Acc: {train_acc.result().numpy():.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc.result().numpy():.4f}")

    print("Done")
