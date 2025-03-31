import tensorflow as tf
from SASRec import SASRec
from MLP import MLP


class BoostingMoE(tf.keras.Model):

    def __init__(self, num_items, embed_dim, num_heads, num_layers, num_experts, dropout_rate=0.2, lambda_reg=0.01):
        super(BoostingMoE, self).__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.lambda_reg = lambda_reg
        self.num_rounds = 2  # boosting 轮数

        # 创建多个SASRec专家
        self.experts = [
            SASRec(
                num_items=num_items,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_experts)
        ]

        # 门控网络 (根据输入动态选择专家)
        self.gate = tf.keras.layers.Dense(self.num_experts, activation="sigmoid")

    @tf.function
    def call(self, inputs, training=False):
        # inputs: shape = (batch_size, seq_len)，是 item_id 序列
        if isinstance(inputs, dict):
            if "inputs" in inputs:
                inputs = inputs["inputs"]
            else:
                raise ValueError("Expected 'inputs' key in input dict")

        assert type(inputs) == tf.Tensor, "inputs must be a Tensor"
        expert_outputs = []

        for expert in self.experts:
            # expert 负责嵌入 + 位置编码 + transformer 等
            # 输出 shape: (batch, num_items)
            expert_output = expert(inputs, training=training)
            expert_outputs.append(expert_output)

        return expert_outputs  # List of (batch_size, num_items)

    @tf.function
    def train_step(self, data):
        x, y = data  # x 是 dict, y 是 pos_labels
        inputs = x["inputs"]
        neg_labels = x["neg_labels"]  # shape: (batch_size, num_neg)
        pos_labels = y  # shape: (batch_size,)
        batch_size = tf.shape(inputs)[0]

        current_prediction = tf.zeros((batch_size, self.num_items), dtype=tf.float32)
        selected_outputs = []
        expert1_output = None

        with tf.GradientTape() as tape:
            for round_idx in range(self.num_rounds):
                expert_outputs = self(inputs, training=True)
                residual_losses, best_expert_idx = self.compute_residual_losses(
                    pos_labels, neg_labels, current_prediction, expert_outputs
                )

                best_expert_output = tf.gather(expert_outputs, best_expert_idx)
                current_prediction += best_expert_output
                selected_outputs.append(best_expert_output)

                if round_idx == 0:
                    expert1_output = tf.identity(current_prediction)

            # =========================
            # 构建正负样本的 logits 子集
            # =========================

            # shape: (batch_size,)
            pos_scores = tf.gather(current_prediction, pos_labels, axis=1, batch_dims=1)

            # shape: (batch_size, num_neg)
            neg_scores = tf.gather(current_prediction, neg_labels, axis=1, batch_dims=1)

            # 拼接：正样本放在第 0 列
            # shape: (batch_size, 1 + num_neg)
            combined_logits = tf.concat([tf.expand_dims(pos_scores, 1), neg_scores], axis=1)

            # 标签都是 0（因为正样本在第一列）
            labels = tf.zeros((batch_size,), dtype=tf.int32)

            # 使用 sparse softmax cross-entropy
            softmax_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=combined_logits)
            )

            # 正则项
            reg_loss = tf.add_n([tf.reduce_sum(tf.square(output)) for output in selected_outputs])
            total_loss = softmax_loss + self.lambda_reg * reg_loss

        # 反向传播
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 更新指标
        self.compiled_metrics.update_state(pos_labels, tf.argmax(current_prediction, axis=1))

        # Explained Variance
        ev1 = self.explained_variance(pos_labels, expert1_output)
        ev_total = self.explained_variance(pos_labels, current_prediction)
        delta_ev = ev_total - ev1

        metrics = {"loss": total_loss}
        metrics.update({m.name: m.result() for m in self.metrics})
        metrics["explained_variance"] = ev_total
        metrics["delta_ev"] = delta_ev
        return metrics

    def compute_residual_losses(self, pos_labels, neg_labels, y_pred, expert_outputs):
        """
        pos_labels: shape (batch_size,)         # 正样本 item id
        neg_labels: shape (batch_size, num_neg) # 负样本 item id
        y_pred: 当前预测 logits（累加过的），shape: (batch_size, num_items)
        expert_outputs: List of (batch_size, num_items) logits
        """
        residual_losses = []

        for expert_output in expert_outputs:
            # BCE Loss
            pos_scores = tf.gather(expert_output, pos_labels, axis=1, batch_dims=1)  # shape: (batch_size,)
            neg_scores = tf.gather(expert_output, neg_labels, axis=1, batch_dims=1)  # shape: (batch_size, num_neg)

            pos_labels_bce = tf.ones_like(pos_scores)
            neg_labels_bce = tf.zeros_like(neg_scores)

            pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_labels_bce, logits=pos_scores)
            neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=neg_labels_bce, logits=neg_scores)

            loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
            residual_losses.append(loss)

            residual_losses = tf.stack(residual_losses)  # shape: (num_experts,)
            max_index = tf.argmax(residual_losses)
            return residual_losses, max_index

    # 为了量化我们方法的贡献，我们还可以定义一个自定义指标
    def explained_variance(self, y_true, y_pred):
        y_true = tf.cast(tf.one_hot(y_true, depth=tf.shape(y_pred)[-1]), dtype=tf.float32)
        residual = y_true - tf.nn.softmax(y_pred)
        var_residual = tf.reduce_mean(tf.square(residual))
        var_true = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)))
        return 1.0 - var_residual / (var_true + 1e-8)
