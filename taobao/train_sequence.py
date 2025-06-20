import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time
import wandb
import random
import tqdm
import pickle
import os
import json


class Args:
    def __init__(self):
        # 自动从 DataFrame 推断特征规模
        self.user_num = 1210271
        self.item_num = 249274

        # 模型结构参数
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = 50  # 用户历史序列长度

        # 数据集相关
        self.data_name = "beauty"
        self.train_file = "train.csv"
        self.valid_file = "valid.csv"
        self.test_file = "test.csv"

        # MoE 相关
        self.num_experts = 4
        self.alpha = 0.1
        self.top_k = 2

        # 上下文特征 embedding 尺寸
        self.context_emb_dim = 8
        self.negative_samples = 0  # 负采样数量

        # 学习参数
        self.lr = 1e-3
        self.epochs = 50
        self.batch_size = 1024

        self.patience = 5  # 早停耐心
        self.weight_decay = 1e-6  # 权重衰减
        # 设备自动检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassicFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ClassicFeedForward, self).__init__()

        self.linear1 = torch.nn.Linear(hidden_units, 4 * hidden_units)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(4 * hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.linear1(inputs)  # [B, L, H] → [B, L, H]
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)

        outputs = self.linear2(outputs)  # [B, L, H] → [B, L, H]
        outputs = self.dropout2(outputs)

        outputs += inputs

        return outputs


class TransformerGate(nn.Module):
    def __init__(self, hidden_dim=64, num_experts=4, num_heads=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(hidden_dim, num_experts)  # 输出每个 token 对各专家的打分

    def forward(self, x):  # x: (B, L, H)
        h = self.encoder(x)  # (B, L, H)
        logits = self.output_proj(h)  # (B, L, E)
        return logits


class TaobaoDataset(Dataset):
    def __init__(self, csv_path, cache_path=None, max_seq_len=50, split="train"):
        self.csv_path = csv_path
        self.cache_path = cache_path or csv_path.replace(".csv", f"_{split}_cache.pkl")
        self.max_seq_len = max_seq_len
        self.samples = []

        if os.path.exists(self.cache_path):
            print(f"[Cache] Loading samples from {self.cache_path} ...")
            with open(self.cache_path, "rb") as f:
                self.samples = pickle.load(f)
        else:
            print(f"[Build] Processing raw CSV: {csv_path}")
            reader = pd.read_csv(csv_path, chunksize=500_000)

            for chunk in reader:
                for _, row in chunk.iterrows():
                    log_seq = json.loads(row["log_seq"])
                    log_seq = log_seq[-self.max_seq_len :]
                    padded_seq = [0] * (self.max_seq_len - len(log_seq)) + log_seq

                    self.samples.append(
                        {
                            "user": int(row["user_id"]),
                            "log_seq": padded_seq,
                            "target_item": int(row["target_item"]),
                            "label": int(row["clk"]),
                        }
                    )

            with open(self.cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"[Cache Saved] {self.cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "user": torch.tensor(sample["user"], dtype=torch.long),
            "log_seq": torch.tensor(sample["log_seq"], dtype=torch.long),
            "target_item": torch.tensor(sample["target_item"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


class SparseMoE(nn.Module):
    def __init__(self, hidden_units, num_experts, expert_hidden_dim, top_k=2, dropout=0.1):

        super(SparseMoE, self).__init__()

        self.num_experts = num_experts
        self.hidden_units = hidden_units
        self.expert_hidden_dim = expert_hidden_dim
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过专家数量

        # 专家网络：每个专家是一个两层FFN
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_units, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, hidden_units),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]  # 4*2*64*256+64*4=
        )

        self.gate = nn.Linear(hidden_units, num_experts)
        self.tau = 1.0
        self.layer_norm = nn.LayerNorm(hidden_units)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Step 1: 计算门控 logits
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)

        # Step 2: 添加 Gumbel 噪声（可微 trick）
        gumbel_noise = -torch.empty_like(gate_logits).exponential_().log()
        noisy_logits = (gate_logits + gumbel_noise) / self.tau  # (B, L, E)

        # Step 3: 获取 Top-K 掩码（只保留 topk）
        topk_vals, topk_indices = torch.topk(noisy_logits, k=self.top_k, dim=-1)  # (B, L, k)
        topk_mask = torch.zeros_like(noisy_logits).scatter_(-1, topk_indices, 1.0)  # (B, L, E)

        # Step 4: 将非 topk 的位置置为 -inf，使其 softmax 后为 0
        masked_logits = noisy_logits.masked_fill(topk_mask == 0, float("-inf"))
        gate_weights = F.softmax(masked_logits, dim=-1)  # (B, L, E)

        # Step 5: 计算所有专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个 (B, L, H)

        expert_outputs = torch.stack(expert_outputs, dim=0)  # (E, B, L, H)
        expert_outputs = expert_outputs.permute(1, 2, 0, 3)  # (B, L, E, H)

        # Step 6: 加权求和专家输出
        gate_weights = gate_weights.unsqueeze(-1)  # (B, L, E, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=2)  # (B, L, H)

        # Step 7: 残差连接 + LayerNorm
        output = self.layer_norm(x + output)

        return output, []


class BoostingMoE(nn.Module):
    def __init__(self, hidden_units, num_experts, expert_hidden_dim, alpha=0.5, dropout=0.1):

        super(BoostingMoE, self).__init__()

        self.num_experts = num_experts
        self.hidden_units = hidden_units
        self.expert_hidden_dim = expert_hidden_dim
        self.alpha = alpha

        # 专家网络：每个专家是一个两层FFN
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_units, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, hidden_units),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

        # 门控网络：为每个专家生成权重
        self.gate = nn.Linear(hidden_units, num_experts)

        # LayerNorm：稳定输出
        self.layer_norm = nn.LayerNorm(hidden_units)

        # 当前训练的专家索引（用于顺序训练）
        self.current_expert_idx = 0

        # 注意力聚合各个专家的输出
        self.attn_proj_q = nn.Linear(hidden_units, hidden_units)
        self.attn_proj_k = nn.Linear(hidden_units, hidden_units)
        self.attn_proj_v = nn.Linear(hidden_units, hidden_units)

        # 新增：1D CNN 建模专家序列
        self.conv1d = nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, groups=1)

    def set_expert_idx(self, idx):
        """设置当前训练的专家索引"""
        self.current_expert_idx = idx % self.num_experts

    def forward(self, x):
        residual = x
        boost_input = x.detach()
        accumulated_output = torch.zeros_like(x)
        expert_outputs = []

        for idx in range(self.current_expert_idx + 1):
            expert_out = self.experts[idx](boost_input)
            expert_out = nn.Dropout(0.2)(expert_out)  # Dropout

            if idx < self.current_expert_idx:
                boost_input = boost_input + self.alpha * expert_out.detach()
                accumulated_output += self.alpha * expert_out.detach()
            else:
                boost_input = boost_input + self.alpha * expert_out
                accumulated_output += self.alpha * expert_out

            expert_outputs.append(expert_out if idx == self.current_expert_idx else expert_out.detach())

        stacked = torch.stack(expert_outputs, dim=2)

        # 方法1：gate-based
        # gate_logits = self.gate(x)
        # gate_weights = F.sigmoid(gate_logits[:, :, : stacked.size(2)])  # 截断成 (B, L, E_used)
        # fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=2)  # (B, L, D)

        # 方法2：attention-based
        q = self.attn_proj_q(x).unsqueeze(2)
        k = self.attn_proj_k(stacked)
        v = self.attn_proj_v(stacked)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        fused = torch.matmul(attn_weights, v).squeeze(2)

        # 方法3：Conv1d-based
        # B, L, E, D = stacked.size()
        # x_conv = stacked.view(B * L, E, D).transpose(1, 2)  # (B*L, D, E)
        # x_conv = self.conv1d(x_conv)  # (B*L, D, E)
        # fused = x_conv.transpose(1, 2).view(B, L, E, D).mean(dim=2)  # (B, L, D)

        return self.layer_norm(residual + fused)


class SparseBoostingMoE(nn.Module):
    def __init__(self, hidden_units, num_experts, expert_hidden_dim, top_k=1, alpha=0.5, dropout=0.1):
        super(SparseBoostingMoE, self).__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_units = hidden_units
        self.expert_hidden_dim = expert_hidden_dim
        self.alpha = alpha

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_units, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, hidden_units),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = TransformerGate()
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.tau = 1.0  # 温度参数，用于控制Gumbel-Softmax的平滑度
        self.register_buffer("expert_ema_scores", torch.zeros(num_experts))
        self.ema_decay = 0.99

    def forward(self, x):
        residual = x
        boost_input = x
        batch_expert_scores = torch.zeros(self.num_experts, device=x.device)
        expert_outputs = []

        for i in range(self.top_k):
            gate_logits = self.gate(boost_input)  # (B, L, E)
            gate_probs = F.softmax(gate_logits, dim=-1)  # (B, L, E)
            entropy = -(gate_probs * gate_probs.log()).sum(dim=-1, keepdim=True)  # (B, L, 1)
            # ema_scaled = self.expert_ema_scores.view(1, 1, -1)  # shape: (1,1,E)
            gate_logits = gate_logits * entropy

            # --- Gumbel noise + temperature scaling ---
            gumbel_noise = -torch.empty_like(gate_logits).exponential_().log()
            noisy_logits = (gate_logits + gumbel_noise) / self.tau  # (B, L, E)

            # --- Top-K masking ---
            topk_vals, topk_indices = torch.topk(noisy_logits, k=self.top_k, dim=-1)  # (B, L, K)
            mask = torch.zeros_like(noisy_logits).scatter_(-1, topk_indices, 1.0)
            masked_logits = noisy_logits.masked_fill(mask == 0, float("-inf"))  # (B, L, E)

            # --- Softmax over masked logits to get sparse weights ---
            gate_weights = F.softmax(masked_logits, dim=-1)  # (B, L, E)

            # --- Experts computation ---
            expert_out = torch.zeros_like(x)
            for expert_id, expert in enumerate(self.experts):
                expert_result = expert(boost_input)  # (B, L, H)
                weight = gate_weights[..., expert_id].unsqueeze(-1)  # (B, L, 1)
                contribution = weight * expert_result
                expert_out += contribution
                # expert_contribution_norm = contribution.norm(p=2, dim=(-2, -1))  # 每个样本的贡献
                # batch_expert_scores[expert_id] = expert_contribution_norm.mean()

            # with torch.no_grad():  # 防止参与梯度传播
            #     self.expert_ema_scores = (
            #         self.ema_decay * self.expert_ema_scores + (1.0 - self.ema_decay) * batch_expert_scores
            #     )

            # --- Boost residual ---
            if i < self.top_k - 1:
                boost_input = boost_input + self.alpha * expert_out.detach()
            else:
                boost_input = boost_input + self.alpha * expert_out

            expert_outputs.append(expert_out)

        # Final residual connection and normalization
        output = self.layer_norm(residual + boost_input)
        with torch.no_grad():
            avg_gate_per_expert = gate_weights.mean(dim=(0, 1))  # (E,)
            target_distribution = torch.full_like(avg_gate_per_expert, 1.0 / self.num_experts)
            epsilon = 1e-8
            kl_loss = F.kl_div((avg_gate_per_expert + epsilon).log(), target_distribution, reduction="batchmean")
        return output, expert_outputs, kl_loss


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)

        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(
                SparseBoostingMoE(
                    args.hidden_units, args.num_experts, args.hidden_units, args.top_k, args.alpha, args.dropout_rate
                )
            )
            # self.forward_layers.append(ClassicFeedForward(args.hidden_units, args.dropout_rate))
            # self.forward_layers.append(
            #     SparseMoE(args.hidden_units, args.num_experts, args.hidden_units, args.top_k, args.dropout_rate)
            # )

    def log2feats(self, log_seqs):
        all_layer_expert_outputs = []
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5

        poss = torch.arange(1, log_seqs.shape[1] + 1, device=self.dev).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        poss *= (log_seqs != 0).long().to(self.dev)
        seqs += self.pos_emb(poss)

        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = seqs.transpose(0, 1)

            seqs = self.forward_layernorms[i](seqs)
            # Regular Code
            # seqs, _ = self.forward_layers[i](seqs)
            # Boosting Code
            seqs, expert_output, kl_loss = self.forward_layers[i](seqs)
            all_layer_expert_outputs.append(expert_output)

        log_feats = self.last_layernorm(seqs)
        # return log_feats
        return log_feats, all_layer_expert_outputs, kl_loss

    def forward(self, user_ids, log_seqs, item_ids):

        # log_feats = self.log2feats(log_seqs)
        log_feats, all_layer_expert_outputs, kl_loss = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        user_emb = self.user_emb(user_ids.to(self.dev))
        combined_feat = final_feat + user_emb
        item_emb = self.item_emb(item_ids.to(self.dev))
        logits = (combined_feat * item_emb).sum(dim=-1)
        # return logits
        return logits, all_layer_expert_outputs, kl_loss


def train_model(model, train_loader, valid_loader, test_loader, args, exp_name):
    print("Training model...")
    criterion = nn.BCEWithLogitsLoss()
    # 确保优化器包含 weight_decay，如果 Args 中没有，可以在这里添加
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )  # <-- 添加 weight_decay

    # ==================== 早停相关变量 ====================
    best_valid_loss = float("inf")  # 记录迄今为止最好的验证损失
    best_epoch = 0  # 记录最佳验证损失所在的 epoch
    patience_counter = 0  # 记录验证损失没有改善的 epoch 数量
    # best_model_state = None          # 用于保存最佳模型的状态字典
    # ====================================================

    # 在训练开始前，设置一个模型保存路径
    model_save_path = f"best_model_{exp_name}.pth"

    for epoch in range(1, args.epochs + 1):

        model.train()
        start_time = time.time()
        total_loss = 0
        total_samples = 0

        for i, batch in enumerate(train_loader):
            # unpack batch
            user_id = batch["user"].to(args.device)
            log_seq = batch["log_seq"].to(args.device)
            adgroup_id = batch["target_item"].to(args.device)
            label = batch["label"].float().to(args.device)

            # forward
            logits, all_layer_experts_output, kl_loss = model(user_id, log_seq, adgroup_id)
            # logits = model(user_id, log_seq, adgroup_id)
            loss = criterion(logits.view(-1), label.view(-1))
            # target_emb = model.item_emb(adgroup_id)
            # residual_loss = 0.0
            # for layer_experts in all_layer_experts_output:
            #     residual = target_emb.detach()
            #     for expert_out in layer_experts:
            #         expert_pred = expert_out[:, -1, :]
            #         residual_loss += F.mse_loss(expert_pred, residual)
            #         residual = residual - expert_pred.detach()
            total_batch_loss = loss
            optimizer.zero_grad()
            total_batch_loss.backward()

            if hasattr(args, "grad_clip_norm") and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()
            total_loss += total_batch_loss.item() * user_id.size(0)
            total_samples += user_id.size(0)

        # 计算平均训练损失
        train_loss = total_loss / total_samples

        # evaluate on validation set
        valid_loss, valid_auc = evaluate(model, valid_loader, criterion, args)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid AUC: {valid_auc:.4f} | Elapsed Time: {elapsed:.2f}s"
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_auc": valid_auc,
                "elapsed_time": elapsed,
            }
        )

        # ==================== 早停逻辑 ====================
        # 通常我们希望验证损失越小越好，所以是寻找最小损失
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            # best_model_state = model.state_dict() # 保存当前最佳模型的状态
            torch.save(model.state_dict(), model_save_path)  # 直接保存到文件，避免内存占用过高
            patience_counter = 0  # 重置耐心计数器
            print(f"  --> Valid Loss improved. Saving model to {model_save_path}")
        else:
            patience_counter += 1
            print(f"  --> Valid Loss did not improve. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"Early stopping triggered! No improvement for {args.patience} epochs.")
            print(f"Best Valid Loss: {best_valid_loss:.4f} at Epoch {best_epoch}")
            break  # 停止训练循环
        # ====================================================

    # 最终测试：加载最佳模型进行测试
    print("Final test...")
    # 确保加载了最佳模型
    if "model_save_path" in locals() and torch.cuda.is_available():  # 检查路径变量是否存在且CUDA可用
        model.load_state_dict(torch.load(model_save_path))
    elif "model_save_path" in locals():  # 如果没有CUDA，或者模型在CPU上训练
        model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
    else:
        print("Warning: No best model saved or path not found. Testing with the last trained model.")
        # 如果没有保存最佳模型，则使用最后一个epoch的模型进行测试

    test_loss, test_auc = evaluate(model, test_loader, criterion, args)
    wandb.log({"test_loss": test_loss, "test_auc": test_auc})
    print(f"Final Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")
    print("Final test done.")


def evaluate(model, data_loader, criterion, args):
    print("Evaluating model...")
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            user_id = batch["user"].to(args.device)
            log_seq = batch["log_seq"].to(args.device)
            adgroup_id = batch["target_item"].to(args.device)
            label = batch["label"].float().to(args.device)

            logits, _, _ = model(user_id, log_seq, adgroup_id)
            # logits = model(user_id, log_seq, adgroup_id)
            loss = criterion(logits.view(-1), label.view(-1))
            total_loss += loss.item() * label.size(0)
            total_samples += label.size(0)

            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = label.view(-1).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")

    # 计算 AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        print("Warning: Only one class present in labels, AUC is not defined.")
        auc = float("nan")

    print("Evaluation Done.")
    return avg_loss, auc


if __name__ == "__main__":

    exp_name = "beauty_boostingmoe_entropy_multiply"  # 实验名称
    args = Args()
    with open(f"args_{exp_name}.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    config = args.__dict__
    wandb.init(project="beauty", name=exp_name, config=config)
    model = SASRec(args.user_num, args.item_num, args).to(args.device)
    print("model init success")

    data_dir = f"data/{args.data_name}"
    all_adgroup_ids = None

    # 构建训练集
    print("Building train dataset...")
    train_dataset = TaobaoDataset(
        csv_path=f"{data_dir}/{args.train_file}",
        max_seq_len=args.maxlen,
        split="train",
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Train dataset size: {len(train_dataset)}")

    # 构建验证集
    print("Building validation dataset...")
    valid_dataset = TaobaoDataset(
        csv_path=f"{data_dir}/{args.valid_file}",
        max_seq_len=args.maxlen,
        split="valid",
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Validation dataset size: {len(valid_dataset)}")

    # 构建测试集
    print("Building test dataset...")
    test_dataset = TaobaoDataset(
        csv_path=f"{data_dir}/{args.test_file}",
        max_seq_len=args.maxlen,
        split="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test dataset size: {len(test_dataset)}")

    # 训练模型
    wandb.watch(model, log="all")
    train_model(model, train_loader, valid_loader, test_loader, args, exp_name=exp_name)
