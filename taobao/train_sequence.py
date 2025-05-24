import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time
import wandb


class Args:
    def __init__(self, df: pd.DataFrame):
        # 自动从 DataFrame 推断特征规模
        self.user_num = df["user_id_enc"].max()
        self.item_num = df["adgroup_id_enc"].max()
        self.pid_size = df["pid_enc"].max()  # 若 pid 从 0 开始，+1 保证包含

        # 模型结构参数
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.1
        self.maxlen = 10  # 用户历史序列长度

        # MoE 相关
        self.num_experts = 4
        self.alpha = 0.1

        # 上下文特征 embedding 尺寸
        self.context_emb_dim = 8

        # 学习参数
        self.lr = 1e-3
        self.epochs = 3
        self.batch_size = 1024

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


class CTRDataset(Dataset):
    def __init__(self, csv_path, max_seq_len=50):
        self.max_seq_len = max_seq_len

        # 读取数据
        df = pd.read_csv(csv_path)

        # 构建用户历史点击序列
        self.user_hist = defaultdict(list)
        self.samples = []

        for row in df.itertuples(index=False):
            user_id = row.user_id_enc
            adgroup_id = row.adgroup_id_enc
            label = row.label

            pid = row.pid_enc
            hour = row.hour
            dow = row.dayofweek
            hour_block = row.hour_block
            is_weekend = row.is_weekend

            # 当前历史序列（复制）
            hist_seq = self.user_hist[user_id][-self.max_seq_len :]

            self.samples.append(
                {
                    "user_id": user_id,
                    "log_seq": hist_seq.copy(),
                    "item_id": adgroup_id,
                    "label": label,
                    "pid": pid,
                    "hour": hour,
                    "dayofweek": dow,
                    "hour_block": hour_block,
                    "is_weekend": is_weekend,
                }
            )

            # 添加当前行为到用户历史
            if label == 1:  # 仅将正样本加入历史（模拟点击序列）
                self.user_hist[user_id].append(adgroup_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        log_seq = data["log_seq"]
        pad_len = self.max_seq_len - len(log_seq)
        if pad_len > 0:
            # 左侧 padding（用 0 补齐）
            log_seq = [0] * pad_len + log_seq
        else:
            # 截断（只保留 max_seq_len）
            log_seq = log_seq[-self.max_seq_len :]

        return {
            "user_id": torch.tensor(data["user_id"], dtype=torch.long),
            "log_seq": torch.tensor(log_seq, dtype=torch.long),
            "item_id": torch.tensor(data["item_id"], dtype=torch.long),
            "label": torch.tensor(data["label"], dtype=torch.float),
            "pid": torch.tensor(data["pid"], dtype=torch.long),
            "hour": torch.tensor(data["hour"], dtype=torch.long),
            "dayofweek": torch.tensor(data["dayofweek"], dtype=torch.long),
            "hour_block": torch.tensor(data["hour_block"], dtype=torch.long),
            "is_weekend": torch.tensor(data["is_weekend"], dtype=torch.long),
        }


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


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)

        # 上下文特征
        self.user_emb = torch.nn.Embedding(user_num + 1, args.context_emb_dim, padding_idx=0)
        self.pid_emb = torch.nn.Embedding(args.pid_size + 1, args.context_emb_dim, padding_idx=0)
        self.hour_emb = torch.nn.Embedding(24, args.context_emb_dim)
        self.dow_emb = torch.nn.Embedding(7, args.context_emb_dim)
        self.hour_block_emb = torch.nn.Embedding(4, args.context_emb_dim)
        self.is_weekend_emb = torch.nn.Embedding(2, args.context_emb_dim)

        # 融合线性层：把 context + final_feat -> 投影到 hidden_units
        concat_dim = args.hidden_units + 6 * args.context_emb_dim
        self.fusion_layer = torch.nn.Linear(concat_dim, args.hidden_units)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(
                BoostingMoE(args.hidden_units, args.num_experts, args.hidden_units, args.alpha, args.dropout_rate)
            )
            # self.forward_layers.append(ClassicFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, log_seqs):
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
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, item_ids, pid, hour, dow, hour_block, is_weekend):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # (B, H)

        # 获取上下文特征 embedding
        context_embs = torch.cat(
            [
                self.user_emb(user_ids.to(self.dev)).squeeze(1),
                self.pid_emb(pid.to(self.dev)).squeeze(1),
                self.hour_emb(hour.to(self.dev)).squeeze(1),
                self.dow_emb(dow.to(self.dev)).squeeze(1),
                self.hour_block_emb(hour_block.to(self.dev)).squeeze(1),
                self.is_weekend_emb(is_weekend.to(self.dev)).squeeze(1),
            ],
            dim=-1,
        )

        fused_feat = self.fusion_layer(torch.cat([final_feat, context_embs], dim=-1))  # (B, H)

        item_emb = self.item_emb(item_ids.to(self.dev))
        logits = (fused_feat * item_emb).sum(dim=-1)  # 点积

        return logits  # 直接传入 BCEWithLogitsLoss


def train_model(model, train_loader, valid_loader, test_loader, args):
    print("Training model...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        # 设置当前训练的专家索引
        for expert_idx in range(args.num_experts):

            for fwd_layer in model.forward_layers:
                fwd_layer.set_expert_idx(expert_idx)
            print(f"Epoch {epoch} Training expert {expert_idx + 1}/{args.num_experts}")

            model.train()
            start_time = time.time()
            total_loss = 0

            for i, batch in enumerate(train_loader):
                # unpack batch
                user_id = batch["user_id"].to(args.device)
                log_seq = batch["log_seq"].to(args.device)
                item_id = batch["item_id"].to(args.device)
                label = batch["label"].float().to(args.device)
                pid = batch["pid"].to(args.device)
                hour = batch["hour"].to(args.device)
                dow = batch["dayofweek"].to(args.device)
                hour_block = batch["hour_block"].to(args.device)
                is_weekend = batch["is_weekend"].to(args.device)

                # forward
                logits = model(user_id, log_seq, item_id, pid, hour, dow, hour_block, is_weekend)
                loss = criterion(logits.view(-1), label.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(
                    f"Epoch {epoch} | Expert {expert_idx + 1} | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )
            # 计算平均损失

            train_loss = total_loss / len(train_loader)
            # evaluate
            valid_loss, valid_auc, valid_ctr, valid_true_ctr = evaluate(model, valid_loader, criterion, args)
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} | Expert {expert_idx + 1} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid AUC: {valid_auc:.4f} | "
                f"Valid CTR: {valid_ctr:.4f} | Valid True CTR: {valid_true_ctr:.4f} | Elapsed Time: {elapsed:.2f}s"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_auc": valid_auc,
                    "valid_ctr": valid_ctr,
                    "valid_true_ctr": valid_true_ctr,
                    "elapsed_time": elapsed,
                }
            )
        # 最终测试结果 + 绘图
        print("Final test...")
        test_loss, test_auc, test_ctr, test_true_ctr = evaluate(model, test_loader, criterion, args)
        wandb.log({"test_loss": test_loss, "test_auc": test_auc, "test_ctr": test_ctr, "test_true_ctr": test_true_ctr})
        print("Final test done.")


def evaluate(model, data_loader, criterion, args):
    print("Evaluating model...")
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            user_id = batch["user_id"].to(args.device)
            log_seq = batch["log_seq"].to(args.device)
            item_id = batch["item_id"].to(args.device)
            label = batch["label"].float().to(args.device)
            pid = batch["pid"].to(args.device)
            hour = batch["hour"].to(args.device)
            dow = batch["dayofweek"].to(args.device)
            hour_block = batch["hour_block"].to(args.device)
            is_weekend = batch["is_weekend"].to(args.device)

            logits = model(user_id, log_seq, item_id, pid, hour, dow, hour_block, is_weekend)
            loss = criterion(logits.view(-1), label.view(-1))
            total_loss += loss.item()

            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = label.view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    # 计算 AUC
    auc = roc_auc_score(all_labels, all_probs)

    # CTR 计算
    predicted_ctr = sum(all_probs) / len(all_probs)
    actual_ctr = sum(all_labels) / len(all_labels)

    print("Test Done.")
    return total_loss / len(data_loader), auc, predicted_ctr, actual_ctr


if __name__ == "__main__":

    # 初始化 wandb
    wandb.init(project="SASRec", name="boostingmoe-v2.2-debug:cpu")
    config = wandb.config

    # 读取数据
    df = pd.read_csv("processed_train.csv")
    args = Args(df)
    config = args.__dict__
    print("data loaded success")

    # 初始化模型
    model = SASRec(args.user_num, args.item_num, args).to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("model init success")

    # CTRDataset 接收两个参数：csv路径和最大序列长度
    train_dataset = CTRDataset("train_final.csv", max_seq_len=args.maxlen)
    val_dataset = CTRDataset("valid.csv", max_seq_len=args.maxlen)
    test_dataset = CTRDataset("test.csv", max_seq_len=args.maxlen)
    print("Datasets initialized with time-split files")

    # 构建 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Dataloaders initialized successfully")

    # 训练模型
    wandb.watch(model, log="all")
    train_model(model, train_loader, val_loader, test_loader, args)
