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


class Args:
    def __init__(self, df: pd.DataFrame):
        # 自动从 DataFrame 推断特征规模
        self.user_num = 1141729
        self.item_num = 846811

        # 模型结构参数
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = 10  # 用户历史序列长度

        # 数据集相关
        self.data_name = "ad"
        self.train_file = "train.csv"
        self.valid_file = "valid.csv"
        self.test_file = "test.csv"

        # MoE 相关
        self.num_experts = 4
        self.alpha = 0.1
        self.top_k = 4

        # 上下文特征 embedding 尺寸
        self.context_emb_dim = 8
        self.negative_samples = 0  # 负采样数量

        # 学习参数
        self.lr = 1e-3
        self.epochs = 10
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


class CTRDataset(Dataset):
    def __init__(self, csv_file, max_seq_len=50):
        """
        csv_file: csv 文件路径，包含 user_id, log_seq, adgroup_id, clk
        max_seq_len: 对log_seq做截断或padding的最大长度
        """
        import pandas as pd

        self.data = pd.read_csv(csv_file)
        self.data["log_seq"] = self.data["log_seq"].fillna("").astype(str)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row["user"]
        adgroup_id = row["adgroup_id"]
        label = row["clk"]

        # log_seq存储为字符串，用逗号分隔，转为list[int]
        if row["log_seq"] == "":
            seq = []
        else:
            seq = list(map(int, row["log_seq"].split(",")))

        # 截断或padding到 max_seq_len，左侧padding 0
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len :]
        else:
            seq = [0] * (self.max_seq_len - len(seq)) + seq

        sample = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "log_seq": torch.tensor(seq, dtype=torch.long),
            "adgroup_id": torch.tensor(adgroup_id, dtype=torch.long),
            "clk": torch.tensor(label, dtype=torch.float),
        }
        return sample


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

        self.layer_norm = nn.LayerNorm(hidden_units)

    def forward(self, x):

        batch_size, seq_len, _ = x.size()

        # 门控网络计算专家权重
        gate_logits = self.gate(x)  # 形状：(batch_size, seq_len, num_experts)

        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # 归一化top-k权重
        # top_k_weights 形状：(batch_size, seq_len, top_k)
        # top_k_indices 形状：(batch_size, seq_len, top_k)

        # 初始化输出张量
        output = torch.zeros_like(x)  # 形状：(batch_size, seq_len, hidden_units)

        # 计算top-k专家的输出（优化为批量操作）
        for k in range(self.top_k):
            # 获取当前专家索引和权重
            expert_idx = top_k_indices[:, :, k]  # 形状：(batch_size, seq_len)
            expert_weight = top_k_weights[:, :, k].unsqueeze(-1)  # 形状：(batch_size, seq_len, 1)

            # 创建掩码，标记每个位置的专家输出
            expert_output = torch.zeros_like(x)  # 形状：(batch_size, seq_len, hidden_units)
            for idx in range(self.num_experts):
                mask = (expert_idx == idx).unsqueeze(-1)  # 形状：(batch_size, seq_len, 1)
                if mask.any():
                    # 只计算选中专家的输出
                    expert_output += mask.float() * self.experts[idx](x)  # 广播计算

            # 加权累加到输出
            output += expert_weight * expert_output

        # 残差连接和LayerNorm
        output = self.layer_norm(x + output)

        return output


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

        self.gate = nn.Linear(hidden_units, num_experts)
        self.layer_norm = nn.LayerNorm(hidden_units)

    def forward(self, x):
        residual = x
        B, L, D = x.size()
        gate_logits = self.gate(x)  # (B, L, E)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, L, E)
        topk_vals, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (B, L, k)
        boost_input = x.detach()
        stacked_outputs = []

        for i in range(self.top_k):
            expert_idx = topk_indices[:, :, i]  # (B, L)
            expert_out = self._forward_selected_expert(boost_input, expert_idx)  # (B, L, D)
            if i < self.top_k - 1:
                boost_input = boost_input + self.alpha * expert_out.detach()
            else:
                boost_input = boost_input + self.alpha * expert_out
            stacked_outputs.append(expert_out)

        fused = sum(topk_vals[:, :, i].unsqueeze(-1) * stacked_outputs[i] for i in range(self.top_k))
        return self.layer_norm(residual + fused)

    def _forward_selected_expert(self, x, expert_indices):
        B, L, D = x.size()
        out = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            mask = expert_indices == expert_id  # (B, L)
            if mask.sum() == 0:
                continue

            masked_x = x[mask]  # (N_selected, D)
            masked_out = self.experts[expert_id](masked_x)  # (N_selected, D)
            out[mask] = masked_out

        return out


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
            # self.forward_layers.append(
            #     SparseBoostingMoE(
            #         args.hidden_units, args.num_experts, args.hidden_units, args.top_k, args.alpha, args.dropout_rate
            #     )
            # )
            # self.forward_layers.append(ClassicFeedForward(args.hidden_units, args.dropout_rate))
            self.forward_layers.append(
                SparseMoE(args.hidden_units, args.num_experts, args.hidden_units, args.top_k, args.dropout_rate)
            )

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

    def forward(self, user_ids, log_seqs, item_ids):

        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        user_emb = self.user_emb(user_ids.to(self.dev))
        combined_feat = final_feat + user_emb
        item_emb = self.item_emb(item_ids.to(self.dev))
        logits = (combined_feat * item_emb).sum(dim=-1)

        return logits


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

        for i, batch in enumerate(train_loader):
            # unpack batch
            user_id = batch["user"].to(args.device)
            log_seq = batch["log_seq"].to(args.device)
            adgroup_id = batch["adgroup_id"].to(args.device)
            label = batch["clk"].float().to(args.device)

            # forward
            logits = model(user_id, log_seq, adgroup_id)
            loss = criterion(logits.view(-1), label.view(-1))
            optimizer.zero_grad()
            loss.backward()

            if hasattr(args, "grad_clip_norm") and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()
            total_loss += loss.item()

        # 计算平均训练损失
        train_loss = total_loss / len(train_loader)

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
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            user_id = batch["user_id"].to(args.device)
            log_seq = batch["log_seq"].to(args.device)
            adgroup_id = batch["adgroup_id"].to(args.device)
            label = batch["label"].float().to(args.device)

            logits = model(user_id, log_seq, adgroup_id)
            loss = criterion(logits.view(-1), label.view(-1))
            total_loss += loss.item()

            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = label.view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    # 计算 AUC
    auc = roc_auc_score(all_labels, all_probs)

    print("Test Done.")
    return total_loss / len(data_loader), auc


if __name__ == "__main__":

    exp_name = "boosting_sasrec_1e-6_dr_0.2.log"  # 实验名称
    wandb.init(project="SASRec", name=exp_name)
    config = wandb.config

    df = pd.read_csv("processed_train.csv")
    args = Args(df)
    config = args.__dict__
    print("data loaded success")

    model = SASRec(args.user_num, args.item_num, args).to(args.device)
    print("model init success")

    data_dir = f"data/ad"
    all_adgroup_ids = pd.read_csv(f"{data_dir}/train.csv")["adgroup_id_enc"].unique().tolist()

    # 初始化一个空的字典来累积用户的历史
    current_user_histories = {}

    # 1. 构建训练集 Dataset (1-6天)
    print("Building train dataset (Days 1-6)...")
    train_dataset = CTRDataset(
        csv_path=f"{data_dir}/train.csv",
        all_adgroup_ids=all_adgroup_ids,  # 负采样空间只用训练集物品
        max_seq_len=args.maxlen,
        num_negative_samples=args.negative_samples,
        split="train",
        user_histories=current_user_histories,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Train dataset size: {len(train_dataset)}")

    # 更新历史：将训练集生成的用户历史保存下来，供后续数据集使用
    current_user_histories.update(train_dataset.user_current_histories)
    print(f"User histories after train data: {len(current_user_histories)} users.")

    # 3. 构建验证集 Dataset，传入训练集历史
    print("Building validation dataset with train histories (Day 7)...")
    valid_dataset = CTRDataset(
        csv_path=f"{data_dir}/valid.csv",
        all_adgroup_ids=all_adgroup_ids,
        max_seq_len=args.maxlen,
        num_negative_samples=args.negative_samples,
        split="valid",
        user_histories=current_user_histories,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Validation dataset size: {len(valid_dataset)}")

    # 再次更新历史：将第7天的点击也加入到历史中，以便测试集使用
    current_user_histories.update(valid_dataset.user_current_histories)
    print(f"User histories after valid data: {len(current_user_histories)} users.")

    # 4. 构建测试集 Dataset，传入训练集历史
    print("Building test dataset with train histories (Day 8)...")
    test_dataset = CTRDataset(
        csv_path=f"{data_dir}/test.csv",
        all_adgroup_ids=None,
        max_seq_len=args.maxlen,
        num_negative_samples=args.negative_samples,
        split="test",
        user_histories=current_user_histories,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # 训练模型
    wandb.watch(model, log="all")
    train_model(model, train_loader, valid_loader, test_loader, args, exp_name=exp_name)
