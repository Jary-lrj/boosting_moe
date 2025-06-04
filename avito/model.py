import torch
from torch.nn import (
    Module,
    Linear,
    Dropout,
    ReLU,
    LayerNorm,
    ModuleList,
    Conv1d,
    Sequential,
    Embedding,
    MultiheadAttention,
)
from torch.utils.data import Dataset
import pandas as pd
import random
import torch.nn.functional as F
from datetime import datetime


class ClassicFeedForward(Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ClassicFeedForward, self).__init__()

        self.linear1 = Linear(hidden_units, 1024)
        self.dropout1 = Dropout(dropout_rate)
        self.relu = ReLU()
        self.linear2 = Linear(1024, hidden_units)
        self.dropout2 = Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.linear1(inputs)  # [B, L, H] → [B, L, H]
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)

        outputs = self.linear2(outputs)  # [B, L, H] → [B, L, H]
        outputs = self.dropout2(outputs)

        outputs += inputs

        return outputs


class SparseMoE(Module):
    def __init__(self, hidden_units, num_experts, expert_hidden_dim, top_k=2, dropout=0.1):

        super(SparseMoE, self).__init__()

        self.num_experts = num_experts
        self.hidden_units = hidden_units
        self.expert_hidden_dim = expert_hidden_dim
        self.top_k = min(top_k, num_experts)  # 确保top_k不超过专家数量

        # 专家网络：每个专家是一个两层FFN
        self.experts = ModuleList(
            [
                Sequential(
                    Linear(hidden_units, expert_hidden_dim),
                    ReLU(),
                    Dropout(dropout),
                    Linear(expert_hidden_dim, hidden_units),
                    Dropout(dropout),
                )
                for _ in range(num_experts)
            ]  # 4*2*64*256+64*4=
        )

        self.gate = Linear(hidden_units, num_experts)

        self.layer_norm = LayerNorm(hidden_units)

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


class BoostingMoE(Module):
    def __init__(self, hidden_units, num_experts, expert_hidden_dim, alpha=0.5, dropout=0.1):

        super(BoostingMoE, self).__init__()

        self.num_experts = num_experts
        self.hidden_units = hidden_units
        self.expert_hidden_dim = expert_hidden_dim
        self.alpha = alpha

        # 专家网络：每个专家是一个两层FFN
        self.experts = ModuleList(
            [
                Sequential(
                    Linear(hidden_units, expert_hidden_dim),
                    ReLU(),
                    Dropout(dropout),
                    Linear(expert_hidden_dim, hidden_units),
                    Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

        # 门控网络：为每个专家生成权重
        self.gate = Linear(hidden_units, num_experts)

        # LayerNorm：稳定输出
        self.layer_norm = LayerNorm(hidden_units)

        # 当前训练的专家索引（用于顺序训练）
        self.current_expert_idx = 0

        # 注意力聚合各个专家的输出
        self.attn_proj_q = Linear(hidden_units, hidden_units)
        self.attn_proj_k = Linear(hidden_units, hidden_units)
        self.attn_proj_v = Linear(hidden_units, hidden_units)

        # 新增：1D CNN 建模专家序列
        self.conv1d = Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, groups=1)

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
            expert_out = Dropout(0.2)(expert_out)  # Dropout

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


class SparseBoostingMoE(Module):
    def __init__(self, hidden_units, num_experts, expert_hidden_dim, top_k=1, alpha=0.5, dropout=0.1):
        super(SparseBoostingMoE, self).__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_units = hidden_units
        self.expert_hidden_dim = expert_hidden_dim
        self.alpha = alpha

        self.experts = ModuleList(
            [
                Sequential(
                    Linear(hidden_units, expert_hidden_dim),
                    ReLU(),
                    Dropout(dropout),
                    Linear(expert_hidden_dim, hidden_units),
                    Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = Linear(hidden_units, num_experts)
        self.layer_norm = LayerNorm(hidden_units)
        self.tau = 1.0  # 温度参数，用于控制Gumbel-Softmax的平滑度

    def forward(self, x):
        residual = x
        boost_input = x
        expert_outputs = []

        for i in range(self.top_k):
            gate_logits = self.gate(boost_input)

            gumbel_noise = -torch.empty_like(gate_logits).exponential_().log()
            y = (gate_logits + gumbel_noise) / self.tau
            gate_weights = F.softmax(y, dim=-1)  # (B, L, E)

            expert_out = torch.zeros_like(x)
            for expert_id, expert in enumerate(self.experts):
                expert_result = expert(boost_input)
                expert_out += gate_weights[..., expert_id].unsqueeze(-1) * expert_result

            # Boost residual
            if i < self.top_k - 1:
                boost_input = boost_input + self.alpha * expert_out.detach()
            else:
                boost_input = boost_input + self.alpha * expert_out

            expert_outputs.append(expert_out)
            return self.layer_norm(residual + boost_input), expert_outputs


class SASRec(Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.user_emb = Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)

        self.emb_dropout = Dropout(p=args.dropout_rate)

        self.attention_layernorms = ModuleList()
        self.attention_layers = ModuleList()
        self.forward_layernorms = ModuleList()
        self.forward_layers = ModuleList()

        self.last_layernorm = LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(LayerNorm(args.hidden_units, eps=1e-8))
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
            seqs, expert_output = self.forward_layers[i](seqs)
            all_layer_expert_outputs.append(expert_output)

        log_feats = self.last_layernorm(seqs)
        return log_feats, all_layer_expert_outputs

    def log2feats_noboosting(self, log_seqs):
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
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats, all_layer_expert_outputs

    def forward(self, user_ids, log_seqs, item_ids):

        log_feats, all_layer_expert_outputs = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        user_emb = self.user_emb(user_ids.to(self.dev))
        combined_feat = final_feat + user_emb
        item_emb = self.item_emb(item_ids.to(self.dev))
        logits = (combined_feat * item_emb).sum(dim=-1)
        probs = torch.sigmoid(logits)

        return probs, all_layer_expert_outputs
