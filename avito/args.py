import torch


class Args:
    def __init__(self):
        self.user_num = 771769
        self.item_num = 1503424
        self.name = "avito_boostingmoe"

        # 模型结构参数
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = 10  # 用户历史序列长度

        # 数据集相关
        self.data = "avito"
        self.train_path = "train.csv"
        self.valid_path = "valid.csv"
        self.test_path = "test.csv"

        # MoE 相关
        self.num_experts = 4
        self.alpha = 1
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
