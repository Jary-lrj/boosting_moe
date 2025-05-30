import torch
from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self, csv_file, max_seq_len=50):
        """
        csv_file: csv 文件路径，包含 user_id, log_seq, item_id, deal_probability
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
        user_id = row["user_id"]
        item_id = row["item_id"]
        label = row["deal_probability"]

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
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "deal_probability": torch.tensor(label, dtype=torch.float),
        }
        return sample


def get_dataloaders(args):

    train_dataset = SeqDataset(args.train_path, max_seq_len=args.maxlen)
    valid_dataset = SeqDataset(args.valid_path, max_seq_len=args.maxlen)
    test_dataset = SeqDataset(args.test_path, max_seq_len=args.maxlen)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
