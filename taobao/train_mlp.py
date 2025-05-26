import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import wandb


# ---------- Dataset ----------
class CTRDataset(Dataset):
    def __init__(self, df):
        self.user_id = torch.LongTensor(df["user_id_enc"].values)
        self.adgroup_id = torch.LongTensor(df["adgroup_id_enc"].values)
        self.pid = torch.LongTensor(df["pid_enc"].values)
        self.hour = torch.FloatTensor(df["hour"].values)
        self.dayofweek = torch.FloatTensor(df["dayofweek"].values)
        self.hour_block = torch.FloatTensor(df["hour_block"].values)
        self.is_weekend = torch.FloatTensor(df["is_weekend"].values)
        self.label = torch.FloatTensor(df["label"].values)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.user_id[idx],
            self.adgroup_id[idx],
            self.pid[idx],
            self.hour[idx],
            self.dayofweek[idx],
            self.hour_block[idx],
            self.is_weekend[idx],
            self.label[idx],
        )


# ---------- Model ----------
class CTRMLPModel(nn.Module):
    def __init__(self, num_users, num_ads, num_pids, embedding_dim=16, hidden_units=128):
        super(CTRMLPModel, self).__init__()
        self.user_emb = nn.Embedding(num_users + 1, embedding_dim)
        self.ad_emb = nn.Embedding(num_ads + 1, embedding_dim)
        self.pid_emb = nn.Embedding(num_pids + 1, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 4, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_id, ad_id, pid_id, hour, dayofweek, hour_block, is_weekend):
        user_vec = self.user_emb(user_id)
        ad_vec = self.ad_emb(ad_id)
        pid_vec = self.pid_emb(pid_id)

        x = torch.cat(
            [
                user_vec,
                ad_vec,
                pid_vec,
                hour.unsqueeze(1),
                dayofweek.unsqueeze(1),
                hour_block.unsqueeze(1),
                is_weekend.unsqueeze(1),
            ],
            dim=1,
        )

        return self.mlp(x).squeeze(1)


# ---------- Training ----------
def train():
    print("Loading data...")
    df = pd.read_csv("processed_train.csv")

    # Embedding范围
    num_users = df["user_id_enc"].max()
    num_ads = df["adgroup_id_enc"].max()
    num_pids = df["pid_enc"].max()

    train_df = pd.read_csv("train_final.csv")
    val_df = pd.read_csv("valid.csv")
    test_df = pd.read_csv("test.csv")

    # Loaders
    train_loader = DataLoader(CTRDataset(train_df), batch_size=1024, shuffle=True)
    val_loader = DataLoader(CTRDataset(val_df), batch_size=1024, shuffle=False)
    test_loader = DataLoader(CTRDataset(test_df), batch_size=1024, shuffle=False)

    # Model setup
    print("Initializing model...")
    model = CTRMLPModel(num_users, num_ads, num_pids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print("Starting training...")
    # Training loop
    for epoch in range(10):
        print(f"Training Epoch {epoch + 1}/{10}")
        model.train()
        start_time = time.time()
        total_loss = 0.0

        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            y_pred = model(*batch[:-1])
            y_true = batch[-1]

            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time  # End timing
        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Time: {epoch_time:.2f}s")
        wandb.log({"train_loss": total_loss, "epoch_time": epoch_time})

        # Validation AUC
        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = [x.to(device) for x in batch]
                y_pred = model(*batch[:-1])
                y_preds.extend(y_pred.cpu().numpy())
                y_trues.extend(batch[-1].cpu().numpy())

        val_predicted_ctr = np.mean(y_preds)
        val_real_ctr = np.mean(y_trues)
        print(f"Val Predicted CTR: {val_predicted_ctr:.4f} | Val Real CTR: {val_real_ctr:.4f}")
        wandb.log({"val_predicted_ctr": val_predicted_ctr, "val_real_ctr": val_real_ctr})
        val_auc = roc_auc_score(y_trues, y_preds)
        print(f"Validation AUC: {val_auc:.4f}")
        wandb.log({"val_auc": val_auc})

    # Test Set Evaluation
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = [x.to(device) for x in batch]
            y_pred = model(*batch[:-1])
            y_preds.extend(y_pred.cpu().numpy())
            y_trues.extend(batch[-1].cpu().numpy())

    test_predicted_ctr = np.mean(y_preds)
    test_real_ctr = np.mean(y_trues)
    print(f"Predicted CTR: {test_predicted_ctr:.4f} | Real CTR: {test_real_ctr:.4f}")
    wandb.log({"test_predicted_ctr": test_predicted_ctr, "test_real_ctr": test_real_ctr})
    test_auc = roc_auc_score(y_trues, y_preds)
    print(f"Final Test AUC: {test_auc:.4f}")
    wandb.log({"test_auc": test_auc})


if __name__ == "__main__":
    wandb.init(project="SASRec", name="MLP")
    train()
