import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wandb
import numpy as np


def train_loop(model, train_loader, valid_loader, test_loader, args):
    wandb.init(project=args.data, name=args.name, reinit=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)

    model.to(args.device)
    best_valid_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80)
        for batch in pbar:
            user_ids = batch["user_id"].to(args.device)
            log_seqs = batch["log_seq"].to(args.device)
            item_ids = batch["item_id"].to(args.device)
            labels = batch["deal_probability"].to(args.device)

            optimizer.zero_grad()
            outputs = model(user_ids, log_seqs, item_ids).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累计损失乘以batch_size，是因为loss是平均每条样本的损失，
            # 乘以batch_size可以得到该batch的总损失，最后除以总样本数求平均。
            total_loss += loss.item() * user_ids.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader.dataset)

        # 验证阶段
        avg_valid_loss, valid_auc = evaluate_rmse(model, valid_loader, criterion, args)

        print(
            f"Epoch {epoch} finished. Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid AUC: {valid_auc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "valid_auc": valid_auc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step(avg_valid_loss)

        # 早停判断
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{args.name}_model.pth")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print("Training complete.")

    # 加载最佳模型，测试集评估
    model.load_state_dict(torch.load(f"best_{args.name}_model.pth"))
    test_loss, test_auc = evaluate_rmse(model, test_loader, criterion, args)
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")

    wandb.log({"test_loss": test_loss, "test_auc": test_auc})


def evaluate_rmse(model, dataloader, criterion, args):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch["user_id"].to(args.device)
            log_seqs = batch["log_seq"].to(args.device)
            item_ids = batch["item_id"].to(args.device)
            labels = batch["deal_probability"].to(args.device)

            outputs = model(user_ids, log_seqs, item_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return avg_loss, rmse
