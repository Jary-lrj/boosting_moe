import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wandb
import numpy as np
import torch.nn.functional as F


def train_loop(model, train_loader, valid_loader, test_loader, args):

    wandb.init(project=args.data, name=args.name, reinit=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)

    model.to(args.device)
    best_valid_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80)
        for batch in pbar:
            user_ids = batch["user_id"].to(args.device)
            log_seqs = batch["log_seq"].to(args.device)
            item_ids = batch["item_id"].to(args.device)
            labels = batch["deal_probability"].to(args.device)

            optimizer.zero_grad()
            probs, all_expert_outputs = model(user_ids, log_seqs, item_ids)

            # 主任务损失（MSE）
            mse_loss = F.mse_loss(probs, labels.float())

            # 获取 label 对应 item embedding
            target_emb = model.item_emb(item_ids)
            residual_loss = 0.0
            for layer_experts in all_expert_outputs:
                residual = target_emb.detach()
                for expert_out in layer_experts:
                    expert_pred = expert_out[:, -1, :]  # 只取最后一个位置
                    residual_loss += F.mse_loss(expert_pred, residual)
                    residual = residual - expert_pred.detach()

            total_batch_loss = mse_loss + args.alpha * residual_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item() * user_ids.size(0)
            pbar.set_postfix(loss=total_batch_loss.item())

        avg_train_loss = total_loss / len(train_loader.dataset)

        avg_valid_loss, valid_rmse = evaluate_rmse(model, valid_loader, F.mse_loss, args)

        print(
            f"Epoch {epoch} finished. Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid RMSE: {valid_rmse:.4f}"
        )

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "valid_rmse": valid_rmse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step(avg_valid_loss)

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

    model.load_state_dict(torch.load(f"best_{args.name}_model.pth"))
    test_loss, test_rmse = evaluate_rmse(model, test_loader, F.mse_loss, args)
    print(f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}")
    wandb.log({"test_loss": test_loss, "test_rmse": test_rmse})


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

            probs, _ = model(user_ids, log_seqs, item_ids)  # unpack logits only
            loss = criterion(probs, labels.float())
            total_loss += loss.item() * labels.size(0)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return avg_loss, rmse
