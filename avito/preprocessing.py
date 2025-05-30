import pandas as pd


def generate_datasets_from_csv(input_csv):
    df = pd.read_csv(input_csv)

    # 保留必要字段
    df = df[["user_id", "item_id", "item_seq_number", "deal_probability"]]

    # user_id 和 item_id 重编码为从1开始连续整数
    df["user_id"] = df["user_id"].astype("category").cat.codes + 1
    df["item_id"] = df["item_id"].astype("category").cat.codes + 1
    print(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}")

    # 排序
    df = df.sort_values(["user_id", "item_seq_number"])

    train_records = []
    valid_records = []
    test_records = []

    for user_id, group in df.groupby("user_id"):
        seq = group["item_id"].tolist()
        probs = group["deal_probability"].tolist()
        n = len(seq)
        if n < 3:
            # 少于3条全部训练
            for i in range(n):
                train_records.append(
                    {"user_id": user_id, "log_seq": seq[:i], "item_id": seq[i], "deal_probability": probs[i]}
                )
            continue

        # 训练集
        for i in range(1, n - 2):
            train_records.append(
                {"user_id": user_id, "log_seq": seq[:i], "item_id": seq[i], "deal_probability": probs[i]}
            )
        # 验证集
        valid_records.append(
            {"user_id": user_id, "log_seq": seq[:-2], "item_id": seq[-2], "deal_probability": probs[-2]}
        )
        # 测试集
        test_records.append(
            {"user_id": user_id, "log_seq": seq[:-1], "item_id": seq[-1], "deal_probability": probs[-1]}
        )

    train_df = pd.DataFrame(train_records)
    valid_df = pd.DataFrame(valid_records)
    test_df = pd.DataFrame(test_records)

    train_df["log_seq"] = train_df["log_seq"].apply(lambda x: ",".join(map(str, x)) if x else "")
    valid_df["log_seq"] = valid_df["log_seq"].apply(lambda x: ",".join(map(str, x)) if x else "")
    test_df["log_seq"] = test_df["log_seq"].apply(lambda x: ",".join(map(str, x)) if x else "")

    train_df.to_csv("train.csv", index=False)
    valid_df.to_csv("valid.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("Completed! train.csv, valid.csv, test.csv saved.")


if __name__ == "__main__":
    generate_datasets_from_csv("data/train.csv")
