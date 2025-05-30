import pandas as pd
from sklearn.model_selection import train_test_split


def generate_datasets_from_csv(input_csv):
    df = pd.read_csv(input_csv)

    # 保留必要字段
    df = df[["user", "adgroup_id", "time_stamp", "clk"]]

    # user 和 adgroup_id 重编码为从1开始连续整数
    df["user"] = df["user"].astype("category").cat.codes + 1
    df["adgroup_id"] = df["adgroup_id"].astype("category").cat.codes + 1
    print(f"Unique users: {df['user'].nunique()}, Unique items: {df['adgroup_id'].nunique()}")

    # 排序
    df = df.sort_values(["user", "time_stamp"])

    train_records = []
    valid_records = []
    test_records = []

    for user, group in df.groupby("user"):
        seq = group["adgroup_id"].tolist()
        probs = group["clk"].tolist()
        n = len(seq)
        if n < 3:
            # 少于3条全部训练
            for i in range(n):
                train_records.append({"user": user, "log_seq": seq[:i], "adgroup_id": seq[i], "clk": probs[i]})
            continue

        # 训练集
        for i in range(1, n - 2):
            train_records.append({"user": user, "log_seq": seq[:i], "adgroup_id": seq[i], "clk": probs[i]})
        # 验证集
        valid_records.append({"user": user, "log_seq": seq[:-2], "adgroup_id": seq[-2], "clk": probs[-2]})
        # 测试集
        test_records.append({"user": user, "log_seq": seq[:-1], "adgroup_id": seq[-1], "clk": probs[-1]})

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
    generate_datasets_from_csv("raw_sample.csv")
