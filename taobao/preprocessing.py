import pandas as pd
from tqdm import tqdm
import gc


def generate_datasets_from_csv(input_csv):
    df = pd.read_csv(input_csv)

    # 保留必要字段
    df = df[["user", "adgroup_id", "time_stamp", "clk"]]

    # 编码
    df["user"] = df["user"].astype("category").cat.codes + 1
    df["adgroup_id"] = df["adgroup_id"].astype("category").cat.codes + 1
    print(f"Unique users: {df['user'].nunique()}, Unique items: {df['adgroup_id'].nunique()}")

    # 排序
    df = df.sort_values(["user", "time_stamp"]).reset_index(drop=True)

    train_records = []
    valid_records = []
    test_records = []

    # 分组后处理
    user_groups = df.groupby("user")
    for user, group in tqdm(user_groups, desc="Processing users"):
        ad_ids = group["adgroup_id"].tolist()  # ⬅ 关键：使用 .tolist() 确保是 list
        clks = group["clk"].tolist()
        n = len(ad_ids)

        if n < 3:
            continue

        # 滑窗生成训练样本
        for i in range(1, n - 2):
            train_records.append([user, ad_ids[:i].copy(), ad_ids[i], clks[i]])

        # 验证样本
        valid_records.append([user, ad_ids[: n - 2].copy(), ad_ids[n - 2], clks[n - 2]])

        # 测试样本
        test_records.append([user, ad_ids[: n - 1].copy(), ad_ids[n - 1], clks[n - 1]])

    def process_and_save(records, filename):
        if not records:
            print(f"No data for {filename}, skipping.")
            return
        df_out = pd.DataFrame(records, columns=["user", "log_seq", "adgroup_id", "clk"])
        df_out["log_seq"] = df_out["log_seq"].apply(lambda x: ",".join(map(str, x)) if x else "")
        df_out.to_csv(filename, index=False)
        print(f"Saved {filename}, samples: {len(df_out)}")

    # 保存文件
    process_and_save(train_records, "train_ctr.csv")
    process_and_save(valid_records, "valid_ctr.csv")
    process_and_save(test_records, "test_ctr.csv")

    del df, user_groups, train_records, valid_records, test_records
    gc.collect()

    print("Completed.")


if __name__ == "__main__":
    generate_datasets_from_csv("raw_sample.csv")
