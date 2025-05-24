import pandas as pd
from sklearn.model_selection import train_test_split

# ========== Step 1：读取原始数据并处理 ==========
df = pd.read_csv("raw_sample.csv")  # 包含字段: user, time_stamp, adgroup_id, pid, noclk, clk


# 编码函数
def encode_column(col):
    encoded, _ = pd.factorize(col)
    return encoded + 1


# 编码
df["user_id_enc"] = encode_column(df["user"])
df["adgroup_id_enc"] = encode_column(df["adgroup_id"])
df["pid_enc"] = encode_column(df["pid"])

# 时间特征
df["datetime"] = pd.to_datetime(df["time_stamp"], unit="s")
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["hour_block"] = df["hour"] // 6
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# 标签
df["label"] = df["clk"]

# 最终使用的字段
feature_cols = ["user_id_enc", "adgroup_id_enc", "pid_enc", "hour", "dayofweek", "hour_block", "is_weekend", "label"]

# ========== Step 2：按时间划分 train 和 valid ==========
train_start = pd.to_datetime("20170506").date()
train_end = pd.to_datetime("20170512").date()
val_day = pd.to_datetime("20170513").date()

train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)][feature_cols]
val_df = df[df["date"] == val_day][feature_cols]

# 保存原始划分
train_df.to_csv("train.csv", index=False)
val_df.to_csv("valid.csv", index=False)
print("train.csv 和 valid.csv 保存完成")

# ========== Step 3：从 train.csv 中再划出 test ==========
train_final_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 保存最终数据
train_final_df.to_csv("train_final.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("train_final.csv 和 test.csv 保存完成")
