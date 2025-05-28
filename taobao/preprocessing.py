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

# ========== Step 2：按时间划分 train、valid 和 test ==========
train_start = pd.to_datetime("20170506").date()
train_end = pd.to_datetime("20170511").date()
valid_day = pd.to_datetime("20170512").date()
test_day = pd.to_datetime("20170513").date()

# 按日期划分
train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
valid_df = df[df["date"] == valid_day]
test_df = df[df["date"] == test_day]

# 按点击先后排序
train_df = train_df.sort_values(by=['user_id_enc', 'datetime'])
valid_df = valid_df.sort_values(by=['user_id_enc', 'datetime'])
test_df = test_df.sort_values(by=['user_id_enc', 'datetime'])

train_df = train_df[feature_cols]
valid_df = valid_df[feature_cols]
test_df = test_df[feature_cols] 

# 保存数据
train_df.to_csv("train.csv", index=False)
valid_df.to_csv("valid.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("train_final.csv, valid.csv 和 test.csv 保存完成")
