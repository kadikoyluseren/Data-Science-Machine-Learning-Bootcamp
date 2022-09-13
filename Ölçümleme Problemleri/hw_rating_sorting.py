import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("week4/datasets/amazon_review.csv")

df.head()

# G1
# 1
df["overall"].mean()

# 2
df["reviewTime"].dtype

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()

df["recency"] = (current_date - df["reviewTime"]).dt.days

df["quantile"] = pd.qcut(df["recency"], 4)

df["quantile"].unique()
# (0-280]
# (280-430]
# (430-600]
# (600-1063]


# ağırlıklarda birine 90 verip kalan 3üne 10u dağıtsam biraz manipüşasyon olur
df.loc[df["recency"] <= 280, "overall"].mean() * 28 / 100 + \
df.loc[(df["recency"] > 280) & (df["recency"] <= 430), "overall"].mean() * 26 / 100 + \
df.loc[(df["recency"] > 430) & (df["recency"] <= 600), "overall"].mean() * 24 / 100 + \
df.loc[df["recency"] > 600, "overall"].mean() * 22 / 100

# 3
df.loc[df["recency"] <= 280, "overall"].mean()

df.loc[(df["recency"] > 280) & (df["recency"] <= 430), "overall"].mean()

df.loc[(df["recency"] > 430) & (df["recency"] <= 600), "overall"].mean()

df.loc[df["recency"] > 600, "overall"].mean()

# G2

df["total_vote"].sum()
df["helpful_yes"].sum()

# 1
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df["helpful_no"].sum()


# 2

def score_pos_neg_diff(pos, neg):
    return pos - neg


def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    else:
        return pos / (pos + neg)


def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# 3
df.sort_values("wilson_lower_bound", ascending=False)
