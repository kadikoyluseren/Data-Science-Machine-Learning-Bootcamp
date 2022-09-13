import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1
df = sns.load_dataset("titanic")
df.head()

# 2
df["sex"].value_counts()

# 3
# [df[col].nunique() for col in df.columns]
{col: df[col].nunique() for col in df.columns}

# 4
df["pclass"].nunique()
#len(df["pclass"].nunique()) olmadı

# 5
[df[col].nunique() for col in df.columns if col in ["pclass", "parch"]]
df[["pclass","parch"]].nunique()
# 6
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df.dtypes

# 7
df[df["embarked"] == "C"]

# 8
df[df["embarked"] != "S"]

# 9
df[(df["age"] < 30) & (df["sex"] == "female")]

# 10
df[(df["fare"] > 500) | (df["age"] > 70)]

# 11
df.isnull().sum()

# 12
df = df.drop("who", axis=1)

# 13
df["deck"]
df = df.fillna({"deck": "C"})

# df.fillna(df["deck"].mode(),inplace=True)
# mod_of_deck=df["deck"].mode()
# df = df.fillna({"deck": mod_of_deck})

# df["deck"].fillna(df["deck"].mode(), inplace=True)
# df = df.fillna({"deck": (df["deck"].mode())})

# df.loc[:, df.columns.str.contains("deck")].apply(lambda x: df.mode()).head()

# for col in df.columns:
#     if ("deck" in col) & (df["deck"].empty()):
#             df["deck"] = df["deck"].mode()

# df["deck"].mode()

# df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()
# df[["age"]].apply(lambda x: x/10 if x==22 else x).head()
# df[["deck"]]=df[["deck"]].apply(lambda x: x.mode() if x.empty()==True else x).head()
# df[["deck"]].apply(lambda x: x.mode()).head()

# if df["deck"].isnull():
#    df["deck"]=df["deck"].mode()

# [df["deck"].mode() for i in df["deck"] if df["deck"].isnull()==True]

# 14
# df["age"].dtype
df["age"]
df["age"].fillna(df["age"].median(), inplace=True)

# 15
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})


# 16
def convert_bool(dataframe, variable, cut_value):
    return pd.cut(dataframe[variable], [0, cut_value - 1, df[variable].max()], labels=[0, 1])

def age_30_flag(age):
    if age<30:
        return 1
    else:
        return 0


df["age_flag"] = convert_bool(df, "age", 30)

# df["age_flag"] = pd.cut(df["age"], [0, 29, df["age"].max()], labels=[0, 1])
# df.drop("age_",axis=1, inplace=True)

# def bool_age(dataframe, variable):
#     if df["age"] < 30:
#         df["age_flag"] = 1
#     else:
#         df["age_flag"] = 0
#     # return df["age_flag"]


# df[(df["age"]<30) | (df["age"]>=30)]


# 17
df = sns.load_dataset("tips")

# 18
df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

# 19
df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

# 20
# lunch-kadın total_bill-tip day'e göre sum min max mean
df[(df["sex"] == "Female") & (df["time"] == "Lunch")].groupby(["day"]).agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                            "tip": ["sum", "min", "max", "mean"]})

# 21
# df[(df["size"]<3) & (df["total_bill"]>10)].mean()
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), :]["total_bill"].mean()
#df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

# 22
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# 23
x=df.groupby("sex")["total_bill"].mean()
df[df["sex"]=="Female"]["total_bill"].mean()

# total_bill_flag

def sex_mean(sex, variable):
    if sex=="Female":
        if df[df["sex"] == "Female"][variable]< df[df["sex"] == "Female"][variable].mean():
            df["total_bill_flag"] = 0
        else:
            df["total_bill_flag"] = 1
    else:
        if df[df["sex"] == "Male"][variable] < df[df["sex"] == "Male"][variable].mean():
            df["total_bill_flag"] = 0
        else:
            df["total_bill_flag"] = 1

df["total_bill_flag"]=0

def sex_mean(sex, variable):
    if sex=="Female":
        return df[df["sex"] == "Female"][variable].mean()
    else:
        return df[df["sex"] == "Male"][variable].mean()
sex_mean("Female","total_bill")

# 24
# total_bill_flag cinsiyete göre ortalamanın altı ve üstü olanların sayısı
df.groupby("sex")[df["total_bill_flag"]==0].count()

# 25
new_df = df.sort_values("total_bill_tip_sum",ascending=False)

