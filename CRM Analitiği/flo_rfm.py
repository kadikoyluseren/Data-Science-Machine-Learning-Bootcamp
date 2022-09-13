import pandas as pd
import datetime as dt

# G1
# 1
df_ = pd.read_csv("week3/datasets/flo_data_20k.csv")
df = df_.copy()

# 2
df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.dtypes

# 3
df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + \
                                         df["order_num_total_ever_offline"]
df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_online"] + \
                                              df["customer_value_total_ever_offline"]

# 4
df.dtypes
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

# 5
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_ever_omnichannel": "sum",
                                 "customer_value_total_ever_omnichannel": "sum"}). \
    sort_values(by="customer_value_total_ever_omnichannel", ascending=False)

# 6
df.groupby("master_id").agg({"customer_value_total_ever_omnichannel": "sum"}). \
    sort_values(by="customer_value_total_ever_omnichannel", ascending=False).head(10)

df.groupby("master_id").agg({"order_num_total_ever_omnichannel": "sum"}). \
    sort_values(by="order_num_total_ever_omnichannel", ascending=False).head(10)


def data_prep(df):
    # omnichannel variables
    df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + \
                                             df["order_num_total_ever_offline"]
    df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_online"] + \
                                                  df["customer_value_total_ever_offline"]

    # object to datetime
    # burayı daha sonra column date içeriyorsa vs gibi düzenleyebilirsin
    df["first_order_date"] = pd.to_datetime(df["first_order_date"])
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])
    df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
    df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
    return df



# G2
# 1
today_date = dt.datetime(2021, 6, 1)

# 2-3
rfm = df.groupby('master_id').agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                              'order_num_total_ever_omnichannel': lambda num: num.nunique(),
                                              "customer_value_total_ever_omnichannel": lambda price: price.sum()})
# 4
rfm.columns = ['recency', 'frequency', "monetary"]

# G3
# 1-2
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# 3
rfm["RF_SCORE"]=(rfm["recency_score"].astype(str)+rfm["frequency_score"].astype(str))

# G4
#1
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm = rfm[["recency", "frequency", "monetary", "segment"]]

# G5
# 1
rfm.groupby("segment")["recency","frequency","monetary"].mean()

# 2.a
loyal_champ=rfm[(rfm["segment"]=="champions") | (rfm["segment"]=="loyal_customers")]
interest_kadin=df[(df["interested_in_categories_12"]).str.contains("KADIN")]

loyal_champ.reset_index(inplace=True)
interest_kadin.reset_index(inplace=True)
lck_df=pd.DataFrame(list(set(loyal_champ.master_id) & set(interest_kadin.master_id)))

#w_hvc = pd.merge(interest_kadin, loyal_champ, on=["master_id"]) # neslihan hanım
#len(list(set(loyal_champ.master_id) & set(interest_kadin.master_id)))

lck_df.to_csv("case1.csv")

# 2.b
loose_sleep_new = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="about_to_sleep") | (rfm["segment"]=="new_customers")]

searchfor=["ERKEK","COCUK"]
interest_cocuk_erkek = df[(df["interested_in_categories_12"]).str.contains("|".join(searchfor))] # str.contains("ERKEK|COCUK")

loose_sleep_new.reset_index(inplace=True)
interest_cocuk_erkek.reset_index(inplace=True)

lsn_ec=pd.DataFrame(list(set(loose_sleep_new.master_id) & set(interest_cocuk_erkek.master_id)))

lsn_ec.to_csv("case2.csv")


















