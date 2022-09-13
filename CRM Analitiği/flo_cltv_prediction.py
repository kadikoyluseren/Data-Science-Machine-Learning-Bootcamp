import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)  # None yerine yazdığın sayı kadar gösterir
# pd.set_option('display.max_rows', None) #bütün satırları görmek istersen
pd.set_option('display.float_format', lambda x: '%.5f' % x)  # ondalık virgül sonrası basamak sayısı

# G1
# 1
df_ = pd.read_csv("week3/datasets/flo_data_20k.csv")
df = df_.copy()


# 2
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)


# 3
df.isnull().sum()
df.describe().T

replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")

# 4
df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + \
                                         df["order_num_total_ever_offline"]
df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_online"] + \
                                              df["customer_value_total_ever_offline"]

df.head()

# 5
df.dtypes
# burayı döngülü yapabilirsin
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

# G2
# 1
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

# 2
# df_cltv["customer_id"] = df["master_id"]
# df_cltv["recency_cltv_weekly"] =
# df_cltv["T_weekly"]
# df_cltv["frequency"]
# df_cltv["monetary_cltv_avg"]

df_cltv=pd.DataFrame({ "customer_id":df["master_id"],
                       "recency_cltv_weekly":((df["last_order_date"]-df["first_order_date"]).dt.days)/7,
                       "T_weekly": ((today_date-df["first_order_date"]).dt.days)/7,
                       "frequency": df["order_num_total_ever_omnichannel"],
                       "monetary_cltv_avg": df["customer_value_total_ever_omnichannel"]/df["order_num_total_ever_omnichannel"]
})

df_cltv.head()

# G3
# 1
bgf=BetaGeoFitter(penalizer_coef=0.001) # bunu nasıl belirliyoruz

bgf.fit(df_cltv['frequency'],
        df_cltv['recency_cltv_weekly'],
        df_cltv['T_weekly'])

df_cltv["exp_sales_3_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(12,  # kaç haftalık tahmin
                                                        df_cltv['frequency'],
                                                        df_cltv['recency_cltv_weekly'],
                                                        df_cltv['T_weekly'])

df_cltv["exp_sales_6_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(24,  # kaç haftalık tahmin
                                                        df_cltv['frequency'],
                                                        df_cltv['recency_cltv_weekly'],
                                                        df_cltv['T_weekly'])

# 2
ggf=GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(df_cltv['frequency'], df_cltv['monetary_cltv_avg'])

df_cltv["exp_average_value"]=ggf.conditional_expected_average_profit(df_cltv['frequency'], # toplam işlem
                                                                     df_cltv['monetary_cltv_avg'])  # işlem başı ort kazanç

df_cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                   df_cltv['frequency'],
                                   df_cltv['recency_cltv_weekly'],
                                   df_cltv['T_weekly'],
                                   df_cltv['monetary_cltv_avg'],
                                   time=6,  # tahmin kaç aylık
                                   freq="W",  # T'nin frekans bilgisi.(girdiğin veri frekansı tipi haftalık="W")
                                   discount_rate=0.01) # belki indirim yaparsın onu göz önünde bulundur

df_cltv.sort_values("cltv", ascending=False).head(20)

# G4
# 1
df_cltv["segment"] = pd.qcut(df_cltv["cltv"], 4, labels=["D", "C", "B", "A"])


df_cltv.groupby("segment")["cltv","frequency"].agg({"sum","count","mean"})

df_cltv.describe().T























































