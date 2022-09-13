#####################################################
# IYZICO TRANSACTION FORECASTING
#####################################################

# !pip install lightgbm
# conda install lightgbm

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

df_ = pd.read_csv('week8/datasets/iyzico_data.csv', index_col=0, parse_dates=['transaction_date'])
df=df_.copy()

#####################################################
# EDA
#####################################################

df["transaction_date"].min(), df["transaction_date"].max()

check_df(df)

df[["merchant_id"]].nunique()

df.groupby(["merchant_id"])["Total_Transaction"].sum()

df.groupby(["merchant_id"])["Total_Paid"].sum()

#####################################################
# FEATURE ENGINEERING
#####################################################

df.head()

def create_date_features(df):
    df['month'] = df.transaction_date.dt.month
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df["is_wknd"] = df.transaction_date.dt.weekday // 4
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["merchant_id", "year"]).agg({"Total_Transaction": ["sum"]})

df.groupby(["merchant_id", "year"]).\
    agg({"Total_Transaction": ["sum"]}).\
    unstack().plot.bar(title='Total Transactions of Merchants by Years',
               ylabel='Total_Transaction', xlabel='Merchant ID')


########################
# Random Noise
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

########################
# Lag/Shifted Features
########################
# gecikme featureları


df.sort_values(by=['merchant_id', 'transaction_date'], axis=0, inplace=True)

pd.DataFrame({"Total_Transaction": df["Total_Transaction"].values[0:10], # ilk 10 gerçek satış değeri
              "lag1": df["Total_Transaction"].shift(1).values[0:10], # 1. gecikme
              "lag2": df["Total_Transaction"].shift(2).values[0:10], # 2. gecikme
              "lag3": df["Total_Transaction"].shift(3).values[0:10],
              "lag4": df["Total_Transaction"].shift(4).values[0:10]})

def lag_features(dataframe, lags):
    for lag in lags: # farklı gecikme değerlerinde gezilsin
        dataframe['transaction_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
        # gezme işlemleri neticesinde üretilecek olan featurelar otomatik/dinamik isimlendirilsin
            lambda x: x.shift(lag)) + random_noise(dataframe)
        # ve üzerine rastgele gürültü eklenerek yeni featurelar üretilsin
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
# 91 gün shift 3 ay öncesi
# elimdeki problem 3 aylık olduğu için 3 ay gerideki featurelara odaklanırsak ancak 3 ay sonrasının değerlerini doğru tahminleriz
# 3 ay ya da katlarına yakın lag'lar seçtik

check_df(df)
# train laglar NaN # train'in 3 ay öncesinde veri yok ki
# test laglar dolu gelir # testte var
# traindeki 4.aydan itibaren dolu gelir

# ağaca dayalı yöntem kullanılacağı için NaN değerleri takmıyoruz

########################
# Rolling Mean Features
########################
# Hareketli Ortalama Özellikleri
# kendisi dahil önceki değerlere bakıyoruz
# yarını tahmin ederken sıkıntı
# o yüzden shift 1 ile 1 fark alıyoruz, kendini ihmal eidyoruz
pd.DataFrame({"Total_Transaction": df["Total_Transaction"].values[0:10],
              "roll2": df["Total_Transaction"].rolling(window=2).mean().values[0:10],
              "roll3": df["Total_Transaction"].rolling(window=3).mean().values[0:10],
              "roll5": df["Total_Transaction"].rolling(window=5).mean().values[0:10]})

pd.DataFrame({"Total_Transaction": df["Total_Transaction"].values[0:10],
              "roll2": df["Total_Transaction"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["Total_Transaction"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["Total_Transaction"].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_features(dataframe, windows):
    for window in windows: # kaç adım geriye gideyim
        dataframe['transaction_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])
# 1-1,5 yıl öncesi

########################
# Exponentially Weighted Mean Features
########################

pd.DataFrame({"Total_Transaction": df["Total_Transaction"].values[0:10],
              "roll2": df["Total_Transaction"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["Total_Transaction"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["Total_Transaction"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["Total_Transaction"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["Total_Transaction"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['transaction_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)

########################
# One-Hot Encoding
########################

df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = pd.get_dummies(df, columns=cat_cols)

check_df(df)

df.shape

########################
# Converting sales to log(1+sales)
########################

df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)
check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label() # sayısal bağımlı değişken gerçek değerler
    smape_val = smape(np.expm1(preds), np.expm1(labels))  # tahmin
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################

train = df.loc[(df["transaction_date"] < "2020-01-01"), :]

val = df.loc[(df["transaction_date"] >= "2020-01-01") & (df["transaction_date"] < "2020-04-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date',"Total_Transaction","Total_Paid", "year"]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM ile Zaman Serisi Modeli
########################
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000, # iterasyon/optim sayısı
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols) # cols : bağımsız değişkenlerin isimleri

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], # belirli iterasyon sonucu ilerleme olmazsa validasyon hatada, çalışmayı/modellemeyi durdurur
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)

feat_imp = plot_lgb_importances(model, num=200)
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values
imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

#train = df.loc[~df.Total_Transaction.isna()]
train = df.loc[(df["transaction_date"] < "2020-01-01"), :]
test = df.loc[(df["transaction_date"] >= "2020-01-01") & (df["transaction_date"] < "2020-03-01"), :]
cols = [col for col in train.columns if col not in ['transaction_date', "Total_Paid","Total_Transaction","year"]]
Y_train = train['Total_Transaction']
X_train = train[cols]

test = df.loc[df.Total_Transaction.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)
# log alınmış halleri

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)
# log tersi aldık, orj satış sayıları geldi

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


