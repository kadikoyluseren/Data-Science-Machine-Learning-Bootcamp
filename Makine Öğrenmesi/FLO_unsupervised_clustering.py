##################################################################
# FLO Gözetimsiz Öğrenme İle Müşteri Segmentasyonu
##################################################################
# müşterilerin davranışlarına göre öbeklenmesi

import seaborn as sns
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def check_df(dataframe, head=5):
    """

    Veri setinin shape, types, head, tail, NA, quantities değerlerini ekrana yazdırır.
    Parameters
    ----------
    dataframe: dataframe
        Kullanılacak veri setidir.
    head: int
        Veri setinden kaç satırın gösterilmek istendiğini ifade eder.
    Returns
    -------

    Examples
    -------

    """
    print("### Shape ###")
    print(dataframe.shape)
    print("### Types ###")
    print(dataframe.dtypes)
    print("### Head ###")
    print(dataframe.head(head))
    print("### Tail ###")
    print(dataframe.tail(head))
    print("### NA ###")
    print(dataframe.isnull().sum())
    print("### Quantities ###")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20): # th: threshold
    """
    Veri setindeki kategorik, numerik, kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframes: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    cat_cols: list
        Numerik değişken listesi
    cat_cols: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    -------
    cat_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols=[col for col in df.columns if df[col].dtypes in ["int","float"]]
    num_cols=[col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def my_scatter(cluster_type): # hi_cluster/k_cluster
    fig = plt.figure(figsize=(10, 7))
    ax = Axes3D(fig)
    x = rfm.iloc[:, 0]
    y = rfm.iloc[:, 1]
    z = rfm.iloc[:, 2]
    c = rfm[cluster_type]
    ax.scatter(x, y, z, c=c, cmap='coolwarm')
    cb = ax.scatter(x, y, z, c=c, cmap='coolwarm')
    plt.title('First 3 Principal Components')
    ax.set_ylabel('frequency')
    ax.set_xlabel('recency')
    ax.set_zlabel('monetary')
    plt.colorbar(cb)
    plt.legend()


##################################################################
# Görev 1:  Veriyi Hazırlama
##################################################################
# Adım1:  flo_data_20k.csv verisini okutunuz
df_ = pd.read_csv("week3/datasets/flo_data_20k.csv")
df = df_.copy()
df.head()

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Adım2:  Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.


def data_prep(df):
    # omnichannel variables
    df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + \
                                             df["order_num_total_ever_offline"]
    df["customer_value_total_omnichannel"] = df["customer_value_total_ever_online"] + \
                                                  df["customer_value_total_ever_offline"]

    # object to datetime
    date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    return df


data_prep(df)

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby('master_id').agg({'last_order_date': lambda date: (today_date - date.max()).days,
                                    'order_num_total_omnichannel': lambda num: num.sum(),
                                    "customer_value_total_omnichannel": lambda price: price.sum()})
rfm.columns = ['recency', 'frequency', "monetary"]


##################################################################
# Görev 2:  K-Means ile Müşteri Segmentasyonu
##################################################################
# Adım 1: Değişkenleri standartlaştırınız.
sc = MinMaxScaler((0, 1))
rfm_std = sc.fit_transform(rfm) # fit_transform sonrası tip numpy array'i
type(rfm)
type(rfm_std)
rfm_std[0:5]

# Adım 2: Optimum küme sayısını belirleyiniz.
kmeans = KMeans(n_clusters=4).fit(rfm_std) # KMeans modeli kurulmuş oldu
kmeans.get_params()
kmeans.inertia_

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(rfm_std)
    ssd.append(kmeans.inertia_) # inertia : SSD : en yakın clustera olan uzaklıklar
# yüksek cluster sayısı = düşük hata

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(rfm_std)
elbow.show()
# mavi: distortion : distance
# yeşil: fit etme süresi
elbow.elbow_value_

# Adım 3:  Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(rfm_std)
kmeans.n_clusters # cluster sayısı
kmeans.cluster_centers_ # cluster merkezleri (stdlaştırılmış birer gözlem birimi)
kmeans.labels_ # küme etiketleri, hesaplanan kümeler

clusters_kmeans = kmeans.labels_

rfm["k_cluster"] = clusters_kmeans+1
rfm.head()

my_scatter("k_cluster")

# Adım 4: Her bir segmenti istatistiksel olarak inceleyiniz.
rfm.groupby("k_cluster").agg(["count","mean","median"])


##################################################################
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
##################################################################
# Adım 1: Görev2'de standırlaştırdığınız dataframe'i kullanarak optimum kümesayısını belirleyiniz.
rfm_std

hc_average = linkage(rfm_std, "average")

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
#dend = dendrogram(hc_average)
dend=dendrogram(hc_average,
           truncate_mode="lastp", # daha sade görüntü
           p=10, # en aşağıda görünecek dal sayısı
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.4, color='r', linestyle='--')
plt.axhline(y=0.5, color='y', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

# Adım 2: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average") # ön tanımlı uzaklık metriği Euclid
# ward ve average yaygın, complete de karşılaşılabilir

hi_clusters = cluster.fit_predict(rfm_std)

rfm["hi_cluster"] = hi_clusters+1

my_scatter("hi_cluster")

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
rfm.groupby("hi_cluster").agg(["count","mean","median"])



#############################################
df_clustered=df_.copy()
df_clustered["k_cluster"] = clusters_kmeans+1
df_clustered["hi_cluster"] = hi_clusters+1
df_clustered

df_clustered[df_clustered["k_cluster"]==df_clustered["hi_cluster"]]
df_clustered[df_clustered["k_cluster"]!=df_clustered["hi_cluster"]]