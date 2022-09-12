################################
# Unsupervised Learning
################################

# pip install yellowbrick

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

################################
# K-Means
################################

# eyaletlerin suç istatistiklerine göre kümelere ayrılması

df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum() # eksik değer var mı
df.info() # kaç değişken kaç gözlem birimi ve tip bilgileri
df.describe().T # sayısal değişkenlerin betimsel istatistikleri
# murder min max 50% değerlerine bakıldığında sıkıntı yok gibi
# aykırı değer/anormallik yok gibi

# cinayet ortalamalarında 7.78 altında üstünde bakılabilir

# uzaklık ve grad desc temelli yöntemlerde değişkenlerin stdlaştırılamsı önemli

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df) # fit_transform sonrası tip numpy arrayi
df[0:5] # o yüzden head atmadık
#type(df)

# model kuralım
# ama bağımlı değişken yok / df içinde sadece bağımsız değişkenler var
kmeans = KMeans(n_clusters=4, random_state=17).fit(df) # KMeans modeli kurulmuş oldu # ön tanımlı cluster değeri 8)
kmeans.get_params()
# n_cluster bizim tarafımızdan belirlenmeli
# max_iter
# random state aynı değeri elde etmek için

kmeans.n_clusters # cluster sayısı
kmeans.cluster_centers_ # cluster merkezleri (stdlaştırılmış birer gözlem birimi)
kmeans.labels_ # küme etiketleri, hesaplanan kümeler
kmeans.inertia_ # ctrl kmeans ile sınıf içine bakalım, SSD: en yakın clustera olan uzaklıklar

################################
# Optimum Küme Sayısının Belirlenmesi
################################
# farklı k parametre değerlerine göre SSE'ye bakıp karar vermeliyiz.

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()
# gözlem birimi kadar cluster olursa 0 olur, kendilerine uzaklıkları 0'dır :DDDD
# çok yüksek cluster sayısında tabi ki hata düşük olacaktır

# grafiği kendi bilgimizle yorumlamalıyız
# çalışma başında iş bilgisi sayesinde bizim optimum küme sayısı hakkında fikir sahibi olmamız gerek

# eğimin en şiddetli olduğu noktalar

# neden farklı sonuçlar veriyor
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
# mavi: distortion: distance
# yeşil: fit etme süresi
elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################
# dışsal parametrenin optimum değerini yukarıda bulduk

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)
#  bunu yapma sebebim gözlem birimlerinin stdlaştırılmamış halini görmek

df["cluster"] = clusters_kmeans

df.head()

df["cluster"] = df["cluster"] + 1

df[df["cluster"]==5]

df.groupby("cluster").agg(["count","mean","median"])
# ortalması birbirine yakın clusterlar bir araya gelse? diye düşünebiliriz ama diğer faktörler de var
# ama benim için önemli bi değişken varsa ona göre kendim de gruplama yapabilirim

df.to_csv("clusters.csv")


################################
# Hierarchical Clustering
################################

df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)

#uzaklık temelli stdlaştırayım
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")
 # birleştirici : agglomerative
# gözlem birimlerini euclid uzaklığına göre bir araya getirir
# gözlem birimleri çalışma başında tek başına değerlendirilir
# en benzer olanlar bir araya getirilir iteratif olarak


plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10) # gözlem birimlerinin indexleri yazı boyutu
plt.show()

df.shape
# buna bakarak yapmak istediğim kümeleme sayısına göre kimlerin hangi kümede olacağını görürüm
# 2 küme istersem şunlar, 3 ise bunlar

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp", # daha sade görüntü
           p=15, # kaç yaprak olacak en sonda
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--') # 6 küme görüyorum ama 2'sini ayrı almak mantıksız 5 de yapabilirim
plt.axhline(y=0.6, color='b', linestyle='--') # 4 küme
plt.show()

# elbow gibi düşünülebilir

################################
# Final Modeli Oluşturmak
################################
# küme sayısına yukarıda karar verdik şimdi final modelde sıra
# hangi gözlem birimi hangi sınıfta olacak bilgisini vermemiz lazım

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average") # ön tanımlı uzaklık metriği Euclid
# ward ve average yaygın, complete de karşılaşılabilir
clusters = cluster.fit_predict(df)

df = pd.read_csv("week7/datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1

df
# ikisi de 5 cluster'a ayırdı ama farklı clusterlara atandı gözlem birimleri
# ikisinde ortak olanlar ve ayrık olanlar kimler??? ve neden???


################################
# Principal Component Analysis
################################
# veri setinin asıl amacı maaş tahmini
# ama biz bağımlı değişkenle ilgilenmiyoruz (salary)
# kategorikler ile de ilgilenmiyoruz

df = pd.read_csv("week7/datasets/Hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

# 16 değişken var amacım 2-3 değişkenle bu veri setini ifade etmek

df = StandardScaler().fit_transform(df)

pca = PCA() # model nesnesi
pca_fit = pca.fit_transform(df)

# bileşenlerin başarısı onların açıkladığı varyans oranlarına göre belirlenir
# bilgi = varyans

pca.explained_variance_ratio_ # değişkenlerin açıkladığı varyans oranları, bilgi oranları
np.cumsum(pca.explained_variance_ratio_) # peşpeşe bileşenlerin açıklayacağı oranlar
# cumsum : bileşenler bir araya geldiğinde toplam ne kadar açıklayabiliyorlar



################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

# veri görselleştirme işi içinse direkt 2 bileşene indirmeyi seçebilirim
# reg'de çoklu doğrusal bağlantı problemini gidermek için değişken sayısı kadar bileşen oluşturma ;
# böylece bilginin tamamı korunmuş olur ama değişkenler birbirinden bağımsız olur ;
# yükse korelasyon, çoklu doğrusal bağlantı problemine sahip olmaz.


################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


################################
# BONUS: Principal Component Regression
################################
# diyelim ki hitters veri seti doğrusal olarak modellenmek istiyor
# ve değişkenler arasında çoklu doğrusal bağlanto problemi var
# bu doğrusal regresyon modellerinde sağlanması gereken varsayımlardandır
# değişkenler arasında yüksek korelasyon çeşitli problemlere sebep olur


# PCR : Princ Comp Reg
# önce pca uygulayıp değişken boyutunu indirge
# sonra bunlar üzerine reg kur

df = pd.read_csv("week7/datasets/Hitters.csv")
df.shape

len(pca_fit)

# gözlem birimi sayım aynı ama değişkenim 3 adet

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1) # axis1 yanyana koy
final_df.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# sınıf sayısı 2
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
# daha sonra kullanma derdim yok fit etmiyorum
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
y.mean()


cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))

# Elimde bi veri seti var. Ama veri setinde label yok. Ama sınıflandırma problemi çözmek istiyorum.
# Ne yapabilirim?
# 1000 insan kümeleniyor 4 küme ile.
# Yeni müşteri geldi, ne yapacağız?
# Önce unsupervised cluster çıkarırım, etiketlerim.
# Sonra bir sınıflandırıcıya sokarım.


################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("week7/datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)
# bu fonksiyona göndereceğin X'in sayısal değerlerden oluştuğuna emin ol
plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("week7/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")




















