import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# G1
# 1
df_control = pd.read_excel("week4/datasets/ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("week4/datasets/ab_testing.xlsx", sheet_name="Test Group")

# 2
df_control.head()
df_test.head()

df_control.size
df_test.size

df_control.describe().T
df_test.describe().T

# 3
df=pd.concat((df_control,df_test))
df.head()
df.size

# G2
# 1

# H0: M1 = M2
# Kontrol ve Test Grupları Kazanç Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2
df_control["Purchase"].mean()
df_control["Purchase"].median()
df_control.describe().T

df_test["Purchase"].mean()
df_test["Purchase"].median()
df_test.describe().T

# baktığımızda arada fark var gibi görünmekte
# ama bu şans eseri mi istatistiki bir fark mı

# G3
# 1

# Normallik
# Ho normal dağılım sağlanmaktadır
test_stat, pvalue = shapiro(df_control["Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df_test["Purchase"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# reddedilemez

# VH
# Ho varyanslar homojendir
test_stat, pvalue = levene(df_control["Purchase"],
                           df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# reddedilemez

# 2
test_stat, pvalue = ttest_ind(df_control["Purchase"],
                              df_test["Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# reddedilemez, fark yoktur


#direkt mannwithnyu yapsam?

# parametriklik bağımlılıkla ilgili
# bir parametreye bağlı hareket varsa parametrik
# birden fazla parametre varsa non-param










































