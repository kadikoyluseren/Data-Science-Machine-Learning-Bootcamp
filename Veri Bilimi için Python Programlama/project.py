import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = pd.read_csv("../input/titanic/train.csv")
df.head()

df.shape

df.columns

df=df.drop(["PassengerID"],axis=1)

df.info()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if (df[col].nunique() < 10) and df[col].dtypes in ["int64", "float64"]]
num_but_cat

cat_but_car = [col for col in df.columns if
                 (df[col].nunique() > 30) and str(df[col].dtypes) in ["category", "object"]]
cat_but_car

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]
num_cols


def view_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if (df[col].nunique() < 10) and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   (df[col].nunique() > 20) and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols,num_cols,cat_but_car = view_col_names(df)

cat_cols
num_cols
df.describe().T

df.isnull().sum()

df = df.drop(['Cabin'],axis=1)

df = df.drop(['Ticket'],axis=1)

#we can also drop the Name feature since it's unlikely to useful information. You realize that this feature is cardinal variable.
df = df.drop(['Name'],axis=1)

df["Embarked"].unique()

df["Embarked"].value_counts()

print("Number of people embarking in Southampton (S):")
print(df[df["Embarked"] == "S"].shape[0])

print("Number of people embarking in Cherbourg (C):")
print(df[df["Embarked"] == "C"].shape[0])


print("Number of people embarking in Queenstown (Q):")
print(df[df["Embarked"] == "Q"].shape[0])

df = df.fillna({"Embarked": "S"})

df["Age"].median()

df.fillna(df["Age"].median(),inplace=True)

print(df.isnull().sum())
print("-"*15)

def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("-"*30,'\n')


for col in cat_cols:
    cat_summary(df,col)


def cat_summary(dataframe, col_name, plot=False):
    pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                  "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)

fig, qaxis = plt.subplots(1, 3,figsize=(20,10))

sns.barplot(x = 'Sex', y = 'Survived', data=df, ax = qaxis[0])
sns.barplot(x = 'Pclass', y = 'Survived', data=df, ax = qaxis[1])
sns.barplot(x = 'Embarked', y = 'Survived', data=df, ax = qaxis[2])

print("Percentage of females who survived:", df["Survived"][df["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df["Survived"][df["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


print("Percentage of 1st Class Passengers who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of 2nd Class Passengers who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of 3rd Class Passengers who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)

fig, qaxis = plt.subplots(1,2,figsize=(20,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=df, ax = qaxis[0])
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=df, ax  = qaxis[1])


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    dataframe[numerical_col].describe(quantiles)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df,col,plot=True)


plt.figure(figsize=[16,12])

plt.subplot(231)
plt.hist(x = [df[df['Survived']==1]['Fare'], df[df['Survived']==0]['Fare']],
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(232)
plt.hist(x = [df[df['Survived']==1]['Age'], df[df['Survived']==0]['Age']],
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age ($)')
plt.ylabel('# of Passengers')
plt.legend()


