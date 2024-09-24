import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
#%matplotlib inline
sns.set_style('whitegrid')

col = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

df = pd.read_csv("C:/Users/Armando/Documents/Python Scripts/MachineLearning/Datasets/breast-cancer-wisconsin.data.csv", names=col, header=None)
df.head

#Data Pre-processing
np.where(df.isnull())
df.info()

df['Bare Nuclei'].describe()
df['Bare Nuclei'].value_counts()

#How do we drop the ?
df[df['Bare Nuclei'] == "?"]
df['Class'].value_counts()
df['Bare Nuclei'].replace("?", np.NAN, inplace = True)
df = df.dropna()

df['Bare Nuclei'].value_counts()
df['Class'] = df['Class']/ 2 -1
df['Class'].value_counts()

df.columns

X = df.drop(['id', 'Class'], axis = 1)
X_col = X.columns

# Modelo
y = df['Class']
X = StandardScaler().fit_transform(X.values)

df1 = pd.DataFrame(X, columns = X_col)
df1.head()
X_train, X_test, y_train, y_test = train_test_split(df1, y, train_size=0.8, random_state=42)

pd.DataFrame(MinMaxScaler().fit_transform(df.drop(['id', 'Class'], axis=1).values), columns=X_col).head()

















