import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn as sk
import sys
import matplotlib.pyplot as plt
from sklearn import svm
from jedi.api.refactoring import inline

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris, make_moons
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict, train_test_split
from sklearn.base import clone
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier
#import graphviz
from sklearn import preprocessing
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier


'''
KNN es no parámetrico porque no tiene una función. NO REQUIERE UNA DISTRIBUCIÓN ESPECÍFICA
KNN: Se ha utilizado para predecir el cancer

DESVENTAJA: No es eficiente en datos grandes
            Susceptible a sobreajustes 
            
            
NO UTILIZA NINGUNA FUNCION. 

PARA REALIZAR PREDICCIONES NECESITA BUSCAR LO QUE TIENE A SU ALREDEDOR PARA CLASIFICARLO


PASOS:
1) ESCOGE EL NÚMERO K
2) SELECCIONA LA DISTANCIA MÉTRICA
3) ENCUENTRA EL KNN DE LA MUESTRA
4) ASIGNA LA ETIQUETA CLASE POR MAYORÍA DE VOTOS

'''


df = sns.load_dataset('iris')
print(df.head())

X_train = df[['petal_length', 'petal_width']]
print('Impresion de datos de X_train')
print(X_train.head())
species_to_num = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['species'] = df['species'].map(species_to_num)
y_train = df['species']

print('Impresión de datos de Y_Train')
print(y_train.head())

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)

Xv = X_train.values.reshape(-1, 1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y_train.min(), y_train.max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print('Impresion de xx')
print(xx)
print(xx.ravel())

print('Impresion de yy')
print(yy)
print(yy.ravel())


z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

fig = plt.figure(figsize=(8,5))
ax = plt.contourf(xx, yy, z, cmap='afmhot', alpha=0.3)

plt.figure(figsize=(8,5))
plt.contourf(xx, yy, z, cmap='afmhot', alpha=0.3)
plt.scatter(X_train.values[:, 0], X_train.values[:, 1], c=y_train, s=40, alpha=0.9, edgecolors='k')
plt.show()



'''
Proyecto Detección de Cancer
-------------------------------------------------------------------------------------------------------------
'''


def print_score(clf, X_train, X_test, y_train, y_test, train=True):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)

    if train:
        '''
        training process
        '''
        res = clf.predict(X_train)
        print("Train Result: \n")
        print("Accuracy score: {0:.4f}\n".format(accuracy_score(y_train, res)))
        print("Classification Report: \n {}\n".format(classification_report(y_train, res)))
        print("Confussion Matrix: \n {}\n".format(confusion_matrix(y_train, res)))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_train), lb.transform(res))))

    elif train == False:

        res_test = clf.predict(X_test)
        print("Train Result: \n")
        print("Accuracy score: {0:.4f}\n".format(accuracy_score(y_test, res_test)))
        print("Classification Report: \n {}\n".format(classification_report(y_test, res_test)))
        print("Confussion Matrix: \n {}\n".format(confusion_matrix(y_test, res_test)))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_test), lb.transform(res_test))))






col = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv("C:/Users/Armando/Documents/MachineLearning/Datasets/breast-cancer-wisconsin.data.csv", names=col, header=None)

np.where(df.isnull())
print(df.head())
print(df.info())
analDF = df[df['Bare Nuclei'] == "?"]
print(analDF)

print(df['Class'].value_counts())


df['Bare Nuclei'].replace("?", np.NaN, inplace=True)
df = df.dropna()
print(df['Bare Nuclei'].describe())
print(df['Bare Nuclei'].value_counts())

# Class 2 es Cancer Benigno y 4 es cancer maligno
# Se transforma 2 en cero y 4 en 1 para una mejor clasificación

df['Class'] = df['Class'] / 2 - 1
print(df.columns)


# Creación de variables X, y
X = df.drop(['id', 'Class'], axis=1)
x_col = X.columns
y = df['Class']
print(X)

X = StandardScaler().fit_transform(X.values)        # Normalización de los datos Transforma los datos para que tenga una distribución de promedio 0 y desviación estandar 1
df1 = pd.DataFrame(X, columns=x_col)
print(df1.head())

X_train, X_test, y_train, y_test = train_test_split(df1, y, train_size=0.8, random_state=42)
pd.DataFrame(MinMaxScaler().fit_transform(df.drop(['id', 'Class'], axis=1).values), columns=x_col).head()

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

print_score(knn, X_train, X_test, y_train, y_test, train=False)
print_score(knn, X_train, X_test, y_train, y_test, train=True)

knn.get_params()
params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

grid_search_cv = GridSearchCV(KNeighborsClassifier(), params, n_jobs=-1, verbose=1, cv=10)
grid_search_cv.fit(X_train, y_train)
print(grid_search_cv.best_estimator_)


print_score(grid_search_cv, X_train, X_test, y_train, y_test, train=False)
print_score(grid_search_cv, X_train, X_test, y_train, y_test, train=True)

grid_search_cv.cv_results_['mean_test_score']
grid_search_cv.cv_results_