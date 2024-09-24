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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict, train_test_split
from sklearn.base import clone
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import graphviz
from sklearn import preprocessing
import xgboost as xgb



'''
Solo compara los resultados de todos los modelos de Ensemble Machine Learning, no te pierdes de nada
'''

path = 'C:/Users/Armando/Documents/Ciencia_Datos/Udemy/Datos/HumanResources_Analytics/WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(path)
print(df.head())


'''Función para imprimir score'''

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


num_col = list (df.describe().columns)
col_categorical = list(set(df.columns).difference(num_col))
remove_list = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Department', 'BusinessTravel']
col_numerical = [e for e in num_col if e not in remove_list]
attrition_to_num = {'Yes': 0, 'No': 1}
df['Attrition_num'] = df['Attrition'].map(attrition_to_num)
col_categorical.remove('Attrition')
df_cat = pd.get_dummies(df[col_categorical])
X = pd.concat([df[col_categorical], df_cat], axis=1)
y = df['Attrition_num']


X_train, X_test, y_train, y_test = train_test_split(X, y)

# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)
#
# print_score(clf, X_train, y_train, X_test, y_test, train=False)

'''BAGGING'''

# clf = BaggingClassifier(random_state=42)
# clf.fit(X_train, y_train)
#
# print_score(clf, X_train, y_train, X_test, y_test, train=False)





'''
Ensemble of Ensembles part 2
'''

print(df.Attrition.value_counts() / df.Attrition.count())

class_weight = {0: 0.839, 1: 0.161}

pd.Series(list(y_train)).value_counts() / pd.Series(list(y_train)).count()

forest = RandomForestClassifier(class_weight= class_weight, n_estimators=100)

ada = AdaBoostClassifier(base_estimator=forest, n_estimators=100, learning_rate=0.5, random_state=42)
ada.fit(X_train, y_train.ravel())

print_score(ada, X_train, X_test, y_train, y_test, train=True)
print_score(ada, X_train, X_test, y_train, y_test, train=True)