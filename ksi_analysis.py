# -*- coding: utf-8 -*-
"""ksi_analysis.ipynb

# Toronto Traffic Accident Analysis

- Daniel Siegel - 101367445
- Michael McAllister - 101359469
- Hom Kandel - 101385341
- Eduardo Bastos de Moraes - 101345799

## Project Definition:

For this project, we were expected to build either a Random Forest Classifier or a Random Forest Regressor. This model is meant to analyze a database with the following 3 criteria:

1.	The database needs to have at least 2 classes
2.	The database should at least have 300 samples/rows
3.	The database should at least have 8 columns/features

Report:
Put your results in a report. Your report needs to include the following sections:
•	The Problem statement,
•	The Database,
•	The model you picked to solve the problem,
•	Results, the model performance (test, valid), the loss, predictions…
(like use of confusion matrices etc…)
•	Conclusions

In your results please comment and discuss the followings:
1.	Evaluate the model, how?
2.	How your model change when the number of estimators (decision trees) changes?
3.	What is the best number of estimators? How you can select the best number of estimators?

# Problem Statement

For our project, we have decided to analyze the Killed or Seriously Injured (KSI) database if we can determine the conditions most likely to lead to fatality.

# Import the database and libraries
"""

import pandas as pd
import numpy as np
import pandas_profiling
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/eduardomoraes/KSI/main/KSI_CLEAN.csv')

"""# Initial observations of the dataset"""

# Criteria 1
print("The target class, FATAL, aligns with the number of fatalities in a car accident. There are two classes - 0 (no) and 1 (yes). The data is split as follows:")
df['FATAL'].value_counts()

# Criteria 2 & 3
print('The dataset has', df.shape[0], 'rows and', df.shape[1], 'columns.')

df.head(5)

df.info()

df.columns

df['CYCLISTYPE']

plt.figure(figsize=(20,5))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)

# The dataset is pretty good. it doesn't have any Nan values on its columns

for column_name in ['FATAL','DISABILITY', 'ALCOHOL', 'REDLIGHT',
                    'AG_DRIV', 'SPEEDING', 'PASSENGER', 'EMERG_VEH',
                    'TRSN_CITY_VEH', 'TRUCK', 'MOTORCYCLE', 'AUTOMOBILE',
                    'CYCLIST', 'PEDESTRIAN']:
    df[column_name] = df[column_name].astype('int64')

df['FATAL']

df.profile_report()

"""# Data Preprocessing"""

string_columns=df.select_dtypes(include=[object])
string_columns.head(5)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

string_columns_transformed = string_columns.apply(le.fit_transform)
string_columns_transformed.head()

string_columns.shape

string_columns_transformed.shape

df = df.drop(columns=df.select_dtypes(['object']).columns)

df.shape

df=pd.concat([df,string_columns_transformed],axis=1)
df.shape

df.columns

df['FATAL']

"""## Seletion of Best best features"""

y=df.iloc[:,25]
#X=df.iloc[:,:-1]
X = df.drop(columns=['FATAL', 'DISABILITY', 'ALCOHOL', 'REDLIGHT', 'AG_DRIV', 'SPEEDING', 'PASSENGER', 'EMERG_VEH', 'TRSN_CITY_VEH', 'TRUCK', 'MOTORCYCLE', 'AUTOMOBILE', 'CYCLIST', 'PEDESTRIAN', 'FATAL_NO', 'ACCNUM'])

X.head(5)
X.shape

"""#Applying Filter Feature Selection - Pearson Correlation"""

feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=10


def corr_selector(X, y,num_feats):
    # Your code goes here (Multiple lines)
    corr_list = []
    feature_name = X.columns.tolist()

    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        corr_list.append(cor)

    corr_list = [0 if np.isnan(i) else i for i in corr_list]

    corr_feature = X.iloc[:,np.argsort(np.abs(corr_list))[-num_feats:]].columns.tolist()

    corr_support = [True if i in corr_feature else False for i in feature_name]
    # Your code ends here
    return corr_support, corr_feature

corr_support, corr_feature = corr_selector(X, y,num_feats)
print(str(len(corr_feature)), 'selected features')

corr_feature

"""#Appying Chi-Squared Selector function"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def chi_squared_selector(X, y, num_feats):

    X_n = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_n, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()

    return chi_support, chi_feature

chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')

chi_feature

"""#Appying Wrapper Feature Selection - Recursive Feature Elimination"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_n = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_n, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature

rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')

rfe_feature

"""#Appying Embedded Selection - Lasso: SelectFromModel"""

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature

embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')

embedded_lr_feature

"""#Appying Tree based(Random Forest): SelectFromModel"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=500), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature

embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedded_rf_feature)), 'selected features')

embedded_rf_feature

"""#Appying Tree based(Light GBM): SelectFromModel"""

from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature

embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
print(str(len(embedded_lgbm_feature)), 'selected features')

embedded_lgbm_feature

"""#What are the best features?"""

pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':corr_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)
