# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Toronto Traffic Accident Analysis
# 
# - Daniel Siegel - 101367445
# - Michael McAllister - 101359469
# - Hom Kandel - 101385341
# - Eduardo Bastos de Moraes - 101345799
# %% [markdown]
# ## Project Definition:
# 
# For this project, we were expected to build either a Random Forest Classifier or a Random Forest Regressor. This model is meant to analyze a database with the following 3 criteria:
# 
# 1.	The database needs to have at least 2 classes
# 2.	The database should at least have 300 samples/rows
# 3.	The database should at least have 8 columns/features
# 
# Report:
# Put your results in a report. Your report needs to include the following sections:
# •	The Problem statement, 
# •	The Database, 
# •	The model you picked to solve the problem, 
# •	Results, the model performance (test, valid), the loss, predictions…
# (like use of confusion matrices etc…)
# •	Conclusions
#                                                      
# In your results please comment and discuss the followings:
# 1.	Evaluate the model, how?
# 2.	How your model change when the number of estimators (decision trees) changes?
# 3.	What is the best number of estimators? How you can select the best number of estimators?
# 
# %% [markdown]
# # Problem Statement
# %% [markdown]
# For our project, we have decided to analyze the Killed or Seriously Injured (KSI) database if we can determine the conditions most likely to lead to fatality.
# %% [markdown]
# # Import the database and libraries

# %%
import pandas as pd
import numpy as np
import pandas_profiling
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px


# %%
data = pd.read_csv("https://raw.githubusercontent.com/danielmaxsiegel/GBC-ML1/main/datasets/KSI_CLEAN.csv")
RSEED = 42 # The answer to the ultimate question of life, the universe, and everything

# %% [markdown]
# # Initial observations of the dataset

# %%
# Criteria 1: At least 2 classes. We have ~11,000 rows of class A, ~2000 rows of class B, and a very small class C.
print("The target class, INJURY - the injury suffered by each person in the accident")
data['INJURY'].value_counts()


# %%
# Criteria 2 & 3: over 300 rows and 8 or more columns/features
print('The dataset has', data.shape[0], 'rows and', data.shape[1], 'columns.')


# %%
pd.set_option('display.max_columns', None)
data.head()


# %%
data.info()


# %%
#Adding the graphs for the problem statement chart 
graf1=data.groupby(['YEAR','INJURY'])['ACCNUM'].nunique().reset_index()
graf1.rename(columns = {'ACCNUM':'Accidents','YEAR':'Year','INJURY':'Injury'},inplace=True)
fig1 = px.bar(graf1,x="Year", y='Accidents', color="Injury",title='Number of accidents in Toronto by year',text='Accidents')
fig2 = px.pie(graf1, names='Injury',values='Accidents',color='Injury',title='Distribution of accidents in Toronto by injury',width=500)
print(fig1.show(),fig2.show())


# %%
# Clean INJURY column (severity of injury)
# It appears that the injury code left blank means no injury, or the party is on the police report but indirectly involved in the accident so left blank
data.loc[data['ACCNUM'] == 1311542, ['ACCNUM', 'IMPACTYPE', 'INVTYPE']]


# %%
data['ACCLASS'].value_counts()


# %%
data.loc[data['ACCNUM'] == 5000995174, ['ACCNUM', 'ACCLASS', 'FATAL', 'INJURY']]


# %%
data.loc[data['ACCNUM'] == 1311542, ['ACCNUM', 'IMPACTYPE', 'INVTYPE']]


# %%
# At first glance the data appears to be very clean
plt.figure(figsize=(20,5))
sns.heatmap(data.isna(), cbar=False, cmap='viridis', yticklabels=False)


# %%
#After trying numerous times to try and figure out the blanks values in the database, they've been setup as ' ' strings
plt.figure(figsize=(20,5))
sns.heatmap(data.eq(' '), cbar=False, cmap='viridis', yticklabels=False)

# %% [markdown]
# # Clean Data

# %%
# Drop superfluous columns - some with irrelevant data, some with duplicate information (such as "FATAL_NO")
data = data.drop(['Ward_ID', 'Hood_ID', 'ACCNUM', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTES', 'LATITUDE', 'LONGITUDE', 'Ward_Name', 'Hood_Name', 'Division', 'District', 'STREET1', 'STREET2', 'OFFSET', 'INITDIR', 'ACCLASS', 'FATAL_NO'], axis=1)
data.info()


# %%
for column_name in ['FATAL','DISABILITY', 'ALCOHOL', 'REDLIGHT', 
                    'AG_DRIV', 'SPEEDING', 'PASSENGER', 'EMERG_VEH', 
                    'TRSN_CITY_VEH', 'TRUCK', 'MOTORCYCLE', 'AUTOMOBILE', 
                    'CYCLIST', 'PEDESTRIAN']:
    data[column_name] = data[column_name].astype('int64')


# %%
# Clean ROAD_CLASS column (which type of road the drivers were on) was setup as an ordinal feature
data['ROAD_CLASS'] = data['ROAD_CLASS'].replace(to_replace=['Minor Arterial', 'Laneway', 'Local'], value=0, inplace=False, limit=None, regex=False, method='pad')
data['ROAD_CLASS'] = data['ROAD_CLASS'].replace(to_replace=['Major Arterial', 'Major Arterial Ramp'], value=1, inplace=False, limit=None, regex=False, method='pad')
data['ROAD_CLASS'] = data['ROAD_CLASS'].replace(to_replace=['Collector', 'Expressway', 'Expressway Ramp'], value=2, inplace=False, limit=None, regex=False, method='pad')

# Clean VISIBILITY column (the weather conditions surrounding the accident) was setup as an ordinal feature
original_data = data['VISIBILITY'].value_counts()
data['VISIBILITY'] = data['VISIBILITY'].replace(to_replace=['Clear', 'Other', ' '], value=0, inplace=False, limit=None, regex=False, method='pad')
data['VISIBILITY'] = data['VISIBILITY'].replace(to_replace=['Rain', 'Strong wind', 'Fog, Mist, Smoke, Dust'], value=1, inplace=False, limit=None, regex=False, method='pad')
data['VISIBILITY'] = data['VISIBILITY'].replace(to_replace=['Snow', 'Freezing Rain', 'Drifting Snow'], value=2, inplace=False, limit=None, regex=False, method='pad')

# Clean LIGHT column (the amount of light present at the time of accident) was setup as an ordinal feature, Dusk and Dawn were merged due to similar lighting
data['LIGHT'] = data['LIGHT'].replace(to_replace=['Daylight', 'Daylight, artificial', ' ', 'Other'], value=0, inplace=False, limit=None, regex=False, method='pad')
data['LIGHT'] = data['LIGHT'].replace(to_replace=['Dusk', 'Dusk, artificial', 'Dawn', 'Dawn, artificial'], value=1, inplace=False, limit=None, regex=False, method='pad')
data['LIGHT'] = data['LIGHT'].replace(to_replace=['Dark', 'Dark, artificial'], value=2, inplace=False, limit=None, regex=False, method='pad')

# Clean RDSFCOND column (the road surface condition) was setup as an ordinal feature
data['RDSFCOND'] = data['RDSFCOND'].replace(to_replace=['Dry', ' '], value=0, inplace=False, limit=None, regex=False, method='pad')
data['RDSFCOND'] = data['RDSFCOND'].replace(to_replace=['Wet', 'Other', 'Loose Sand or Gravel', 'Loose Snow'], value=1, inplace=False, limit=None, regex=False, method='pad')
data['RDSFCOND'] = data['RDSFCOND'].replace(to_replace=['Slush', 'Ice', 'Packed Snow', 'Spilled liquid'], value=2, inplace=False, limit=None, regex=False, method='pad')


# %%
# Clean INVAGE column (age of involved party) was setup as an ordinal feature
# setup ordinal list, and average filled unknown values
data['INVAGE'] = data['INVAGE'].replace(to_replace=['0 to 4'], value=0, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['5 to 9'], value=1, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['10 to 14'], value=2, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['15 to 19'], value=3, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['20 to 24'], value=4, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['25 to 29'], value=5, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['30 to 34'], value=6, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['35 to 39'], value=7, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['40 to 44'], value=8, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['45 to 49'], value=9, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['50 to 54'], value=10, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['55 to 59'], value=11, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['60 to 64'], value=12, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['65 to 69'], value=13, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['70 to 74'], value=14, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['75 to 79'], value=15, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['80 to 84'], value=16, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['85 to 89'], value=17, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['90 to 94'], value=18, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['Over 95'], value=19, inplace=False, limit=None, regex=False, method='pad')
data['INVAGE'] = data['INVAGE'].replace(to_replace=['unknown'], value=7, inplace=False, limit=None, regex=False, method='pad')


# %%
# Clean INJURY column (severity of injury)
# It appears that the injury code left blank means no injury, or the party is on the police report but indirectly involved in the accident so left blank
data.loc[data['INJURY'] == ' '].head()


# %%
# Injury will be our label, with ' ' values set to no injury
data['INJURY'] = data['INJURY'].replace(to_replace=['None', ' '], value=0, inplace=False, limit=None, regex=False, method='pad')
data['INJURY'] = data['INJURY'].replace(to_replace=['Minimal', 'Minor'], value=1, inplace=False, limit=None, regex=False, method='pad')
data['INJURY'] = data['INJURY'].replace(to_replace=['Major'], value=2, inplace=False, limit=None, regex=False, method='pad')
data['INJURY'] = data['INJURY'].replace(to_replace=['Fatal'], value=3, inplace=False, limit=None, regex=False, method='pad')


# %%
# Clean VEHTYPE column - the type of vehicle involved.
# As we can see, pedestrian collisions have an other classifier very frequently.  However, it's also been tied to vehicle owners making this a difficult feature to clean
# As other is such a large category, I will not be changing this into an ordinal set as we lack the domain knowledge, and instead categorize using get_dummies.  
# So we ended up grouping vehicle classes, leaving 'other' as it's own category, and leaving ' ' as it's own category 
data['VEHTYPE'].value_counts()


# %%
data.loc[data['VEHTYPE'] == 'Other'].head()


# %%
#Grouping small categories together by weight class/vehicle type and setup as an ordinal feature
data['VEHTYPE'] = data['VEHTYPE'].replace(to_replace=[' '], value='NA', inplace=False, limit=None, regex=False, method='pad')
data['VEHTYPE'] = data['VEHTYPE'].replace(to_replace=['Municipal Transit Bus (TTC)', 'Truck - Open', 'Delivery Van', 'Street Car', 'Truck - Dump', 'Truck-Tractor', 'Bus (Other) (Go Bus, Gray Coach)', 'Truck (other)', 'Intercity Bus', 'Truck - Tank', 'School Bus', 'Construction Equipment', 'Truck - Car Carrier', 'Fire Vehicle', 'Other Emergency Vehicle'], value='Heavy Commercial', inplace=False, limit=None, regex=False, method='pad')
data['VEHTYPE'] = data['VEHTYPE'].replace(to_replace=['Off Road - 2 Wheels', 'Moped'], value='Motorcycle', inplace=False, limit=None, regex=False, method='pad')
data['VEHTYPE'] = data['VEHTYPE'].replace(to_replace=['Pick Up Truck', 'Passenger Van', 'Truck - Closed (Blazer, etc)', 'Tow Truck'], value='Large Auto', inplace=False, limit=None, regex=False, method='pad')
data['VEHTYPE'] = data['VEHTYPE'].replace(to_replace=['Taxi', 'Police Vehicle'], value='Automobile, Station Wagon', inplace=False, limit=None, regex=False, method='pad')


# %%
# Clean INVTYPE column - the involvement of the person in the row of the database
# Grouping small categories together by weight class/vehicle type
data['INVTYPE'] = data['INVTYPE'].replace(to_replace=['Moped Driver'], value='Motorcycle Driver', inplace=False, limit=None, regex=False, method='pad')
data['INVTYPE'] = data['INVTYPE'].replace(to_replace=['Motorcycle Passenger'], value='Motorcycle Driver', inplace=False, limit=None, regex=False, method='pad')
data['INVTYPE'] = data['INVTYPE'].replace(to_replace=['Wheelchair', 'In-Line Skater'], value='Pedestrian', inplace=False, limit=None, regex=False, method='pad')
data['INVTYPE'] = data['INVTYPE'].replace(to_replace=[' ', 'Other Property Owner', 'Driver - Not Hit', 'a', 'Runaway - No Driver', 'Unknown - FTR', 'Pedestrian - Not Hit', 'Witness'], value='Other', inplace=False, limit=None, regex=False, method='pad')
data['INVTYPE'] = data['INVTYPE'].replace(to_replace=['Trailer Owner'], value='Vehicle Owner', inplace=False, limit=None, regex=False, method='pad')


# %%
#Ordinal features and our label has now been setup, now we need to one hot encode all our categorical features


# %%
# Clean LOCCORD - Location Coordinates of accident
data['LOCCOORD'] = data['LOCCOORD'].replace(to_replace=[' ', 'Park, Private Property, Public Lane', 'Entrance Ramp Westbound'], value='Other', inplace=False, limit=None, regex=False, method='pad')
data1 = pd.get_dummies(data[['LOCCOORD']])
data = pd.concat([data,data1], axis=1)
data.drop('LOCCOORD', axis=1, inplace=True)

# Clean ACCLOC - the accident location
data['ACCLOC'] = data['ACCLOC'].replace(to_replace=[' '], value='Other', inplace=False, limit=None, regex=False, method='pad')
data1 = pd.get_dummies(data[['ACCLOC']])
data = pd.concat([data,data1], axis=1)
data.drop('ACCLOC', axis=1, inplace=True)

# Clean TRAFFCTL - the type of traffic control present
data1 = pd.get_dummies(data[['TRAFFCTL']])
data = pd.concat([data,data1], axis=1)
data.drop('TRAFFCTL', axis=1, inplace=True)

# Clean IMPACTYPE - the type of impact
data1 = pd.get_dummies(data[['IMPACTYPE']])
data = pd.concat([data,data1], axis=1)
data.drop('IMPACTYPE', axis=1, inplace=True)

# get_dummies for various columns
data1 = pd.get_dummies(data[['INVTYPE']])
data = pd.concat([data,data1], axis=1)
data.drop('INVTYPE', axis=1, inplace=True)

data1 = pd.get_dummies(data[['VEHTYPE']])
data = pd.concat([data,data1], axis=1)
data.drop('VEHTYPE', axis=1, inplace=True)

data1 = pd.get_dummies(data[['MANOEUVER']])
data = pd.concat([data,data1], axis=1)
data.drop('MANOEUVER', axis=1, inplace=True)

data1 = pd.get_dummies(data[['DRIVACT']])
data = pd.concat([data,data1], axis=1)
data.drop('DRIVACT', axis=1, inplace=True)

data1 = pd.get_dummies(data[['DRIVCOND']])
data = pd.concat([data,data1], axis=1)
data.drop('DRIVCOND', axis=1, inplace=True)

data1 = pd.get_dummies(data[['PEDTYPE']])
data = pd.concat([data,data1], axis=1)
data.drop('PEDTYPE', axis=1, inplace=True)

data1 = pd.get_dummies(data[['PEDACT']])
data = pd.concat([data,data1], axis=1)
data.drop('PEDACT', axis=1, inplace=True)

data1 = pd.get_dummies(data[['PEDCOND']])
data = pd.concat([data,data1], axis=1)
data.drop('PEDCOND', axis=1, inplace=True)

data1 = pd.get_dummies(data[['CYCLISTYPE']])
data = pd.concat([data,data1], axis=1)
data.drop('CYCLISTYPE', axis=1, inplace=True)

data1 = pd.get_dummies(data[['CYCACT']])
data = pd.concat([data,data1], axis=1)
data.drop('CYCACT', axis=1, inplace=True)

data1 = pd.get_dummies(data[['CYCCOND']])
data = pd.concat([data,data1], axis=1)
data.drop('CYCCOND', axis=1, inplace=True)


# %%
data.info()


# %%
data.shape

# %% [markdown]
# # Data Preprocessing

# %%
############### PCA ######################################


# %%
from sklearn.decomposition import TruncatedSVD as svd
from sklearn.pipeline import Pipeline


# %%
# Moving injury target to the end of the dataframe
injury = data['INJURY']
d = data.drop(columns=['INJURY'])
data = pd.concat([d, injury], axis=1)


# %%
data.shape


# %%
data.head()


# %%
y = data.iloc[:,-1]
X = data.iloc[:,6:-2]


# %%
X.head()


# %%
from sklearn.linear_model import LogisticRegression

steps = [('svd', svd), ('m', LogisticRegression())]
model = Pipeline(steps=steps)


# %%
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot  


def get_models():
    models = dict()
    for i in range(1,40):
        steps = [('svd', TruncatedSVD(n_components=i)), ('m', LogisticRegression())]
        models[str(i)] = Pipeline(steps=steps)
    return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()


# %%
svd = TruncatedSVD(n_components=34, n_iter=7, random_state=42)
svd.fit(X)
X_svd = svd.transform(X)


# %%
print("original shape:   ", X.shape)
print("transformed shape:", X_svd.shape)


# %%
a = data.iloc[:,:5]
b = pd.DataFrame(X_svd)


# %%
a.head()


# %%
X = pd.concat([a, b], axis=1)


# %%
X.head()

# %% [markdown]
# ## Seletion of Best best features

# %%
feature_names = list(X.columns)


# %%
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

num_feats = 30

cor_list = cor_support = cor_feature = []
for i in X.columns:
  cor = np.corrcoef(X[i], y)[0, 1]
  cor_list.append(cor)

cor_list = [0 if np.isnan(i) else i for i in cor_list] # replace NaN with 0
cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
cor_support = [True if i in cor_feature else False for i in list(X.columns)]

rfe_support = rfe_feature = []
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=15)
rfe_selector.fit(X_train_minmax, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()

embedded_lr_support = embedded_lr_feature = []
lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
embedded_lr_selector = SelectFromModel(lr, max_features=num_feats)
embedded_lr_selector = embedded_lr_selector.fit(X, y)
embedded_lr_support = embedded_lr_selector.get_support()
embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()

embedded_rf_support = embedded_rf_feature = []
sel = SelectFromModel(RandomForestClassifier(n_estimators = 40, bootstrap=False, max_features=num_feats))
sel.fit(X, y)
embedded_rf_feature = X.columns[(sel.get_support())]
embedded_rf_support = sel.get_support()

embedded_lgbm_support = embedded_lgbm_feature = []
lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
        reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
embedded_lgbm_selector.fit(X, y)
embedded_lgbm_support = embedded_lgbm_selector.get_support()
embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()

pd.set_option('display.max_rows', None)
feature_selection_df = pd.DataFrame({
    'Feature':feature_names,
    'Pearson':cor_support,
    'RFE':rfe_support,
    'Logistic Regression':embedded_lr_support,
    'Random Forest':embedded_rf_support,
    'LightGBM':embedded_lgbm_support
})
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)


# %%
feature_selection_df

# %% [markdown]
# # Run Random Forest

# %%
from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels = train_test_split(X, y, 
                                                          stratify = y,
                                                          test_size = 0.3, 
                                                          random_state = RSEED)


# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

parameters = {
    'n_estimators':[100],
    'max_depth':[10],
    'max_features': ['auto', 'sqrt', None],
    'max_leaf_nodes':[28],
    'min_samples_split': [5],
    'bootstrap': [False]
}

model = RandomForestClassifier(n_estimators=100, 
                               random_state=RSEED, 
                               max_features = 'sqrt')

cv = RandomizedSearchCV(
    scoring='roc_auc',
    estimator = model,
    param_distributions=parameters,
    cv=3,
    n_iter=10,
    random_state=RSEED
  )


# %%
X.head()


# %%
clf = RandomForestClassifier(max_depth=10, random_state=RSEED)
clf.fit(train, train_labels)


# %%
predictions = clf.predict(test)


# %%
from sklearn.metrics import confusion_matrix
import itertools

#  Helper function to plot Confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

cm = confusion_matrix(test_labels, predictions)
plot_confusion_matrix(cm, ['NO INJURY', 'MINOR INJURY', 'MAJOR INJURY', 'FATAL'])


# %%
clf.score(test, test_labels)


# %%



