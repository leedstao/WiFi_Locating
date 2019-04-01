#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import components
import numpy as np
import pandas as pd
import pickle
import gzip
import copy

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost import XGBRegressor as XGR
from xgboost import XGBClassifier as XGC

#Function for training Decision Tree based models
def Model_Tree_Work(X_train, y_train, model_reg=None, model_clf=None):
    ans=[]
    
    for col in ['LONGITUDE','LATITUDE']:
        model_temp=copy.deepcopy(model_reg)
        model_temp.fit(X_train, y_train[col])
        ans.append(model_temp)
        
    for col in ['FLOOR','BUILDINGID']:
        model_temp=copy.deepcopy(model_clf)
        model_temp.fit(X_train, y_train[col])
        ans.append(model_temp)
        
    return ans

#Other functions used later
def Rename_Col(df, text):
    for i in df.columns:
        df=df.rename(index=str, columns={i: i+text})
        
    return df

#setup the parameters
limiter=0
n_neighbors=1
p=1
test_size=0.25
max_depth_DT=100
n_estimators=200
max_depth=None
learning_rate=0.2

print('Loading Training Data.')

#load, clean the training data and feature engineering.
df=pd.read_csv('trainingData.csv')
df=pd.get_dummies(df, columns=['BUILDINGID'], prefix='BUILDING')
df['BUILDINGID']=df[['BUILDING_0','BUILDING_1','BUILDING_2']].idxmax(axis=1)
df['BUILDINGID']=df['BUILDINGID'].apply(lambda x: x[-1])

X_train=df.drop(df.columns[520:],axis=1)
for col in X_train.columns:
    X_train[col]=X_train[col].apply(lambda x:limiter if x==100 else float(x)/100+1)

y_train=df[['LONGITUDE','LATITUDE','FLOOR','BUILDING_0','BUILDING_1','BUILDING_2','BUILDINGID']]

print('Building KNN Model.')

#Building models and save
#K-Nearest-Neighbors Model
model_KNN=KNR(n_neighbors=n_neighbors, weights='uniform',p=p)
model_KNN.fit(X_train, y_train[['LONGITUDE','LATITUDE','FLOOR','BUILDING_0','BUILDING_1','BUILDING_2']])
with gzip.open(pickle_name, 'wb') as f:
    pickle.dump([model_KNN], f, protocol=-1)

print('Building Random Forest Model.')

#Random Forest Model
model_RF= Model_Tree_Work(X_train, y_train, 
                model_reg=RFR(n_estimators=n_estimators, max_depth=max_depth), 
                model_clf=RFC(n_estimators=n_estimators, max_depth=max_depth))
with gzip.open(pickle_name, 'ab') as f:
    pickle.dump(model_RF, f, protocol=-1)

print('Building Gradient Boost Model.')

#Gradient Boost Model
model_GB= Model_Tree_Work(X_train, y_train, 
                model_reg=GBR(n_estimators=n_estimators, learning_rate=learning_rate), 
                model_clf=GBC(n_estimators=n_estimators, learning_rate=learning_rate))
with gzip.open(pickle_name, 'ab') as f:
    pickle.dump(model_GB, f, protocol=-1)

print('Building XGBoost Model.')

#XGBoost Model
model_XG= Model_Tree_Work(X_train, y_train, 
                model_reg=XGR(n_estimators=n_estimators, learning_rate=learning_rate), 
                model_clf=XGC(n_estimators=n_estimators, learning_rate=learning_rate))
with gzip.open(pickle_name, 'ab') as f:
    pickle.dump(model_XG, f, protocol=-1)

#The final ensemble model

print('Calculating Base Model Prediction.')
#Calculating the predictions from previous models
X_ensemble=X_train

#KNN, recover from dummie variables
y_KNN=model_KNN.predict(X_train)
y_KNN=pd.DataFrame(y_KNN, columns=['LONGITUDE','LATITUDE','FLOOR','BUILDING_0','BUILDING_1','BUILDING_2'])
y_KNN['FLOOR']=y_KNN['FLOOR'].astype(int)
y_KNN['BUILDINGID']=y_KNN[['BUILDING_0','BUILDING_1','BUILDING_2']].idxmax(axis=1)
y_KNN['BUILDINGID']=y_KNN['BUILDINGID'].apply(lambda x: x[-1])
y_KNN=y_KNN.drop(['BUILDING_0','BUILDING_1','BUILDING_2'], axis=1)
y_KNN=Rename_Col(y_KNN, '_KNN')
X_ensemble=X_ensemble.join(y_KNN.reset_index()[y_KNN.columns])

cols=['LONGITUDE','LATITUDE','FLOOR','BUILDINGID']

#Random Forest
y_temp=pd.DataFrame()
for i, col in enumerate(cols):
    y_temp[col] = model_RF[i].predict(X_train)
y_temp=Rename_Col(y_temp, '_RF')
X_ensemble=X_ensemble.join(y_temp.reset_index()[y_temp.columns])

#Gradient Boost
y_temp=pd.DataFrame()
for i, col in enumerate(cols):
    y_temp[col] = model_GB[i].predict(X_train)
y_temp=Rename_Col(y_temp, '_GB')
X_ensemble=X_ensemble.join(y_temp.reset_index()[y_temp.columns])

#XGBoost
y_temp=pd.DataFrame()
for i, col in enumerate(cols):
    y_temp[col] = model_XG[i].predict(X_train)
y_temp=Rename_Col(y_temp, '_XG')
X_ensemble=X_ensemble.join(y_temp.reset_index()[y_temp.columns])

print('Building Ensemble Model.')

#Ensembling the models
model_EN= Model_Tree_Work(X_ensemble, y_train, 
                model_reg=RFR(n_estimators=n_estimators, max_depth=max_depth), 
                model_clf=RFC(n_estimators=n_estimators, max_depth=max_depth))
with gzip.open(pickle_name, 'ab') as f:
    pickle.dump(model_EN, f, protocol=-1)

