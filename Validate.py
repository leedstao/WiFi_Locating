#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import gzip

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost import XGBRegressor as XGR
from xgboost import XGBClassifier as XGC

def Rename_Col(df, text):
    for i in df.columns:
        df=df.rename(index=str, columns={i: i+text})
        
    return df

filename=input('Enter the Validation Data File Name:')

limiter=0

try:
    df=pd.read_csv(filename)
except:
    print('File not found')
    exit()

df=pd.get_dummies(df, columns=['BUILDINGID'], prefix='BUILDING')
df['BUILDINGID']=df[['BUILDING_0','BUILDING_1','BUILDING_2']].idxmax(axis=1)
df['BUILDINGID']=df['BUILDINGID'].apply(lambda x: x[-1])

X_val=df.drop(df.columns[520:],axis=1)

for col in X_val.columns:
    X_val[col]=X_val[col].apply(lambda x:limiter if x==100 else float(x)/100+1)

X_ensemble=X_val

#Load pickled models
with gzip.open('Trained_models.pkl','rb') as f:
    model_KNN=pickle.load(f)[0]
    model_RF=pickle.load(f)
    model_GB=pickle.load(f)
    model_XG=pickle.load(f)
    model_EN=pickle.load(f)
    

#KNN, recover from dummie variables
y_KNN=model_KNN.predict(X_val)
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
    y_temp[col] = model_RF[i].predict(X_val)
y_temp=Rename_Col(y_temp, '_RF')
X_ensemble=X_ensemble.join(y_temp.reset_index()[y_temp.columns])

#Gradient Boost
y_temp=pd.DataFrame()
for i, col in enumerate(cols):
    y_temp[col] = model_GB[i].predict(X_val)
y_temp=Rename_Col(y_temp, '_GB')
X_ensemble=X_ensemble.join(y_temp.reset_index()[y_temp.columns])

#XGBoost
y_temp=pd.DataFrame()
for i, col in enumerate(cols):
    y_temp[col] = model_XG[i].predict(X_val)
y_temp=Rename_Col(y_temp, '_XG')
X_ensemble=X_ensemble.join(y_temp.reset_index()[y_temp.columns])

#Final prediction
y_output=pd.DataFrame()
for i, col in enumerate(cols):
    y_output[col] = model_EN[i].predict(X_ensemble)

y_output[cols].to_csv('ValidationPrediction.csv', index=False)

