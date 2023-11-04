#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# We will use dataset from here: https://www.kaggle.com/datasets/prathamtripathi/drug-classification/code

data = pd.read_csv('drug200.csv')


# replace target to num
for i, k in enumerate(list(data['Drug'].value_counts().index)):
    data.loc[data['Drug'] == k, ['Drug']] = i
data['Drug']


# split

df_features = data.drop(['Drug'], axis=1)
df_target = data[['Drug']]

feature_full_train, feature_test, target_full_train, target_test = train_test_split(df_features, df_target, random_state=42, test_size=0.2, stratify=df_target)
feature_train, feature_val, target_train, target_val = train_test_split(feature_full_train, target_full_train, random_state=42, test_size=0.25, stratify=target_full_train)


# use oversimpling for kill disbalance

def upsample(features, target):
    repeater = 100 // (target.value_counts() / target.values.sum() * 100) # repeater for evry class
    features_unsampled = pd.DataFrame(columns=features.columns)
    target_unsampled = pd.DataFrame(columns=target.columns)
    for i, r in enumerate(repeater):
        features_unsampled = shuffle(pd.concat([features_unsampled] + [features[target.values == i]] * int(r)), random_state=42)
        target_unsampled = shuffle(pd.concat([target_unsampled] + [target[target.values == i]] * int(r)), random_state=42)
    
    return features_unsampled.reset_index(drop=True), target_unsampled.reset_index(drop=True)

feature_full_train, target_full_train = upsample(feature_full_train, target_full_train)
feature_train, target_train = upsample(feature_train, target_train)
feature_val, target_val = upsample(feature_val, target_val)
feature_test, target_test = upsample(feature_test, target_test)


target_val = target_val.astype(int).values
target_test = target_test.astype(int).values
target_train = target_train.astype(int).values
target_full_train = target_full_train.astype(int).values


# vectorize fitures

def vectorize(df_train, df_val):
    dv = DictVectorizer(sparse=False)

    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)


    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)

    return X_train, X_val, dv

X_full_train, X_test, dv_full = vectorize(feature_full_train, feature_test)

# train

model_rf = RandomForestRegressor(n_estimators=3, max_depth=5, random_state=1, n_jobs=-1)
model_rf.fit(X_full_train, target_full_train)
y_pred_rf = model_rf.predict(X_test).astype(int)

print('Accurancy on test', accuracy_score(target_test, y_pred_rf))

# save model

with open('dv.bin', 'wb') as f:
    pickle.dump(dv_full, f)

with open('model.bin', 'wb') as f:
    pickle.dump(model_rf, f)




