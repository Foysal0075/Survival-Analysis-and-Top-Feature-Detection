# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 07:17:30 2020

@author: Foysal


"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from imblearn.pipeline import Pipeline
import lightgbm as lgb





covid_data = pd.read_csv('processed_covid_data.csv')
df = pd.DataFrame(covid_data)
df = df.drop('Unnamed: 0', axis=1)

features = np.array(df.columns[:-1])
target = 'survival'

group1 = np.array(df.query('Age<=34').index)
group2 =np.array(df.query('Age>34 and Age<=47').index)
group3 =np.array(df.query('Age>47 and Age<=64').index)
group4 =np.array(df.query('Age>64').index)



groups = ['Age: 0-34', 'Age: 35-47', 'Age: 48-64', 'Age: 65+']
r_state=75

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=r_state )




def lr_shap_values(X_train, y_train, X_test):

    model = LogisticRegression()
    model.fit(X_train, y_train)
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation = "interventional")
    shap_values_lr = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values_lr, X_test)
    # shap.summary_plot(shap_values_lr, X_test, plot_type ='bar')
    shap_sum = np.abs(shap_values_lr).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    print(importance_df)



def dt_shap_values(X_train, y_train, X_test):

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values_dt = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values_dt[0], X_test)
    # shap.summary_plot(shap_values_dt[0], X_test, plot_type ='bar')
    shap_sum = np.abs(shap_values_dt[0]).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    print(importance_df)



def knn_shap_values(X_train, y_train, X_test):

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    knn_shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(knn_shap_values[0], X_test)
    # shap.summary_plot(knn_shap_values[0], X_test, plot_type ='bar')
    shap_sum = np.abs(knn_shap_values[0]).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    print(importance_df)



def svc_shap_values(X_train, y_train, X_test):

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    svc_shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(svc_shap_values[0], X_test)
    # shap.summary_plot(svc_shap_values[0], X_test, plot_type ='bar')
    shap_sum = np.abs(svc_shap_values[0]).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    print(importance_df)



def xgb_shap_values(X_train, y_train, X_test):

    model = XGBClassifier()
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    shap_values_xgb = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values_xgb, X_test)
    # shap.summary_plot(shap_values_xgb, X_test, plot_type ='bar')
    # shap.dependence_plot("Age", shap_values_xgb, X_test, interaction_index = 'rate Po2')
    shap_sum = np.abs(shap_values_xgb).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    print(importance_df)


def lgb_shap_values(X_train, X_test, y_train, y_test):

    train_cols = X_train.columns.tolist()
    train_data = lgb.Dataset(X_train, label= y_train, feature_name=train_cols)
    test_data = lgb.Dataset(X_test, label = y_test, feature_name=train_cols, reference = train_data)

    params = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary',
          'num_leaves': 2000,
          'min_data_in_leaf': 20,
          'max_bin': 200,
          'max_depth': 16,
          'seed': 2018,
          'nthread': 10,}

    lgb_model = lgb.train(params, train_data,
                          num_boost_round=1000,
                          valid_sets=(test_data,),
                          valid_names=('valid',),
                          verbose_eval=25,
                          early_stopping_rounds=20)

    X_importance = X_test

    explainer = shap.TreeExplainer(lgb_model)
    shap_values_lgb = explainer.shap_values(X_importance)
    # shap.summary_plot(shap_values_lgb[1], X_importance)
    # shap.summary_plot(shap_values_lgb[1], X_importance, plot_type ='bar')
    shap_sum = np.abs(shap_values_lgb[1]).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    print(importance_df)

lr_shap_values(X_train, y_train,  X_test)
lgb_shap_values(X_train, X_test, y_train, y_test)
xgb_shap_values(X_train, y_train,  X_test)
svc_shap_values(X_train, y_train,  X_test)
knn_shap_values(X_train, y_train,  X_test)
dt_shap_values(X_train, y_train,  X_test)

def shap_groups(get_shap):

    for value in groups:
        if value== 'Age: 0-34':
            X_train, X_test, y_train, y_test = train_test_split(df.loc[group1][features], df.loc[group1][target], test_size=0.2, random_state=r_state )
            print('\nAge Group :', value)
            get_shap(X_train, y_train, X_test)
    
        if value== 'Age: 35-47':
    
            X_train, X_test, y_train, y_test = train_test_split(df.loc[group2][features], df.loc[group2][target], test_size=0.2, random_state=r_state )
            print('\nAge Group :', value)
            get_shap(X_train, y_train, X_test)
    
        if value== 'Age: 48-64':
    
            X_train, X_test, y_train, y_test = train_test_split(df.loc[group3][features], df.loc[group3][target], test_size=0.2, random_state=r_state )
            print('\nAge Group :', value)
            get_shap(X_train, y_train, X_test)
    
        if value== 'Age: 65+':
    
            X_train, X_test, y_train, y_test = train_test_split(df.loc[group4][features], df.loc[group4][target], test_size=0.2, random_state=r_state )
            print('\nAge Group :', value)
            get_shap(X_train, y_train, X_test)

shap_groups(lr_shap_values)
shap_groups(xgb_shap_values)
shap_groups(dt_shap_values)
shap_groups(knn_shap_values)
shap_groups(svc_shap_values)








