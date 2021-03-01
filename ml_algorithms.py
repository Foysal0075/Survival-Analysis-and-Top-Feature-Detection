# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:39:57 2020
@author: Foysal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, cohen_kappa_score,matthews_corrcoef, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.svm import SVC
from scipy import interp
from pandas_profiling import ProfileReport
from sklearn.utils import shuffle


r_state=75
data = pd.read_csv('processed_covid_data.csv')
df = pd.DataFrame(data)
df = df.drop('Unnamed: 0', axis=1)

"""Creating Profile Report"""
profile = df.profile_report()
profile.to_file("report.html")



features = np.array(df.columns)[0:-1]
target = "survival"


X= df[features]
y=df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=r_state)


"""Wihtout data balancing"""

cv = KFold(n_splits=5, shuffle=True, random_state=r_state)

def calculate_sen_spec(y_actual, y_hat):

    y_actual = np.array(y_actual)
    sensitivity=0
    specificity=0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)

    return sensitivity, specificity

fig = plt.figure(figsize=(10,9))
def calculate_scores_in_cv(model,model_name, X, y):
    score = cross_val_score(model, X, y)
    accuracy_score = score.mean()*100
    std = score.std()*100

    tprs=[]
    mean_fpr = np.linspace(0,1,100)

    aucs=[]
    f1s=[]
    kappas=[]
    sensitivities=[]
    specificities=[]
    mccs=[]

    for (train, test) in cv.split(X,y):
        model.fit(X.iloc[train], y.iloc[train])

        y_score = model.predict(X.iloc[test])
        y_true = y.iloc[test]


        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)

        kappa = cohen_kappa_score(y_true, y_score)
        kappas.append(kappa)

        f1 = f1_score(y_true, y_score)
        f1s.append(f1)

        sen, spec = calculate_sen_spec(y_true, y_score)
        sensitivities.append(sen)
        specificities.append(spec)

        mcc = matthews_corrcoef(y_true, y_score)
        mccs.append(mcc)


    auc, acc, spec, sen, f1, mcc, kappa =np.mean(aucs),accuracy_score, np.mean(specificities),np.mean(sensitivities),np.mean(f1s),np.mean(mccs),np.mean(kappas)
    print(model_name,": " ,auc, acc, spec, sen, f1, mcc, kappa,'\n')



lr_model = LogisticRegression(random_state=r_state)
xgb_model = XGBClassifier()
dt_model = DecisionTreeClassifier(random_state=r_state)
svc_model = SVC(kernel='linear', C = 1.0, probability=True)
knn_model = KNeighborsClassifier(n_neighbors=5)



calculate_scores_in_cv(xgb_model, 'xgboost_model', X,y)
calculate_scores_in_cv(Adboost_model, 'AdaBoostClassifier', X,y)
calculate_scores_in_cv(dt_model, 'DecisionTreeClassifier', X,y)
calculate_scores_in_cv(svc_model, 'SVC', X,y)
calculate_scores_in_cv(lr_model, 'LogisticRegression', X,y)
calculate_scores_in_cv(knn_model, 'KNeighborsClassifier', X,y)


"""With SMOTE data balancing"""

over = SMOTE (sampling_strategy=1, random_state=r_state)
under = RandomUnderSampler(sampling_strategy=1, random_state=r_state)

steps = [('o', over),('u',under)]
pipeline = Pipeline(steps=steps)

sampled_X, sampled_y = pipeline.fit_resample(X,y)


sampled_X, sampled_y = shuffle(sampled_X,  sampled_y, random_state=r_state)

calculate_scores_in_cv(xgb_model, 'xgboost_model', sampled_X, sampled_y)
calculate_scores_in_cv(dt_model, 'DecisionTreeClassifier', sampled_X, sampled_y)
calculate_scores_in_cv(svc_model, 'SVC', sampled_X, sampled_y)
calculate_scores_in_cv(lr_model, 'LogisticRegression', sampled_X, sampled_y)
calculate_scores_in_cv(knn_model, 'KNeighborsClassifier', sampled_X, sampled_y)



fig = plt.figure(figsize=(6,4))
plt.axis([0, 1, 0, 1])
def plot_cv_roc(model, name):
    tprs=[]
    mean_fpr = np.linspace(0,1,100)
    i=1

    for train, test in cv.split(sampled_X, sampled_y):
        prediction = model.fit(sampled_X.iloc[train], sampled_y.iloc[train]).predict(sampled_X.iloc[test])
        fpr, tpr,_ = roc_curve(sampled_y.iloc[test], prediction)
        tprs.append(interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr,lw=2 , label=''+name+' (AUC= %0.4f)' %(mean_auc))

plot_cv_roc(xgb_model, 'XGBoost')
plot_cv_roc(dt_model, 'DT')
plot_cv_roc(svc_model, 'SVC')
plot_cv_roc(lr_model, 'LR')
plot_cv_roc(knn_model, 'KNN')
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color ='#082725')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()








































