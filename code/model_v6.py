#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:58:37 2019

@author: lindaxju
"""
#%%
import pickle
import csv
from datetime import datetime

import pandas as pd
import numpy as np
from collections import defaultdict

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
#%%
with open('data/df_all_2019-10-23-21-38-02.pickle','rb') as read_file:
    df_orig = pickle.load(read_file)
#%%
df_orig.shape
#%%
df_orig.head()
#%%
list(df_orig.columns)
#%%
df_orig.info()
#%%
###############################################################################
###############################Choosing Features###############################
###############################################################################
#%%
# more could be: BorrowerState, EmploymentStatus, EmploymentStatusDuration,
# FirstRecordedCreditLine, ListingCategory_num, and IncomeRange
X_cols_list = ['Term','BorrowerAPR','BorrowerRate','ProsperRating_num',
               'ProsperScore','IsBorrowerHomeowner','CreditScoreRangeLower',
               'CreditScoreRangeUpper','CurrentCreditLines','OpenCreditLines',
               'TotalCreditLinespast7years','OpenRevolvingAccounts',
               'OpenRevolvingMonthlyPayment','InquiriesLast6Months','TotalInquiries',
               'CurrentDelinquencies','AmountDelinquent','DelinquenciesLast7Years',
               'PublicRecordsLast10Years','PublicRecordsLast12Months',
               'RevolvingCreditBalance','BankcardUtilization','AvailableBankcardCredit',
               'TotalTrades','TradesNeverDelinquent_perc','TradesOpenedLast6Months',
               'DebtToIncomeRatio','StatedMonthlyIncome','Recommendations']
#%%
X_orig = df_orig[X_cols_list]
X_orig.info()
X_orig.describe()
#%%
# visualize distributions
X_orig_nobool = X_orig.drop(columns=['IsBorrowerHomeowner'])
X_orig_nobool.hist(bins=30,figsize=(20,20));
#%%
pd.Series(X_orig['IsBorrowerHomeowner']).value_counts().plot('bar',figsize=(4,3),title='IsBorrowerHomeowner',grid=True,rot=360)
#%%
###############################################################################
###############################Fixing  Imbalance###############################
###############################################################################
#%%
y_orig = df_orig['LoanStatus_target']
y_orig.describe()
print(Counter(y_orig))
pd.Series(y_orig).value_counts().plot('bar',figsize=(4,3),title='LoanStatus (Full Dataset)',rot=360)
#%%
# prepare features and target
X_train_val_orig, X_test, y_train_val_orig, y_test = train_test_split(X_orig, y_orig, test_size=0.2,random_state=42)
X_train_orig, X_val, y_train_orig, y_val = train_test_split(X_train_val_orig, y_train_val_orig, test_size=0.2,random_state=42)
#%%
# check imbalance
print("Train-Val {}".format(Counter(y_train_val_orig)))
print("Train {}".format(Counter(y_train_orig)))
print("Val {}".format(Counter(y_val)))
print("Test {}".format(Counter(y_test)))
#%%
# Now add some random oversampling of the minority classes
# Object to over-sample the minority class(es) by picking samples at random
# with replacement.
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_sample(X_train_orig,y_train_orig)
print(Counter(y_train))
pd.Series(y_train).value_counts().plot('bar')
#%%
print("Train {}".format(Counter(y_train)))
print("Val {}".format(Counter(y_val)))
print("Test {}".format(Counter(y_test)))
#%%
pd.Series(y_train).value_counts().plot('bar',figsize=(4,3),title='LoanStatus (Train Dataset)',rot=360)
#%%
pd.Series(y_val).value_counts().plot('bar',figsize=(4,3),title='LoanStatus (Validation Dataset)',rot=360)
#%%
pd.Series(y_test).value_counts().plot('bar',figsize=(4,3),title='LoanStatus (Test Dataset)',rot=360)
#%%
###############################################################################
################################Modeling Set Up################################
###############################################################################
#%%
# Standard scale
# Scale the predictors on train, validation, and test sets
# X_tr, y_train
# X_v, y_val
# X_te, y_test
std = StandardScaler()
std.fit(X_train)
X_tr = std.transform(X_train)
X_v = std.transform(X_val)
X_te = std.transform(X_test)
#%%
# set default threshold
threshold_default = 0.5
#%%
# define return metric
def get_df_return(y_val,y_pred,df_BorrowerRate,invest_amount=25):
    df_return = df_BorrowerRate.copy()
    df_return['y_actual'] = y_val
    df_return['y_pred'] = y_pred
    df_return['step1'] = np.where(df_return['y_actual']==0,0,invest_amount*(1+df_return['BorrowerRate']))
    df_return['return_dollar'] = df_return['y_pred']*df_return['step1']
    return df_return
#%%
# define return metric
def get_invest_amt(df_return,invest_amount=25):
    return sum(df_return['y_pred'])*invest_amount
#%%
# define return metric
def get_return_dollar(df_return,invest_amount=25):
    return sum(df_return['return_dollar'])
#%%
# define return metric
def get_return_perc(return_dict):
    return 100*(return_dict['return_dollar']/return_dict['invest_amt']-1)
#%%
# define return metric
def get_return_dict(y_val,y_pred,df_BorrowerRate,invest_amount=25):
    return_dict = dict(invest_amt=0,return_dollar=0,return_perc=0)
    
    df_return = get_df_return(y_val,y_pred,df_BorrowerRate,invest_amount)
    
    return_dict['invest_amt'] = round(get_invest_amt(df_return),2)
    return_dict['return_dollar'] = round(get_return_dollar(df_return),2)
    return_dict['return_perc'] = round(get_return_perc(return_dict),2)
    
    return return_dict
#%%
# extract validation borrow rates
df_X_val = pd.DataFrame(X_val, columns=X_orig.columns)
df_BorrowerRate_val = df_X_val.drop(df_X_val.columns.difference(['BorrowerRate']),1)
# extract test borrow rates
df_X_test = pd.DataFrame(X_test, columns=X_orig.columns)
df_BorrowerRate_test = df_X_test.drop(df_X_test.columns.difference(['BorrowerRate']),1)
#%%
#################################Random Forest#################################
#%%
# tune parameter [n_estimators]
def tune_randomforest_n_estimators(range_n_estimators):
    return_dict = defaultdict(int)
    
    for n in range_n_estimators:
        randomforest = RandomForestClassifier(n_estimators=n,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)
        y_pred = (randomforest.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['n: '+str(n)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(n)
        
    return return_dict
#%%
#tune_randomforest_n_estimators(range(50,300,50))
# result: 100
n_estimators_rf = 100
#%%
# tune parameter [max_depth]
def tune_randomforest_max_depth(range_max_depth):
    return_dict = defaultdict(int)
    
    for depth in range_max_depth:
        randomforest = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=depth,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)
        y_pred = (randomforest.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['depth: '+str(depth)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(depth)
        
    return return_dict
#%%
#tune_randomforest_max_depth(range(1,11,1))
# result: 6
max_depth_rf = 6
#%%
# tune parameter [prob_threshold]
def tune_randomforest_threshold(range_threshold):
    return_dict = defaultdict(int)
    
    for thresh in range_threshold:
        randomforest = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=max_depth_rf,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)
        y_pred = (randomforest.predict_proba(X_v)[:,1]>thresh/100)*1
        return_dict['thresh: '+str(thresh/100)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(thresh/100)
        
    return return_dict
#%%
#tune_randomforest_threshold(range(90,97,1))
# result: 95%
threshold_rf = 0.95
#%%
# fit model
randomforest = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=max_depth_rf,oob_score=True,random_state=42)
randomforest.fit(X_tr, y_train)
#%%
y_pred = (randomforest.predict_proba(X_v)[:,1]>threshold_rf)*1
get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
print("Random Forest validation confusion matrix with threshold: \n", confusion_matrix(y_val, randomforest.predict_proba(X_v)[:,1]>threshold_rf))
#%%
# feature importance
feature_importances_randomforest = pd.DataFrame(randomforest.feature_importances_,
                                                index = X_orig.columns,
                                                columns=['importance']).sort_values('importance',
                                                        ascending=False)
feature_importances_randomforest[:10]
#%%
# feature importance plot
features = X_orig.columns
importances = randomforest.feature_importances_
indices = np.argsort(importances)[-10:]

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#%%
####################################XGBoost####################################
#%%
# tune parameter [max_depth]
def tune_XGBoost_max_depth(range_max_depth):
    return_dict = defaultdict(int)
    
    for depth in range_max_depth:
        XGBoost = XGBClassifier(max_depth=depth,random_state=42)
        XGBoost.fit(X_tr, y_train)
        y_pred = (XGBoost.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['depth: '+str(depth)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(depth)
        
    return return_dict
#%%
#tune_XGBoost_max_depth(range(1,11,1))
# result: 5
max_depth_xgb = 5
#%%
# tune parameter [n_estimators]
def tune_XGBoost_n_estimators(range_n_estimators):
    return_dict = defaultdict(int)
    
    for n in range_n_estimators:
        XGBoost = XGBClassifier(max_depth=max_depth_xgb,n_estimators=n,random_state=42)
        XGBoost.fit(X_tr, y_train)
        y_pred = (XGBoost.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['n: '+str(n)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(n)
        
    return return_dict
#%%
#tune_XGBoost_n_estimators(range(50,300,50))
# result: 100
n_estimators_xgb = 100
#%%
# tune parameter [prob_threshold]
def tune_XGBoost_threshold(range_threshold):
    return_dict = defaultdict(int)
    
    for thresh in range_threshold:
        XGBoost = XGBClassifier(max_depth=max_depth_xgb,n_estimators=n_estimators_xgb,random_state=42)
        XGBoost.fit(X_tr, y_train)
        y_pred = (XGBoost.predict_proba(X_v)[:,1]>thresh/100)*1
        return_dict['thresh: '+str(thresh/100)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(thresh/100)
        
    return return_dict
#%%
#tune_XGBoost_threshold(range(90,100,1))
# result: 97%
threshold_xgb = 0.97
#%%
X_tr_df = pd.DataFrame(X_tr, columns=X_orig.columns)
X_v_df = pd.DataFrame(X_v, columns=X_orig.columns)
#%%
# fit model
XGBoost = XGBClassifier(max_depth=max_depth_xgb,n_estimators=n_estimators_xgb,random_state=42)
XGBoost.fit(X_tr_df, y_train)
#%%
y_pred = (XGBoost.predict_proba(X_v_df)[:,1]>threshold_xgb)*1
get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
print("XGBoost validation confusion matrix with threshold: \n", confusion_matrix(y_val, XGBoost.predict_proba(X_v_df)[:,1]>threshold_xgb))
#%%
# feature importance
feature_importances_XGBoost = pd.DataFrame(XGBoost.feature_importances_,
                                           index=X_orig.columns,
                                           columns=['importance']).sort_values('importance',
                                                   ascending=False)
feature_importances_XGBoost[:10]
#%%
# feature importance plot
plot_importance(XGBoost,importance_type='gain',max_num_features=10,show_values=False)
plt.show()
#%%
######################################KNN######################################
#%%
# tune parameter [n_neighbors]
def tune_KNN_n_neighbors(range_n_neighbors):
    return_dict = defaultdict(int)
    
    for n in range_n_neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_tr, y_train)
        y_pred_ = (knn.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['n: '+str(n)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(n)
        
    return return_dict
#%%
#tune_KNN_n_neighbors(range(46,65,1))
# result: 54
n_neighbors_knn = 54
#%%
# tune parameter [prob_threshold]
def tune_KNN_threshold(range_threshold):
    return_dict = defaultdict(int)
    
    for thresh in range_threshold:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
        knn.fit(X_tr, y_train)
        y_pred = (knn.predict_proba(X_v)[:,1]>thresh/100)*1
        return_dict['thresh: '+str(thresh)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(thresh)
        
    return return_dict
#%%
#tune_KNN_threshold(range(90,95,1))
# result: 93%
threshold_KNN = 0.93
#%%
# fit model
knn = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
knn.fit(X_tr, y_train)
#%%
y_pred = (knn.predict_proba(X_v)[:,1]>threshold_KNN)*1
get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
print("KNN validation confusion matrix with threshold: \n", confusion_matrix(y_val, knn.predict_proba(X_v)[:,1]>threshold_KNN))
#%%
##############################Logistic Regression##############################
#%%
# tune parameter [C]
def tune_LogReg_C(range_C):
    return_dict = defaultdict(int)
    
    for c in range_C:
        logreg = LogisticRegression(C=c)
        logreg.fit(X_tr,y_train)
        y_pred = (logreg.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['c: '+str(c)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(c)
        
    return return_dict
#%%
#tune_LogReg_C(range(1,20,1))
# result: 15
C_LogReg = 15
#%%
# tune parameter [prob_threshold]
def tune_LogReg_threshold(range_threshold):
    return_dict = defaultdict(int)
    
    for thresh in range_threshold:
        logreg = LogisticRegression(C=C_LogReg)
        logreg.fit(X_tr,y_train)
        y_pred = (logreg.predict_proba(X_v)[:,1]>thresh/100)*1
        return_dict['thresh: '+str(thresh)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(thresh)
        
    return return_dict
#%%
#tune_LogReg_threshold(range(95,100,1))
# result: 99%
threshold_LogReg = 0.99
#%%
# fit model
logreg = LogisticRegression(C=C_LogReg)
logreg.fit(X_tr,y_train)
#%%
y_pred = (logreg.predict_proba(X_v)[:,1]>threshold_LogReg)*1
get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
print("Logistic Regression validation confusion matrix with threshold: \n", confusion_matrix(y_val, logreg.predict_proba(X_v)[:,1]>threshold_LogReg))
#%%
##################################Naive Bayes##################################
#%%
# tune parameter [prob_threshold]
def tune_NB_threshold(range_threshold):
    return_dict = defaultdict(int)
    
    for thresh in range_threshold:
        nb = GaussianNB()
        nb.fit(X_tr, y_train)
        y_pred = (nb.predict_proba(X_v)[:,1]>thresh/100)*1
        return_dict['thresh: '+str(thresh)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(thresh)
        
    return return_dict
#%%
#tune_NB_threshold(range(90,100,1))
# result: 99%
threshold_NB = 0.99
#%%
# fit model
nb = GaussianNB()
nb.fit(X_tr, y_train)
#%%
y_pred = (nb.predict_proba(X_v)[:,1]>threshold_NB)*1
get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
print("Naive Bayes validation confusion matrix: \n", confusion_matrix(y_val, nb.predict_proba(X_v)[:,1]>threshold_NB))
#%%
######################################SVM######################################
#%%
# fit model
#svm_model = svm.SVC(C=3, kernel="linear")
#svm_model = svm.SVC(C=3, kernel="linear", probability=True) # takes a long time
#svm_model.fit(X_tr, y_train)
#%%
#y_pred = svm_model.predict(X_v)
#get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
#print("SVM validation confusion matrix: \n", confusion_matrix(y_val, svm_model.predict(X_v)))
#%%
##################################ROC Curves###################################
#%%
# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'XGBoost',
    'model': XGBoost,
},
{
    'label': 'Random Forest',
    'model': randomforest,
},
{
    'label': 'Logistic Regression',
    'model': logreg,
},
{
    'label': 'KNN',
    'model': knn,
},
{ 
    'label': 'Naive Bayes',
    'model': nb,
}
]
#%%
def plot_roc(models,x_train,y_train,x_test,y_test):
    plt.figure(figsize=(20,15))
    # Below for loop iterates through your models list
    for m in models:
        model = m['model'] # select the model
        model.fit(x_train, y_train) # train the model
    # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(y_test,model.predict_proba(x_test)[:,1])
    # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc), linewidth=7.0)
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--',linewidth=7.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
#    plt.title('ROC curve for on time loan completion')
    plt.legend(loc="lower right")
    
    SMALL_SIZE = 30
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 30
    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.show()   # Display
#%%
plot_roc(models,X_tr,y_train,X_v,y_val)
#%%
#############################Output Final Results##############################
#%%#%%
print("Random Forest Threshold: {}".format(threshold_rf))
print("XGBoost Threshold: {}".format(threshold_xgb))
print("KNN Threshold: {}".format(threshold_KNN))
print("Logistic Regression Threshold: {}".format(threshold_LogReg))
print("Naive Bayes Threshold: {}".format(threshold_NB))
#%%
#%%
y_pred_val_rf = (randomforest.predict_proba(X_v)[:,1]>threshold_rf)*1
print("Random Forest Validation Returns: {}".format(get_return_dict(y_val,y_pred_val_rf,df_BorrowerRate_val)))
y_pred_test_rf = (randomforest.predict_proba(X_te)[:,1]>threshold_rf)*1
print("Random Forest Test Returns: {}".format(get_return_dict(y_test,y_pred_test_rf,df_BorrowerRate_test)))
df_return_test_rf = get_df_return(y_test,y_pred_test_rf,df_BorrowerRate_test)
#%%
X_v_df = pd.DataFrame(X_v, columns=X_orig.columns)
X_te_df = pd.DataFrame(X_te, columns=X_orig.columns)
XGBoost = XGBClassifier(max_depth=max_depth_xgb,n_estimators=n_estimators_xgb,random_state=42)
XGBoost.fit(X_tr_df, y_train)
y_pred_val_xgb = (XGBoost.predict_proba(X_v_df)[:,1]>threshold_xgb)*1
print("XGBoost Validation Returns: {}".format(get_return_dict(y_val,y_pred_val_xgb,df_BorrowerRate_val)))
y_pred_test_xgb = (XGBoost.predict_proba(X_te_df)[:,1]>threshold_xgb)*1
print("XGBoost Test Returns: {}".format(get_return_dict(y_test,y_pred_test_xgb,df_BorrowerRate_test)))
df_return_test_xgb = get_df_return(y_test,y_pred_test_xgb,df_BorrowerRate_test)
#%%
y_pred_val_knn = (knn.predict_proba(X_v)[:,1]>threshold_KNN)*1
print("KNN Validation Returns: {}".format(get_return_dict(y_val,y_pred_val_knn,df_BorrowerRate_val)))
y_pred_test_knn = (knn.predict_proba(X_te)[:,1]>threshold_KNN)*1
print("KNN Test Returns: {}".format(get_return_dict(y_test,y_pred_test_knn,df_BorrowerRate_test)))
df_return_test_knn = get_df_return(y_test,y_pred_test_knn,df_BorrowerRate_test)
#%%
y_pred_val_logreg = (logreg.predict_proba(X_v)[:,1]>threshold_LogReg)*1
print("Logistic Regression Validation Returns: {}".format(get_return_dict(y_val,y_pred_val_logreg,df_BorrowerRate_val)))
y_pred_test_logreg = (logreg.predict_proba(X_te)[:,1]>threshold_LogReg)*1
print("Logistic Regression Test Returns: {}".format(get_return_dict(y_test,y_pred_test_logreg,df_BorrowerRate_test)))
df_return_test_logreg = get_df_return(y_test,y_pred_test_logreg,df_BorrowerRate_test)
#%%
y_pred_val_NB = (nb.predict_proba(X_v)[:,1]>threshold_NB)*1
print("Naive Bayes Validation Returns: {}".format(get_return_dict(y_val,y_pred_val_NB,df_BorrowerRate_val)))
y_pred_test_NB = (nb.predict_proba(X_te)[:,1]>threshold_NB)*1
print("Naive Bayes Test Returns: {}".format(get_return_dict(y_test,y_pred_test_NB,df_BorrowerRate_test)))
df_return_test_NB = get_df_return(y_test,y_pred_test_NB,df_BorrowerRate_test)
#%%
#%%
print("Random Forest Test Returns: {}".format(get_return_dict(y_test,y_pred_test_rf,df_BorrowerRate_test)))
print("XGBoost Test Returns: {}".format(get_return_dict(y_test,y_pred_test_xgb,df_BorrowerRate_test)))
print("KNN Test Returns: {}".format(get_return_dict(y_test,y_pred_test_knn,df_BorrowerRate_test)))
print("Logistic Regression Test Returns: {}".format(get_return_dict(y_test,y_pred_test_logreg,df_BorrowerRate_test)))
print("Naive Bayes Test Returns: {}".format(get_return_dict(y_test,y_pred_test_NB,df_BorrowerRate_test)))
#%%
print("Random Forest test confusion matrix: \n", confusion_matrix(y_test,y_pred_test_rf))
print("XGBoost test confusion matrix: \n", confusion_matrix(y_test,y_pred_test_xgb))
print("KNN test confusion matrix: \n", confusion_matrix(y_test,y_pred_test_knn))
print("Logistic Regression test confusion matrix: \n", confusion_matrix(y_test,y_pred_test_logreg))
print("Naive Bayes test confusion matrix: \n", confusion_matrix(y_test,y_pred_test_NB))
##%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_return_test_rf_'+timestamp
#df_return_test_rf.to_csv(r'data/generated/'+filename+'.csv')
##%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_return_test_xgb_'+timestamp
#df_return_test_xgb.to_csv(r'data/generated/'+filename+'.csv')
##%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_return_test_knn_'+timestamp
#df_return_test_knn.to_csv(r'data/generated/'+filename+'.csv')
##%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_return_test_logreg_'+timestamp
#df_return_test_logreg.to_csv(r'data/generated/'+filename+'.csv')
##%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_return_test_NB_'+timestamp
#df_return_test_NB.to_csv(r'data/generated/'+filename+'.csv')
##%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_orig_'+timestamp
#df_orig.to_csv(r'data/generated/'+filename+'.csv')
#%%
###############################################################################
##################################Presentation#################################
###############################################################################
#%%
y_pred_invest_all = np.asarray([1]*len(y_test))
print("Randomly Investing Test Returns: {}".format(get_return_dict(y_test,y_pred_invest_all,df_BorrowerRate_test)))
#%%
rf_loancharact_df = pd.read_csv('data/generated/df_return_test_rf_characteristics.csv')
#%%
rf_loancharact_df.info()
#%%
# visualize distributions
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
rf_loancharact_cat = ['LoanID','BorrowerState','EmploymentStatus','IsBorrowerHomeowner','ActualLoanStatus']
rf_loancharact_df_nocat = rf_loancharact_df.drop(columns=rf_loancharact_cat)
rf_loancharact_df_nocat.hist(bins=30,figsize=(10,8));
#%%
pd.Series(rf_loancharact_df['BorrowerState']).value_counts().plot('bar',figsize=(8,3),title='BorrowerState',grid=True,rot=360)
#%%
pd.Series(rf_loancharact_df['EmploymentStatus']).value_counts().plot('bar',figsize=(4,3),title='EmploymentStatus',grid=True,rot=360)
#%%
pd.Series(rf_loancharact_df['IsBorrowerHomeowner']).value_counts().plot('bar',figsize=(4,3),title='IsBorrowerHomeowner',grid=True,rot=360)
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_return_'+timestamp
#df_return.to_csv(r'data/'+filename+'.csv')
#%%
#scores = defaultdict(int)
#
#for c in range(100,1100,100):
#    logreg = LogisticRegression(C=c)
#    logreg.fit(X_tr,y_train)
#    scores[c] = logreg.score(X_v,y_val)
#    
#scores
