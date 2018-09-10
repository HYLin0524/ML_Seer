# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:33:36 2018

@author: HIM_LAB
"""

import pandas as pd
import numpy as np
import random
from itertools import *

#preprocessing
from sklearn import preprocessing

#imbalance data
from imblearn.combine import SMOTEENN,SMOTETomek 


#data split
from sklearn.model_selection import train_test_split

#feature selection
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import SelectFromModel, RFECV, RFE

#ensamble
from mlens.ensemble import SuperLearner 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
#single classifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
#Time
import time

tStart = time.time()
pd.options.mode.chained_assignment = None 

colon = pd.read_csv('For_Model.csv', delimiter=',', low_memory=False)
print('-----------Data volumns:', len(colon),'Data Columns:', len(colon.columns))

'''confusion matrix'''
def conf_matrix(y_test, ans):
    CM = confusion_matrix(y_test, ans)
    FP = CM.sum(axis=0) - np.diag(CM)  
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    return FP,FN,TP,TN
'''target setting and standardize'''
def binaryClass(N):
    colon['SURVIVAL MONTHS'][colon['SURVIVAL MONTHS']<N]=0
    colon['SURVIVAL MONTHS'][colon['SURVIVAL MONTHS']>=N]=1
def multiClass(colon):
    colon['SURVIVAL MONTHS'][colon['SURVIVAL MONTHS']<12]=0
    colon['SURVIVAL MONTHS'][(colon['SURVIVAL MONTHS']>=12)&(colon['SURVIVAL MONTHS']<24)]=1
    colon['SURVIVAL MONTHS'][(colon['SURVIVAL MONTHS']>=24)&(colon['SURVIVAL MONTHS']<36)]=2
    colon['SURVIVAL MONTHS'][(colon['SURVIVAL MONTHS']>=36)&(colon['SURVIVAL MONTHS']<48)]=3
    colon['SURVIVAL MONTHS'][(colon['SURVIVAL MONTHS']>=48)&(colon['SURVIVAL MONTHS']<60)]=4
    colon['SURVIVAL MONTHS'][colon['SURVIVAL MONTHS']>=60]=5
def CSSF6_8(colon):
    colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 8'] != '   ']
    colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 6'] != '   ']
    colon[['CS SITE-SPECIFIC FACTOR 6']] = colon[['CS SITE-SPECIFIC FACTOR 6']].astype(int)
    colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 6'] < 998]
    colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 6'] != 991]
    colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 6'] != 988]
    colon['CS SITE-SPECIFIC FACTOR 6'][colon['CS SITE-SPECIFIC FACTOR 6'] == 0] = round(random.uniform(0,1),1) 
    colon['CS SITE-SPECIFIC FACTOR 6'][colon['CS SITE-SPECIFIC FACTOR 6'] == 990] = 0
    for i in range(992,997): #992~996
        s = i % 991
        colon['CS SITE-SPECIFIC FACTOR 6'][colon['CS SITE-SPECIFIC FACTOR 6'] == i] = round(random.uniform(s-1,s),1)
    return colon

#month = 60
#binaryClass(month)
multiClass(colon)

#distribution = colon['SURVIVAL MONTHS'].value_counts()
#scaler = preprocessing.scale(colon['TUMOR_SIZE'])

CSSF = ['CS SITE-SPECIFIC FACTOR 4','CS SITE-SPECIFIC FACTOR 6','CS SITE-SPECIFIC FACTOR 8'] #Null
colon = colon.drop(CSSF,axis=1)
will_dummy_columns = ['RACE RECODE (W, B, AI, API)','PRIMARY SITE','HISTOLOGIC TYPE ICD-O-3','BEHAVIOR CODE ICD-O-3'
                      ,'CS SITE-SPECIFIC FACTOR 1','CS SITE-SPECIFIC FACTOR 2','RX SUMM—SURG PRIM SITE']
colon[will_dummy_columns] = colon[will_dummy_columns].astype(str)

'''model test religion'''
print('Feature and Target Setting')
remain_columns = list(colon.columns) #30 columns
remain_columns.remove('SURVIVAL MONTHS')

'''-----------------------------Total Feature-----------------------------'''
feature_total = remain_columns
#colon = CSSF6_8(colon)

'''Final Setting'''
feature = feature_total
y = colon['SURVIVAL MONTHS']

'''dummy'''
X = pd.get_dummies(colon[feature])
X_dummy_columns = list(X.columns)
print('before Sample columns:',X.shape[1])

'''--------------------Select From Model-----------------------'''
#print('Selecting From Model...')
##Model
#clf = LinearSVC(C=0.01, penalty="l1", dual=False)
#clf.fit(X,y)
##Select
#model = SelectFromModel(clf, threshold='3.5*mean' ,prefit=True)
#X_seleted_boolean1 = model.get_support()
#X_seleted_boolean1 = X_seleted_boolean1.tolist()
#X_seleted1 = list(compress(X_dummy_columns, X_seleted_boolean1))
#Xnew = model.transform(X)
#
###使用Xnew + y 進行平衡
#
##--------------------------------------------------------------------------------------------------#
#
#'''data Sampling'''
#print('Data Sampling...')
#X, y = SMOTEENN(ratio='all',random_state=0).fit_sample(Xnew, y)
#print('After Sample Volumns:', len(X))
#
#'''split data 75 25'''
#print('Data split 75/25...')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#
###-------------------------------------------------------------------------------------------------#
##'''train and prediction'''
#print('Model Building...',month)
###-------------------------------------------------------------------------------------------------#
#'''Random Forest'''
#clf = RandomForestClassifier(n_estimators=30, max_depth=None, random_state=42)
#clf.fit(X_train,y_train)
#ans = clf.predict(X_test)
#FP,FN,TP,TN = conf_matrix(y_test,ans)
#print('-------------------Random Forest------------------') #test 78.85%
#print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
#print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
##fpr, tpr, thresholds = roc_curve(y_test,ans)
##print('AUC:', '%.6f' %auc(fpr,tpr))
#
###-------------------------------------------------------------------------------------------------#
#'''Extremly Randomize tree'''
#clf = ExtraTreesClassifier(n_estimators=30, max_depth=None, random_state=42)
#clf.fit(X_train,y_train)
#ans = clf.predict(X_test)
#FP,FN,TP,TN = conf_matrix(y_test,ans)
#print('--------------------Extra-tree--------------------')
#print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
#print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
##fpr, tpr, thresholds = roc_curve(y_test,ans)
##print('AUC:', '%.6f' %auc(fpr,tpr))
#
###-------------------------------------------------------------------------------------------------#
#'''ensemble SL1'''
#seed = 2018
#np.random.seed(seed)
#ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
#ensemble.add([ExtraTreesClassifier(n_estimators=25,random_state=seed),KNeighborsClassifier(n_neighbors=2)])
#ensemble.add_meta(SVC())
#ensemble.fit(X_train,y_train)
#ans = ensemble.predict(X_test)
#FP,FN,TP,TN = conf_matrix(y_test,ans)
#print('--------------------Super Learner--------------------') 
#print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
#print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
##fpr, tpr, thresholds = roc_curve(y_test,ans)
##print('AUC:', '%.6f' %auc(fpr,tpr))
##
###-------------------------------------------------------------------------------------------------#
#'''ensemble voting'''#(1.6.3 ,90.92)
#clf1 = AdaBoostClassifier(n_estimators=100) #LogisticRegression(random_state=42)
#clf2 = ExtraTreesClassifier(n_estimators=25,random_state=42)
#clf3 = KNeighborsClassifier(n_neighbors=5)
#
#eclf3 = VotingClassifier(estimators=[('ada', clf1), ('et', clf2), ('knn', clf3)],
#       voting='soft', weights=[1,6,3]) #flatten_transform=True
#eclf3 = eclf3.fit(X_train, y_train)
#ans = eclf3.predict(X_test)
#FP,FN,TP,TN = conf_matrix(y_test,ans)
#print('--------------------EL-Voting--------------------') 
#print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
#print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
##fpr, tpr, thresholds = roc_curve(y_test,ans)
##print('AUC:', '%.6f' %auc(fpr,tpr))