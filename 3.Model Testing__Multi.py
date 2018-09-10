# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 20:54:36 2018

@author: HIM_LAB
"""
import pandas as pd
import numpy as np
import random
from itertools import *

#preprocessing
from sklearn import preprocessing

#imbalance data
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN,SMOTETomek 
from imblearn.under_sampling import EditedNearestNeighbours 

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

#month = 12
#binaryClass(month)
multiClass(colon)

#distribution = colon['SURVIVAL MONTHS'].value_counts()
#scaler = preprocessing.scale(colon['TUMOR_SIZE'])

CSSF = ['CS SITE-SPECIFIC FACTOR 4'] #Null
colon = colon.drop(CSSF,axis=1)
will_dummy_columns = ['RACE RECODE (W, B, AI, API)','PRIMARY SITE','HISTOLOGIC TYPE ICD-O-3','BEHAVIOR CODE ICD-O-3'
                      ,'CS SITE-SPECIFIC FACTOR 1','CS SITE-SPECIFIC FACTOR 2','RX SUMM—SURG PRIM SITE']
colon[will_dummy_columns] = colon[will_dummy_columns].astype(str)

'''model test religion'''
print('Feature and Target Setting')
remain_columns = list(colon.columns) #30 columns
remain_columns.remove('SURVIVAL MONTHS')

'''-----------------------------Total Feature-----------------------------'''
#feature_total = remain_columns
#colon = CSSF6_8(colon)
print(colon['SURVIVAL MONTHS'].value_counts())
'''-------------------------------------Testing Features--------------------------------------'''
#feature_test = ['AGE AT DIAGNOSIS','TUMOR_SIZE','REGIONAL NODES POSITIVE','REGIONAL NODES EXAMINED','REASON FOR NO SURGERY','GRADE','RX SUMM—SURG PRIM SITE'] #1year 80%
#feature_test2 = ['AGE AT DIAGNOSIS','TUMOR_SIZE','REGIONAL NODES POSITIVE','REGIONAL NODES EXAMINED','LYMPH NODES','EXTENSION','RACE RECODE (W, B, AI, API)']

'''-------------------------------------Machine Learning : Features--------------------------------------'''
#feature_EFS_10 = ['AGE AT DIAGNOSIS','SEQUENCE NUMBER--CENTRAL','GRADE','REGIONAL NODES EXAMINED',
#                  'EXTENSION','LYMPH NODES','DERIVED AJCC-6 STAGE GRP','RX SUMM—SCOPE REG LN SUR',
#                  'FIRST MALIGNANT PRIMARY INDICATOR','CHEMOTHERAPY RECODE (YES, NO/UNK)']

'''-------------------------------------------Medicine features------------------------------------------'''
'''Doctor select'''#82.5900
#feature_doctor = ['TUMOR_SIZE','DERIVED AJCC-6 M','DERIVED AJCC-6 STAGE GRP','CS SITE-SPECIFIC FACTOR 1',
#                  'CS SITE-SPECIFIC FACTOR 2','CS SITE-SPECIFIC FACTOR 6','CS SITE-SPECIFIC FACTOR 8',
#                  'RACE RECODE (W, B, AI, API)','AGE AT DIAGNOSIS']
#feature_doctor2 = ['TUMOR_SIZE','DERIVED AJCC-6 M','CS SITE-SPECIFIC FACTOR 1','RACE RECODE (W, B, AI, API)',
#                   'AGE AT DIAGNOSIS']

'''2000 Compton paper''' #74.05
#feature_Compton = ['DERIVED AJCC-6 T','DERIVED AJCC-6 N','HISTOLOGIC TYPE ICD-O-3','GRADE','TUMOR_SIZE',
#                   'CS SITE-SPECIFIC FACTOR 1','CS SITE-SPECIFIC FACTOR 6','CS SITE-SPECIFIC FACTOR 8']
#colon = CSSF6_8(colon)
'''2012 Sjo paper'''#69.8400
#feature_Sjo = ['DERIVED AJCC-6 T','DERIVED AJCC-6 N','DERIVED AJCC-6 M','HISTOLOGIC TYPE ICD-O-3',
#               'GRADE','REGIONAL NODES EXAMINED', 'REGIONAL NODES POS RATIO','PRIMARY SITE',
#               'AGE AT DIAGNOSIS','SEX','CS SITE-SPECIFIC FACTOR 1','CS SITE-SPECIFIC FACTOR 8'] 
#colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 8'] != '   ']

'''Association'''#81.9595 / 83.1304
#feature_NCI = ['DERIVED AJCC-6 T','DERIVED AJCC-6 N','DERIVED AJCC-6 M','AGE AT DIAGNOSIS',
#               'DERIVED AJCC-6 STAGE GRP','CS SITE-SPECIFIC FACTOR 1','RACE RECODE (W, B, AI, API)']

#feature_TCOG = ['DERIVED AJCC-6 T','DERIVED AJCC-6 N','DERIVED AJCC-6 M','DERIVED AJCC-6 STAGE GRP',
#                'GRADE','HISTOLOGIC TYPE ICD-O-3','LYMPH NODES','CS SITE-SPECIFIC FACTOR 1']
#feature_Medical = feature_Compton + feature_Sjo + feature_NCI +feature_TCOG
#feature_Medical = list(set(feature_Medical))

#feature_rfe = ['MARITAL STATUS AT DX','AGE AT DIAGNOSIS','GRADE','REGIONAL NODES POSITIVE',
#               'DERIVED AJCC-6 T','REASON FOR NO SURGERY','ORIGIN RECODE NHIA (HISPANIC, NON-HISP)',
#               'REGIONAL NODES NEGATIVE']

'''Final Setting'''
feature = feature_Compton
y = colon['SURVIVAL MONTHS']

#--------------------------------------------------------------------------------------------------#
'''Selection Algorithm'''
print('--------------Selection---------------')
#selection_column = colon[remain_columns].drop(will_dummy_columns,axis=1) #23 columns
#selection_column_col = list(selection_column.columns)
#print('Total column:',len(remain_columns))
#print('dummy column:',len(will_dummy_columns))
#print('no dummy column:',len(selection_column_col))

'''Sequential Feature Selector'''
#subcolon = colon.sample(n=10000)
#sub_x = subcolon[remain_columns]
#sub_x = pd.get_dummies(sub_x)
#sub_y = subcolon['SURVIVAL MONTHS']
#knn = KNeighborsClassifier(n_neighbors=5)
#sffs = SFS(knn, 
#           k_features=20, 
#           forward=True, #or backward
#           floating=False,
#           scoring='accuracy',
#           cv=0,
#           n_jobs=-1)
#sffs = sffs.fit(sub_x, sub_y)

#print('\nSequential Forward Floating Selection (k=3):')
#print(sffs.k_feature_idx_)
#print('CV Score:')
#print(sffs.k_score_)
#print('-------------------------------------------------------------------')

'''RFE CV'''
#estimator = SVC(kernel="linear")
#selector = RFE(estimator, 10, step=1)
#selector = selector.fit(sub_x, sub_y)
#sub_feature = list(sub_x.columns)
#RFE_feature = list(selector.support_)
#RFE_seleted = list(compress(sub_feature, RFE_feature))

'''Exhaustive Feature Selector'''
#knn = KNeighborsClassifier(n_neighbors=5)
#efs1 = EFS(knn, 
#           min_features=1,
#           max_features=8,
#           scoring='accuracy',
#           print_progress=True,
#           cv=0)
#efs1 = efs1.fit(sub_x.as_matrix(),sub_y)

#print('Best accuracy score: %.2f' % efs1.best_score_)
#print('Best subset:', efs1.best_idx_)

'''BoostARoota'''
#from boostaroota import BoostARoota
#X = pd.get_dummies(colon[feature_total])
#br = BoostARoota(metric='logloss')
#br.fit(X,y)
#X = br.transform(X) # transform to importance var

'''dummy'''
X = pd.get_dummies(colon[feature])
X_dummy_columns = list(X.columns)
print('before Sample columns:',X.shape[1])

'''data Sampling'''
print('Data Sampling...')
X, y = SMOTEENN(ratio='all',random_state=0).fit_sample(X, y)
print('After Sample Volumns:', len(X))
print('After Sample Columns:', X.shape[1])

'''split data 75 25'''
print('Data split 75/25...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#'''data Sampling 2'''
#print('train Data Sampling...')
#X_train, y_train = SMOTEENN(ratio='all',random_state=0).fit_sample(X_train, y_train)
#print('test Data Sampling...')
#X_test, y_test = SMOTEENN(ratio='all',random_state=0).fit_sample(X_test, y_test)


#-------------------------------------------------------------------------------------------------#
'''train and prediction'''
print('Model Building...')
#-------------------------------------------------------------------------------------------------#
'''Extremly Randomize tree'''
clf = ExtraTreesClassifier(n_estimators=30, max_depth=None, random_state=42)
clf.fit(X_train,y_train)
ans = clf.predict(X_test)
FP,FN,TP,TN = conf_matrix(y_test,ans)
print('--------------------Extra-tree--------------------')
print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
print('Specificity(TNR):', '%.6f' %np.mean(TN/(TN+FP)))

model = SelectFromModel(clf, prefit=True)
X_seleted_boolean = model.get_support()
X_seleted_boolean = X_seleted_boolean.tolist()
X_seleted = list(compress(X_dummy_columns, X_seleted_boolean))

#-------------------------------------------------------------------------------------------------#
'''Random Forest'''
clf = RandomForestClassifier(n_estimators=30, max_depth=None, random_state=32)
clf.fit(X_train,y_train)
ans = clf.predict(X_test)
FP,FN,TP,TN = conf_matrix(y_test,ans)
print('-------------------Random Forest------------------') #test 78.85%
print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
print('Specificity(TNR):', '%.6f' %np.mean(TN/(TN+FP)))
model = SelectFromModel(clf, prefit=True)
X_seleted_boolean1 = model.get_support()
X_seleted_boolean1 = X_seleted_boolean1.tolist()
X_seleted1 = list(compress(X_dummy_columns, X_seleted_boolean1))

#-------------------------------------------------------------------------------------------------#
'''ensemble SL1'''
seed = 2018
np.random.seed(seed)
ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
ensemble.add([ExtraTreesClassifier(n_estimators=30,random_state=seed),KNeighborsClassifier(n_neighbors=2),AdaBoostClassifier(n_estimators=100)])
ensemble.add_meta(SVC())
ensemble.fit(X_train,y_train)
ans = ensemble.predict(X_test)
FP,FN,TP,TN = conf_matrix(y_test,ans)
print('--------------------Super Learner--------------------') #test 78.85%
print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
print('Specificity(TNR):', '%.6f' %np.mean(TN/(TN+FP)))

'''ensemble SL2'''
#seed = 2018
#np.random.seed(seed)
#ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
#ensemble.add([ExtraTreesClassifier(n_estimators=30,random_state=seed),AdaBoostClassifier(n_estimators=100)])
#ensemble.add_meta(SVC())
#ensemble.fit(X_train,y_train)
#ans = ensemble.predict(X_test)
#FP,FN,TP,TN = conf_matrix(y_test,ans)
#print('--------------------Super Learner--------------------') #test 78.85%
#print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
#print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
#print('Specificity(TNR):', '%.6f' %np.mean(TN/(TN+FP)))

#-------------------------------------------------------------------------------------------------#
'''ensemble voting'''#(1.6.3 ,90.92)
clf1 = AdaBoostClassifier(n_estimators=100) #LogisticRegression(random_state=42)
clf2 = ExtraTreesClassifier(n_estimators=30,random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=2)

eclf3 = VotingClassifier(estimators=[('ada', clf1), ('et', clf2), ('knn', clf3)],
       voting='soft', weights=[1,6,2]) #flatten_transform=True
eclf3 = eclf3.fit(X_train, y_train)
ans = eclf3.predict(X_test)
FP,FN,TP,TN = conf_matrix(y_test,ans)
print('--------------------EL-Voting--------------------') 
print('Precision:','%.6f' %precision_score(y_test, ans, average='macro'))
print('Recall:', '%.6f' %recall_score(y_test, ans, average='macro'))
print('Specificity(TNR):', '%.6f' %np.mean(TN/(TN+FP)))

#-------------------------------------------------------------------------------------------------#
'''XGB'''
#dtrain = xgb.DMatrix(X_train, label = y_train)
#dtest = xgb.DMatrix(X_test, label = y_test)
#xgbc = XGBClassifier()
#xgbc.fit(X_train, y_train)
#print(xgbc.score(X_test,y_test))

#-------------------------------------------------------------------------------------------------#
'''SVM - multiclass''' 
#clf = LinearSVC(C=0.01, penalty="l1", dual=False)
#clf.fit(X_train, y_train) 
#ans = clf.predict(X_test)
#print('-------------------SVM------------------') #test 78.85%
#print('Precision:','%.6f' %precision_score(y_test, ans, average='micro'))
#print('Recall:', '%.6f' %recall_score(y_test, ans, average='micro'))
#print('Specificity(TNR):', '%.6f' %np.mean(TN/(TN+FP)))
#
#model = SelectFromModel(clf, prefit=True)
#X_seleted_boolean2 = model.get_support()
#X_seleted_boolean2 = X_seleted_boolean2.tolist()
#X_seleted2 = list(compress(X_dummy_columns, X_seleted_boolean2))
#-------------------------------------------------------------------------------------------------#
'''sum the dummy importance'''
#print('dummy importance...')
#feature = feature_total #要改
#dummy_size = [] #有dummy過後的欄位數量
#importances = clf.feature_importances_
#for i in feature:
#    if type(colon[feature][i][0]) == str: 
#        dummy_size.append(len(colon[feature][i].drop_duplicates()))
#    else: dummy_size.append(1)
#
#'''排序dummy過的欄位數量'''
#less = []
#more = []
#for i in dummy_size:
#    if i>1: more.append(i)
#    else: less.append(i)
#New = less + more #New = dummy欄位後的順序(有Dummy的都會排在最後)
#
#'''#依照New的每一個大小去加總
##並記錄有加總的index位置，以利後續取columns使用'''
#New_importance = [] #sum dummy importance
#New_importance_avg = []
#dummy_index = [] #有dummy的index位置
#end=0
#for i in range(len(New)):
#    if dummy_size[i]>1: dummy_index.append(i) #return index
#    start = end
#    end = start + New[i]
#    avg = start - end
#    New_importance.append(sum(importances[start:end]))
#    New_importance_avg.append(sum(importances[start:end])/avg)
#    
#imp_columns = feature
#'''排序dummy columns'''
#dummy_over1_col = [] #有dummy過的欄位名稱
#for i in dummy_index:
#    x = imp_columns[i]
#    dummy_over1_col.append(x) #每一回將超出1的加入暫存columns中
#for j in dummy_over1_col:
#    imp_columns.remove(j)
#New_columns = imp_columns + dummy_over1_col #排序完成的columns
#ranking = pd.DataFrame({'Columns':New_columns, 'Importance':New_importance})

'''other model testing'''
#result = cross_val_score(ExtraTreesClassifier(n_estimators=100),X,y,scoring='accuracy',cv=5)
#print(result.mean())

#scores = cross_val_score(clf, sur_5[test_columns], sur_5['SURVIVAL MONTHS'])
#result2 = cross_val_score(tree.DecisionTreeClassifier(),sur_5[test_columns2],
#                         sur_5['SURVIVAL MONTHS'], scoring='accuracy',cv=10)
#clf = svm.SVC()
#clf.fit(X_train, y_train)
#ans = clf.predict(X_test)
#print(accuracy_score(y_test, ans))

'''data testing'''
#test = colon['RX SUMM—SURG TYPE']
#IsSpace = test.apply(lambda x: str(x).isspace())
#df_Has_Val = test[~IsSpace]
#typeCount = df_Has_Val.value_counts()
#drop_colName = []
#for i in range(len(tt)):
#    if type(tt[i])==str and tt[i].isspace():
#        drop_colName.append(tt.index[i])
#sur_count = sur_5['SURVIVAL MONTHS'].value_counts()
tStop = time.time()
total_run_time = (tStop - tStart)/60
print('-----------Executing time(mins) =', total_run_time)