# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:49:41 2017

@author: HIM_LAB
"""

'''
2-1.分割資料集(1~5 year survival  [0_survived , 1_not survival])
2-2.分割資料集(0~60 month)
'''

from decimal import *

import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import random
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN,SMOTETomek 
from imblearn.ensemble import BalancedBaggingClassifier 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

#data = pd.read_csv('2004+.csv', delimiter=',', converters={'PATIENT ID NUMBER': lambda x: str(x)},low_memory=False)
#data = pd.read_csv('colon_ThreeSubset_radiation_1988.csv', delimiter=',', dtype = str, low_memory=False)
data = pd.read_csv('2010+.csv', delimiter=',', dtype = str, low_memory=False)
print('origin data:',len(data))
other_cancer_PID = pd.read_csv('all_cancers_patient_id.csv',delimiter=',', dtype = str)

'''把資料切成colon,rectum'''
colon = data.loc[data['PRIMARY SITE'].str.find('C18') == 0] # == 頭開始
colon = colon.reset_index(drop=True)
rectum1 = data.loc[data['PRIMARY SITE'].str.find('C19') == 0] #乙狀直腸
rectum2 = data.loc[data['PRIMARY SITE'].str.find('C20') == 0] #一般直腸
Anus = data.loc[data['PRIMARY SITE'].str.find('C21') == 0] #Anus
rectum_other = data.loc[data['PRIMARY SITE'].str.find('C26') == 0] #rectum other position
'''del rectum3?, because it's not in colon or rectum position '''
colon = pd.concat([colon,rectum_other])
rectum_Anus = pd.concat([rectum1,rectum2,Anus])
#rectum = rectum.reset_index(drop=True)
del rectum1,rectum2,rectum_other,data #release the memory 

'''comorbidity counter '''
other_category = list(other_cancer_PID.columns)
#select_com = other_cancer_PID['Colon']
other_category.remove('Colon')

def in_cancer(category_name):
    other_cancer_register = other_cancer_PID[category_name] #癌症的patient id
    test = colon['PATIENT ID NUMBER'].isin(other_cancer_register)
    colon[category_name] = test.astype(int) # 0=false 1=true
def in_cancer_select(select_com):
    test = colon['PATIENT ID NUMBER'].isin(select_com)
    colon['combid_cancer'] = test.astype(int) # 0=false 1=true

select_com = rectum_Anus['PATIENT ID NUMBER']
in_cancer_select(select_com)

#for i in other_category: # 9 iteration
#    in_cancer(i)
#    print(i)
#colon['comorbidity_Summary'] = colon[other_category].sum(axis=1) #sum by row for specific columns 新增欄位 - 其他共病癌症數量
#colon = colon.drop(other_category,axis = 1) #只留下共病總數

#del other_cancer_PID
'''fetch dead by cancer case'''#0:Alive or dead due to cancer 1:Dead
colon = colon.loc[colon['SEER OTHER CAUSE OF DEATH CLASSIFICATION'] !='1']
colon = colon.drop(['PATIENT ID NUMBER','YEAR OF DIAGNOSIS','SEER RECORD NUMBER','SEER CAUSE-SPECIFIC DEATH CLASSIFICATION',
                    'SEER OTHER CAUSE OF DEATH CLASSIFICATION','DERIVED HER2 RECODE (2010+)',
                    'LYMPHOMAS: ANN ARBOR STAGING (1983+)','RADIATION RECODE','RX SUMM—SURG/RAD SEQ'],axis=1)

#---資料recode,刪除缺失值,---------------------------------------------------------------------------------------------#
    
'''str --> numeric'''
#IsSpace = colon['TUMOR_SIZE'].apply(lambda x: str(x).isspace()) #判斷欄位中有空值的rows
colon[['TUMOR_SIZE','SURVIVAL MONTHS','SEQUENCE NUMBER--CENTRAL','MARITAL STATUS AT DX','SEX','REGIONAL NODES POSITIVE','REGIONAL NODES EXAMINED','AGE RECODE <1 YEAR OLDS','AGE AT DIAGNOSIS']] = \
colon[['TUMOR_SIZE','SURVIVAL MONTHS','SEQUENCE NUMBER--CENTRAL','MARITAL STATUS AT DX','SEX','REGIONAL NODES POSITIVE','REGIONAL NODES EXAMINED','AGE RECODE <1 YEAR OLDS','AGE AT DIAGNOSIS']].astype(int) #處理連續型資料

'''survival month delete unknown data'''
colon = colon.loc[colon['SURVIVAL MONTHS'] < 9999] #去除未知存活月

'''marital recode  0=(2)married (1)=Single or Divorced...'''
colon['MARITAL STATUS AT DX'][colon['MARITAL STATUS AT DX'] == 2 ] = 0
colon['MARITAL STATUS AT DX'][colon['MARITAL STATUS AT DX'] > 0 ] = 1

'''SEX recode 0=male 1=female'''
colon['SEX'] = colon['SEX'] - 1

'''Race delete unknown'''
#colon = colon.loc[colon['RACE / ETHNICITY']!='99']
#colon = colon.loc[colon['RACE RECODE (WHITE, BLACK, OTHER)']!='9']
colon = colon.loc[colon['RACE RECODE (W, B, AI, API)']!='9']
colon[['ORIGIN RECODE NHIA (HISPANIC, NON-HISP)']] = colon[['ORIGIN RECODE NHIA (HISPANIC, NON-HISP)']].astype(int)

'''Age delete unknown'''
colon = colon.loc[colon['AGE AT DIAGNOSIS']!=999]

'''Sqeuence number central (del 99)'''#line79 已轉換成int
colon = colon.loc[colon['SEQUENCE NUMBER--CENTRAL']!=99]

'''Grade Let unknown grade = grade 0'''
colon['GRADE'][colon['GRADE'] == '9'] = 0
colon[['GRADE']] = colon[['GRADE']].astype(int)

'''positive examine recode'''
colon = colon[(colon['REGIONAL NODES POSITIVE']==98) | (colon['REGIONAL NODES POSITIVE']<=90)] #positve fetch actual number
colon = colon[colon['REGIONAL NODES EXAMINED']<=90] #examine fetch actual number
#recode to positive attribute[No nodes were examined] let examine can Less positive
colon['REGIONAL NODES EXAMINED'][colon['REGIONAL NODES EXAMINED']==0] = 98

'''generate Negative column''' #new 2 columns
colon['REGIONAL NODES NEGATIVE'] = colon['REGIONAL NODES EXAMINED'] - colon['REGIONAL NODES POSITIVE']
colon['REGIONAL NODES POS RATIO'] = colon['REGIONAL NODES POSITIVE'] / colon['REGIONAL NODES EXAMINED'] 
colon['REGIONAL NODES EXAMINED'][colon['REGIONAL NODES EXAMINED']==98] = 0 #變回來原本的編碼[0個被檢測]
colon['REGIONAL NODES POSITIVE'][colon['REGIONAL NODES POSITIVE']==98] = -1 #將都沒檢測到的設為-1

'''Tumor Size [mm -> cm]'''
#[mm -> cm] for continous data
colon['TUMOR_SIZE'][colon['TUMOR_SIZE'] < 990] = colon['TUMOR_SIZE'][colon['TUMOR_SIZE'] < 990]/10
#colon = colon.loc[colon['TUMOR_SIZE']<996]
#unknown(999), Familial(998), ???(997)  == mean (未知數值用mean取代)
tumor_size_continous = colon.loc[colon['TUMOR_SIZE'] < 996]
tumor_size_mean = round(tumor_size_continous['TUMOR_SIZE'].mean(),1)
colon['TUMOR_SIZE'][colon['TUMOR_SIZE'] > 995] = tumor_size_mean

#tumor fuzzy interval for 990-995
for i in range(990,996): 
    s = i % 989
    colon['TUMOR_SIZE'][colon['TUMOR_SIZE'] == i] = round(random.uniform(s-1,s),1)

#CMcount = colon['TUMOR_SIZE'].value_counts() #腫瘤大小的總數(包含未知、息肉)
#LYcount = colon['LYMPH NODES'].value_counts()
#lymph = list(np.sort((colon['LYMPH NODES'].value_counts()).index))

'''Extension'''
#ext_count = colon['EXTENSION'].value_counts()
colon['EXTENSION'][colon['EXTENSION'] == '999'] = -1
colon['EXTENSION'][colon['EXTENSION'] == '950'] = -2
colon[['EXTENSION']] = colon[['EXTENSION']].astype(int)
le = preprocessing.LabelEncoder().fit(colon['EXTENSION'])
colon['EXTENSION'] = le.transform(colon['EXTENSION'])
#ext = list(np.sort((colon['EXTENSION'].value_counts()).index))
#ext_count2 = colon['EXTENSION'].value_counts()

'''lymph node & METS AT DX'''
del colon['CS METS AT DX'] #one cate massive large than others
#lymphNodes = {'000':0,'050':1,'100':1,'110':1,
#              '200':2,'210':2,'220':2,
#              '300':3,'400':4,'410':4,
#              '800':5,'999':6} #mapping 
#colon['LYMPH NODES'] = colon['LYMPH NODES'].map(lymphNodes)
colon[['LYMPH NODES']] = colon[['LYMPH NODES']].astype(int)
le = preprocessing.LabelEncoder().fit(colon['LYMPH NODES'])
colon['LYMPH NODES'] = le.transform(colon['LYMPH NODES'])
lym = colon['LYMPH NODES'].value_counts()

'''CSSF series delele isspace rows(CSSF1,2) & excessive isspace columns(CSSF4,6,8)'''
colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 1'] != '   ']
colon = colon.loc[colon['CS SITE-SPECIFIC FACTOR 2'] != '   ']
colon = colon.drop(['CS SITE-SPECIFIC FACTOR 4','CS SITE-SPECIFIC FACTOR 6','CS SITE-SPECIFIC FACTOR 8'],axis=1)
cs1 = colon['CS SITE-SPECIFIC FACTOR 1'].value_counts()
cs2 = colon['CS SITE-SPECIFIC FACTOR 2'].value_counts()


'''delete CSSF == space(null)'''
#CSSF = ['CS SITE-SPECIFIC FACTOR 1','CS SITE-SPECIFIC FACTOR 2','CS SITE-SPECIFIC FACTOR 3','CS SITE-SPECIFIC FACTOR 4','CS SITE-SPECIFIC FACTOR 5',
#        'CS SITE-SPECIFIC FACTOR 6','CS SITE-SPECIFIC FACTOR 7','CS SITE-SPECIFIC FACTOR 8','CS SITE-SPECIFIC FACTOR 9','CS SITE-SPECIFIC FACTOR 10']
#CSSF_count = []
#for i in CSSF:
#    CSSF_count.append(colon[i].value_counts())
colon = colon.drop(['CS SITE-SPECIFIC FACTOR 3','CS SITE-SPECIFIC FACTOR 5','CS SITE-SPECIFIC FACTOR 7',
                    'CS SITE-SPECIFIC FACTOR 9','CS SITE-SPECIFIC FACTOR 10'], axis=1)

'''6th TNM + Stage'''
colon = colon.loc[colon['DERIVED AJCC-6 T'] != '88'] #88 = not applicable 
#T0 Tx Tis T1 2 3 4
colon['DERIVED AJCC-6 T'][colon['DERIVED AJCC-6 T']=='99'] = '01' #99 = Tx , Tx is worse than T0
colon['DERIVED AJCC-6 N'][colon['DERIVED AJCC-6 N']=='99'] = '01'
colon['DERIVED AJCC-6 M'][colon['DERIVED AJCC-6 M']=='99'] = '01'
colon = colon.loc[colon['DERIVED AJCC-6 T']!='00']
colon = colon.loc[colon['DERIVED AJCC-6 M']!='01'] #delete outllier (less sample)

#refer excel table
le = preprocessing.LabelEncoder().fit(colon['DERIVED AJCC-6 T'])
colon['DERIVED AJCC-6 T'] = le.transform(colon['DERIVED AJCC-6 T'])
le = preprocessing.LabelEncoder().fit(colon['DERIVED AJCC-6 N'])
colon['DERIVED AJCC-6 N'] = le.transform(colon['DERIVED AJCC-6 N'])
le = preprocessing.LabelEncoder().fit(colon['DERIVED AJCC-6 M'])
colon['DERIVED AJCC-6 M'] = le.transform(colon['DERIVED AJCC-6 M'])

    #Stage
colon = colon.loc[colon['DERIVED AJCC-6 STAGE GRP']!='99'] #delete unknown
colon['DERIVED AJCC-6 STAGE GRP'][colon['DERIVED AJCC-6 STAGE GRP'].str.find('3') == 0] = '20'
colon['DERIVED AJCC-6 STAGE GRP'][colon['DERIVED AJCC-6 STAGE GRP'].str.find('5') == 0] = '30'
colon['DERIVED AJCC-6 STAGE GRP'][colon['DERIVED AJCC-6 STAGE GRP'].str.find('7') == 0] = '40'
le = preprocessing.LabelEncoder().fit(colon['DERIVED AJCC-6 STAGE GRP'])
colon['DERIVED AJCC-6 STAGE GRP']=le.transform(colon['DERIVED AJCC-6 STAGE GRP'])

'''RX Surgery recode'''#無序
colon['RX SUMM—SURG PRIM SITE'][colon['RX SUMM—SURG PRIM SITE'].str.find('1') == 0] = '100' #資料太少所以set=100,方便後續刪除
colon['RX SUMM—SURG PRIM SITE'][colon['RX SUMM—SURG PRIM SITE'].str.find('2') == 0] = '20'
colon['RX SUMM—SURG PRIM SITE'][colon['RX SUMM—SURG PRIM SITE'].str.find('3') == 0] = '30'
colon['RX SUMM—SURG PRIM SITE'][colon['RX SUMM—SURG PRIM SITE'].str.find('4') == 0] = '40'
colon['RX SUMM—SURG PRIM SITE'][colon['RX SUMM—SURG PRIM SITE'].str.find('5') == 0] = '50'
colon['RX SUMM—SURG PRIM SITE'][colon['RX SUMM—SURG PRIM SITE'].str.find('6') == 0] = '60' #70只有單一種，沒有子類所以不用recode
colon = colon.loc[colon['RX SUMM—SURG PRIM SITE'].astype(int) < 80] #delete too less category

'''RX Surgery scope'''
scope_sur = colon['RX SUMM—SCOPE REG LN SUR'].value_counts()<700  #個數小於700的標籤
for i in scope_sur.index:
    if scope_sur[i]: colon = colon.loc[colon['RX SUMM—SCOPE REG LN SUR'] != i] #if true then delete
test = colon['RX SUMM—SCOPE REG LN SUR'].value_counts()

colon[['RX SUMM—SCOPE REG LN SUR']] = colon[['RX SUMM—SCOPE REG LN SUR']].astype(int)
le = preprocessing.LabelEncoder().fit(colon['RX SUMM—SCOPE REG LN SUR'])
colon['RX SUMM—SCOPE REG LN SUR'] = le.transform(colon['RX SUMM—SCOPE REG LN SUR'])

'''Reason for no surgery'''#binary category
colon = colon.loc[colon['REASON FOR NO SURGERY'].astype(int) < 8] #delete 8,9
colon['REASON FOR NO SURGERY'][colon['REASON FOR NO SURGERY'].astype(int)>0] = 1 #let >0 = 1 indicate 'no surgery'
colon[['REASON FOR NO SURGERY']] = colon[['REASON FOR NO SURGERY']].astype(int)

'''Behavior & Histology'''
behavior = colon['BEHAVIOR RECODE FOR ANALYSIS'].value_counts()
histology = colon['HISTOLOGY RECODE—BROAD GROUPINGS'].value_counts()

'''First Malignant Primary Indicator'''#binary category
colon['FIRST MALIGNANT PRIMARY INDICATOR'] = colon['FIRST MALIGNANT PRIMARY INDICATOR'].astype(int) 

'''Total number of XXX'''
colon = colon.loc[colon['TOTAL NUMBER OF IN SITU/MALIGNANT TUMORS FOR PATIENT']!=99]
colon[['TOTAL NUMBER OF IN SITU/MALIGNANT TUMORS FOR PATIENT']] = colon[['TOTAL NUMBER OF IN SITU/MALIGNANT TUMORS FOR PATIENT']].astype(int)
colon = colon.drop(['TOTAL NUMBER OF BENIGN/BORDERLINE TUMORS FOR PATIENT'],axis=1) #skew
colon[['CHEMOTHERAPY RECODE (YES, NO/UNK)']] = colon[['CHEMOTHERAPY RECODE (YES, NO/UNK)']].astype(int)

'''CS MET Bone,brain,liver,lung''' #del 8,9 #2010+ col
#colon = colon.loc[colon['CS METS AT DX-BONE'].astype(int) < 8]
#colon = colon.loc[colon['CS METS AT DX-BRAIN'].astype(int) < 8]
#colon = colon.loc[colon['CS METS AT DX-LIVER'].astype(int) < 8]
#colon = colon.loc[colon['CS METS AT DX-LUNG'].astype(int) < 8]

'''drop 2010+ & skew columns'''
drop_2010 = ['CS METS AT DX-BONE','CS METS AT DX-BRAIN','CS METS AT DX-LUNG','CS METS AT DX-LIVER']
drop_skew = ['LATERALITY','STATE-COUNTY RECODE']
drop_worse = ['AGE RECODE <1 YEAR OLDS']
drop_CS_eval = ['CS TUMOR SIZE EXT/EVAL','CS LYMPH NODES EVAL','CS METS EVAL']
colon = colon.drop(drop_2010 + drop_skew + drop_worse + drop_CS_eval,axis=1)

'''Race'''
Race_group1 = ['RACE RECODE (WHITE, BLACK, OTHER)','RACE RECODE (W, B, AI, API)']
#Race_group2 = ['RACE / ETHNICITY','RACE RECODE (W, B, AI, API)']
#Race_group3 = ['RACE / ETHNICITY','RACE RECODE (WHITE, BLACK, OTHER)']
'''HIS/Be'''
group1 = ['HISTOLOGY RECODE—BROAD GROUPINGS','BEHAVIOR RECODE FOR ANALYSIS']
group2 = ['HISTOLOGIC TYPE ICD-O-3','BEHAVIOR CODE ICD-O-3']
colon = colon.drop(Race_group1+group1,axis=1)
'''model testing'''
#survival_5 = rectum[rectum['YEAR OF DIAGNOSIS'].astype(int)<=2011] #2016/11結案，五年被診斷參考至2011/11
#survival_5 = survival_5[survival_5['MONTHA OF DIAGNOSIS'].astype(int)<=11]

sur_5 = colon
sur_5['SURVIVAL MONTHS'][sur_5['SURVIVAL MONTHS']<12]=0
sur_5['SURVIVAL MONTHS'][(sur_5['SURVIVAL MONTHS']>=12)&(sur_5['SURVIVAL MONTHS']<24)]=1
sur_5['SURVIVAL MONTHS'][(sur_5['SURVIVAL MONTHS']>=24)&(sur_5['SURVIVAL MONTHS']<36)]=2
sur_5['SURVIVAL MONTHS'][(sur_5['SURVIVAL MONTHS']>=36)&(sur_5['SURVIVAL MONTHS']<48)]=3
sur_5['SURVIVAL MONTHS'][(sur_5['SURVIVAL MONTHS']>=48)&(sur_5['SURVIVAL MONTHS']<60)]=4
sur_5['SURVIVAL MONTHS'][sur_5['SURVIVAL MONTHS']>=60]=5
print('pre-processed:',len(sur_5))

remain_columns = list(colon.columns)
remain_columns.remove('SURVIVAL MONTHS')
#col_counts = []
#for i in remain_columns:
#    col_counts.append(colon[i].value_counts())

'''model test religion'''
print('Feature and Target Setting')
#feature = ['AGE AT DIAGNOSIS','TUMOR_SIZE','REGIONAL NODES POSITIVE','REGIONAL NODES EXAMINED','REASON FOR NO SURGERY','GRADE','RX SUMM—SURG PRIM SITE'] #1year 80%
#feature = ['AGE AT DIAGNOSIS','TUMOR_SIZE','REGIONAL NODES POSITIVE','REGIONAL NODES EXAMINED','LYMPH NODES','EXTENSION','RACE RECODE (W, B, AI, API)']
feature = remain_columns
X = pd.get_dummies(sur_5[feature])
X_dummy_columns = list(X.columns)
y = sur_5['SURVIVAL MONTHS']
print('Data Sampling...')
#X, y = SMOTEENN(random_state=42).fit_sample(X, y)
print('After Sample:', len(X))
print('Data split 75/25...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print('Model Building...')
clf = ExtraTreesClassifier(n_estimators=30, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(X_train,y_train)
ans = clf.predict(X_test)
print(accuracy_score(y_test, ans))

'''sum the dummy importance'''
print('dummy importance...')

dummy_size = []
importances = clf.feature_importances_
for i in remain_columns:
    if type(sur_5[feature][i][0]) == str: dummy_size.append(len(sur_5[feature][i].drop_duplicates()))
    else: dummy_size.append(1)
#importance_score = []
#end = 0
#for i in range(len(dummy_size)):
#    start = end
#    end = start + dummy_size[i]
#    print(start,end)
#    importance_score.append(np.sum(importances[start:end]))
lst = []
for i in range(len(dummy_size)):
    if dummy_size[i] > 1: 
        lst.append(dummy_size.pop(i))
dummy_size.append(lst)
'''importance check'''
#a = np.asarray(remain_columns)
#b = np.asarray(importance_score)
#b = b.astype(float)
#c = np.vstack((b, a)).T
#ranking = pd.DataFrame(c,columns=['score','names'])
#ranking[['score']] = ranking[['score']].astype(float)

#result = cross_val_score(ExtraTreesClassifier(n_estimators=100),X,y,scoring='accuracy',cv=5)
#print(result.mean())

#scores = cross_val_score(clf, sur_5[test_columns], sur_5['SURVIVAL MONTHS'])
#result2 = cross_val_score(tree.DecisionTreeClassifier(),sur_5[test_columns2],
#                         sur_5['SURVIVAL MONTHS'], scoring='accuracy',cv=10)
#clf = svm.SVC()
#clf.fit(X_train, y_train)
#ans = clf.predict(X_test)
#print(accuracy_score(y_test, ans))

'''testing'''
#test = colon['RX SUMM—SURG TYPE']
#IsSpace = test.apply(lambda x: str(x).isspace())
#df_Has_Val = test[~IsSpace]
#typeCount = df_Has_Val.value_counts()
#drop_colName = []
#for i in range(len(tt)):
#    if type(tt[i])==str and tt[i].isspace():
#        drop_colName.append(tt.index[i])
#sur_count = sur_5['SURVIVAL MONTHS'].value_counts()