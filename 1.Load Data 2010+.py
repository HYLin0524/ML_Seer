# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:06:52 2018

@author: HIM_LAB
"""

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 

data = pd.read_csv('COLRECT2000_2014.TXT' ,sep=" ", error_bad_lines=False, header=None)

'''build Lenght list'''
columnsTotal = pd.read_excel('Feature and positions radiation.xlsx')
Lenght = [] #each column len
for i in range(len(columnsTotal)):
    Lenght.append(columnsTotal['Position'][i].split('-'))
colLenght = len(Lenght)
'''let single item to twice. Ex:19 -> ['19','19']'''
for i in range(len(Lenght)):
    if len(Lenght[i]) == 1 :
        Lenght[i].append(Lenght[i][0])
'''str to int and first item -1'''
for i in range(len(Lenght)):
    Lenght[i][0] = int(Lenght[i][0])-1
    Lenght[i][1] = int(Lenght[i][1])
'''build Column Name list'''
ColName = [] # each column name
for j in range(len(Lenght)):
    ColName.append(columnsTotal['columnsName'][j])

'''Read DataSet and combine 3 dataset'''
SeerData1 = pd.read_csv('COLRECT1973_2014_Radiation.TXT',header=None)
SeerData1.columns = ['Origin']
SeerData2 = pd.read_csv('COLRECT1992_2014_Radiation.TXT',header=None)
SeerData2.columns = ['Origin']
SeerData3 = pd.read_csv('COLRECT2000_2014_Radiation.TXT',header=None)
SeerData3.columns = ['Origin']
'''all dataframe(3subset)'''
threeDataset = pd.concat([SeerData1,SeerData2,SeerData3])
del SeerData1,SeerData2,SeerData3

'''split one column to multi'''
for i in range(len(ColName)): #len = 133
    if(i%10==0):print('split:',i)
    register = []
    for j in threeDataset['Origin']:
        register.append(j[Lenght[i][0]:Lenght[i][1]])
    threeDataset[ColName[i]] = pd.Series(register ,index=threeDataset.index)
del register

'''drop first origin columns'''
threeDataset = threeDataset.drop(['Origin'], axis=1)
threeDataset['YEAR OF DIAGNOSIS'] = threeDataset['YEAR OF DIAGNOSIS'].astype('int', copy=False) #change the column type

drop_useless = ['REGISTRY ID','BIRTHDATE—YEAR','MONTH OF DIAGNOSIS','EOD—EXTENSION PROST PATH',
                'TUMOR MARKER 1','TUMOR MARKER 2','TUMOR MARKER 3','ER STATUS RECODE BREAST CANCER (1990+)',
                'PR STATUS RECODE BREAST CANCER (1990+)','CS SCHEMA—AJCC 6TH ED (PREVIOUSLY CALLED V1)',
                'BREAST ADJUSTED AJCC 6TH T (1988+)','BREAST ADJUSTED AJCC 6TH N (1988+)','BREAST ADJUSTED AJCC 6TH M (1988+)',
                'BREAST ADJUSTED AJCC 6TH STAGE (1988+)','BREAST SUBTYPE (2010+)','IHS LINK','LYMPH VASCULAR INVASION'] #'PATIENT ID NUMBER'
threeDataset = threeDataset.drop(drop_useless, axis=1)
print('------------Delete the useless column: ',len(drop_useless))

'''delete the decisive factor'''
drop_decisive = ['DIAGNOSTIC CONFIRMATION','TYPE OF REPORTING SOURCE','RX SUMM-SURG OTH REG/DIS','RX SUMM-REG LN EXAMINED',
                 'SEER TYPE OF FOLLOW-UP','RADIATION TO BRAIN OR CNS RECODE (1988-1997)','CAUSE OF DEATH TO SEER SITE RECODE',
                 'VITAL STATUS RECODE','COD TO SITE REC KM']
threeDataset = threeDataset.drop(drop_decisive, axis=1)
print('------------Delete the decisive column: ', len(drop_decisive))

drop_null = ['CS SITE-SPECIFIC FACTOR 11','CS SITE-SPECIFIC FACTOR 12','CS SITE-SPECIFIC FACTOR 13',
             'CS SITE-SPECIFIC FACTOR 15','CS SITE-SPECIFIC FACTOR 25','CS SITE-SPECIFIC FACTOR 16']
threeDataset = threeDataset.drop(drop_null, axis=1)
print('------------Delete the null columns: ', len(drop_null))

'''delete the code columns'''
drop_code = ['HISTOLOGY (92-00) ICD-O-2','BEHAVIOR (92-00) ICD-O-2','DERIVED AJCC—FLAG','CS VERSION INPUT ORIGINAL',
             'CS VERSION DERIVED','CS VERSION INPUT CURRENT','RECODE ICD-O-2 TO 9','RECODE ICD-O-2 TO 10',
             'CS SCHEMA V0204+','INSURANCE RECODE (2007+)','SURVIVAL MONTHS FLAG','SITE RECODE ICD-O-3/WHO 2008','LYMPHOMA SUBTYPE RECODE/WHO 2008']
threeDataset = threeDataset.drop(drop_code, axis=1)
print('------------Delete the code column: ',len(drop_code))

drop_rare2 = ['RX SUMM—SURG TYPE','RX SUMM—SCOPE REG 98-02','RX SUMM—SURG OTH 98-02','SEER SUMMARY STAGE 1977',
             'SEER SUMMARY STAGE 2000','DERIVED SS1977','DERIVED SS2000','AJCC STAGE 3RD EDITION (1988-2003)',
             'DERIVED AJCC-7 T','DERIVED AJCC-7 N','DERIVED AJCC-7 M','DERIVED AJCC-7 STAGE GRP']
threeDataset = threeDataset.drop(drop_rare2, axis=1)
print('------------Delete the rare column:',len(drop_rare2))

drop_unknown = ['EOD—OLD 13 DIGIT','EOD—OLD 2 DIGIT','EOD—OLD 4 DIGIT','CODING SYSTEM FOR EOD','PRIMARY BY INTERNATIONAL RULES',
                'ICCC SITE RECODE ICD-O-3/WHO 2008','ICCC SITE REC EXTENDED ICD-O-3/WHO 2008','AYA SITE RECODE/WHO 2008','SEER HISTORIC STAGE A','SUMMARY STAGE 2000 (1998+)'] #'FIRST MALIGNANT PRIMARY INDICATOR'
threeDataset = threeDataset.drop(drop_unknown, axis=1)
print('------------Delete the unknown column:',len(drop_unknown))

drop_important_recode = ['HISTOLOGY RECODE—BRAIN GROUPINGS',
                         'NHIA DERIVED HISPANIC ORIGIN']
threeDataset = threeDataset.drop(drop_important_recode, axis=1)
print('------------Delete the important_recode column:',len(drop_important_recode))

drop_88_03 = ['T VALUE - BASED ON AJCC 3RD (1988-2003)','N VALUE - BASED ON AJCC 3RD (1988-2003)',
              'M VALUE - BASED ON AJCC 3RD (1988-2003)','SEER MODIFIED AJCC STAGE 3RD ED (1988-2003)']
threeDataset = threeDataset.drop(drop_88_03, axis=1)
print('------------Delete the drop_88_03 column:',len(drop_88_03))

outputData0 = threeDataset.loc[threeDataset['YEAR OF DIAGNOSIS'] > 2003] # extract the 2004~forward
outputData0 = outputData0.reset_index(drop=True) # reset the index (because contain origin index)

outputData0 = outputData0.rename(columns={'CS TUMOR SIZE':'TUMOR_SIZE','CS LYMPH NODES':'LYMPH NODES','CS EXTENSION':'EXTENSION'})
outputData0 = outputData0.drop(['EOD—TUMOR SIZE','EOD—LYMPH NODE INVOLV','EOD—EXTENSION'], axis=1)

outputData0.to_csv('2010+.csv' ,index = False ,encoding='utf-8') 
print('2010-2014 Volume=',len(outputData0),'columns count',len(outputData0.columns))






