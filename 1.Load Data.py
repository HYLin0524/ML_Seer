# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:10:39 2017

@author: HIM_LAB
"""

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 

#data = pd.read_csv('COLRECT2000_2014.TXT' ,sep=" ", error_bad_lines=False, header=None)

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

'''Read DataSet'''
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

##---以上必做---以下根據需求實作##

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

drop_null = ['CS SITE-SPECIFIC FACTOR 11','CS SITE-SPECIFIC FACTOR 12','CS SITE-SPECIFIC FACTOR 13',
             'CS SITE-SPECIFIC FACTOR 15','CS SITE-SPECIFIC FACTOR 25','CS SITE-SPECIFIC FACTOR 16']
threeDataset = threeDataset.drop(drop_null, axis=1)
print('------------Delete the decisive null columns: ', len(drop_decisive), len(drop_null))

'''delete the code columns'''
drop_code = ['HISTOLOGY (92-00) ICD-O-2','BEHAVIOR (92-00) ICD-O-2','DERIVED AJCC—FLAG','CS VERSION INPUT ORIGINAL',
             'CS VERSION DERIVED','CS VERSION INPUT CURRENT','AGE RECODE <1 YEAR OLDS','RECODE ICD-O-2 TO 9','RECODE ICD-O-2 TO 10',
             'CS SCHEMA V0204+','INSURANCE RECODE (2007+)','SURVIVAL MONTHS FLAG','SITE RECODE ICD-O-3/WHO 2008','LYMPHOMA SUBTYPE RECODE/WHO 2008']
threeDataset = threeDataset.drop(drop_code, axis=1)
print('------------Delete the code column: ',len(drop_code))

'''delete the rarely columns''' #橫跨年限極少之欄位
drop_rare = ['RX SUMM—SURG TYPE','RX SUMM—SCOPE REG 98-02','RX SUMM—SURG OTH 98-02','SEER SUMMARY STAGE 1977',
             'SEER SUMMARY STAGE 2000','DERIVED AJCC-7 T','DERIVED AJCC-7 N','DERIVED AJCC-7 M','DERIVED AJCC-7 STAGE GRP','DERIVED HER2 RECODE (2010+)',
             'CS METS AT DX-BONE','CS METS AT DX-BRAIN','CS METS AT DX-LIVER','CS METS AT DX-LUNG','DERIVED SS1977','DERIVED SS2000',
             'AJCC STAGE 3RD EDITION (1988-2003)','CS METS EVAL','CS LYMPH NODES EVAL','CS TUMOR SIZE EXT/EVAL','CS METS AT DX','LYMPH VASCULAR INVASION']
threeDataset = threeDataset.drop(drop_rare, axis=1)
print('------------Delete the rare column:',len(drop_rare))

drop_unknown = ['EOD—OLD 13 DIGIT','EOD—OLD 2 DIGIT','EOD—OLD 4 DIGIT','CODING SYSTEM FOR EOD','PRIMARY BY INTERNATIONAL RULES',
                'FIRST MALIGNANT PRIMARY INDICATOR','ICCC SITE RECODE ICD-O-3/WHO 2008','ICCC SITE REC EXTENDED ICD-O-3/WHO 2008','AYA SITE RECODE/WHO 2008',
                'SEER HISTORIC STAGE A','SUMMARY STAGE 2000 (1998+)'] #'FIRST MALIGNANT PRIMARY INDICATOR'
threeDataset = threeDataset.drop(drop_unknown, axis=1)
print('------------Delete the unknown column:',len(drop_unknown))

drop_important_recode = ['BEHAVIOR CODE ICD-O-3','HISTOLOGIC TYPE ICD-O-3','HISTOLOGY RECODE—BRAIN GROUPINGS',
                         'RACE RECODE (WHITE, BLACK, OTHER)','RACE / ETHNICITY','NHIA DERIVED HISPANIC ORIGIN']
threeDataset = threeDataset.drop(drop_important_recode, axis=1)
print('------------Delete the important_recode column:',len(drop_important_recode))

drop_2004 = ['CS SITE-SPECIFIC FACTOR 1','CS SITE-SPECIFIC FACTOR 2','CS SITE-SPECIFIC FACTOR 3','CS SITE-SPECIFIC FACTOR 4','CS SITE-SPECIFIC FACTOR 5',
              'CS SITE-SPECIFIC FACTOR 6','CS SITE-SPECIFIC FACTOR 7','CS SITE-SPECIFIC FACTOR 8','CS SITE-SPECIFIC FACTOR 9','CS SITE-SPECIFIC FACTOR 10']
threeDataset = threeDataset.drop(drop_2004, axis=1)
print('------------Delete the 2004+ column:',len(drop_2004))

colLenght = colLenght - len(drop_useless) - len(drop_decisive) - len(drop_null) - len(drop_code) - len(drop_rare) - len(drop_unknown) - len(drop_important_recode) - len(drop_2004)
print("remain columns:", colLenght)

'''fetch 2010~2014'''
#outputData0 = threeDataset.loc[threeDataset['YEAR OF DIAGNOSIS'] > 2009] # extract the 2004~forward
#outputData0 = outputData0.reset_index(drop=True) # reset the index (because contain origin index)
#outputData0.to_csv('2010+.csv' ,index = False ,encoding='utf-8') 
#print('2010-2014 Volume=',len(outputData0),'columns count',len(outputData0.columns))

'''fetch 2004~2014'''
outputData = threeDataset.loc[threeDataset['YEAR OF DIAGNOSIS'] > 2004] # extract the 2004~forward
outputData = outputData.reset_index(drop=True) # reset the index (because contain origin index)
#outputData.to_csv('2004+.csv' ,index = False ,encoding='utf-8') 
print('2004-2014 Volume=',len(outputData),'columns count',len(outputData.columns))

'''fetch 1988~2003'''
outputData2 = threeDataset.loc[(threeDataset["YEAR OF DIAGNOSIS"] >= 1988) & (threeDataset["YEAR OF DIAGNOSIS"] < 2004)]
outputData2 = outputData2.reset_index(drop=True)
#outputData2.to_csv('1988-2003.csv' ,index = False ,encoding='utf-8') #2004~2014資料
print('1988-2003 Volume=',len(outputData2),'columns count',len(outputData2.columns))

del threeDataset

#change EOD tumor size to CS format
outputData2['EOD—TUMOR SIZE'][outputData2['EOD—TUMOR SIZE']=='001'] = '990'
outputData2['EOD—TUMOR SIZE'][outputData2['EOD—TUMOR SIZE']=='990'] = '989'
outputData2['EOD—TUMOR SIZE'][outputData2['EOD—TUMOR SIZE']=='002'] = '992'
#change CS Lymph node to EOD format
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES'].str.find('0') == 0] = 0
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES'].str.find('1') == 0] = 1
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES'].str.find('2') == 0] = 2
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES']=='300'] = 3
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES'].str.find('4') == 0] = 7
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES']=='800'] = 8
outputData['CS LYMPH NODES'][outputData['CS LYMPH NODES']=='999'] = 9

'''combine EOD(1988-2003) & CS(2004+)''' # TUMOR SIZE + LYMPH NODES
outputData = outputData.rename(columns={'CS TUMOR SIZE':'TUMOR_SIZE','CS LYMPH NODES':'LYMPH NODES'})
outputData = outputData.drop(['EOD—TUMOR SIZE','EOD—LYMPH NODE INVOLV'], axis=1)
outputData2 = outputData2.rename(columns={'EOD—TUMOR SIZE':'TUMOR_SIZE','EOD—LYMPH NODE INVOLV':'LYMPH NODES'})
outputData2 = outputData2.drop(['CS TUMOR SIZE','CS LYMPH NODES'], axis=1)
print('------already combine CS+EOD size&extension------,lenght -2')

# recode TNM Stage (3rd T:0 to 6th T:05)
outputData2['T VALUE - BASED ON AJCC 3RD (1988-2003)'][outputData2['T VALUE - BASED ON AJCC 3RD (1988-2003)']=='0'] = '05'
# recode 6th Stage to 3rd Stage
outputData['DERIVED AJCC-6 STAGE GRP'][outputData['DERIVED AJCC-6 STAGE GRP'].str.find('3') == 0] = '20'
outputData['DERIVED AJCC-6 STAGE GRP'][outputData['DERIVED AJCC-6 STAGE GRP'].str.find('5') == 0] = '30'
outputData['DERIVED AJCC-6 STAGE GRP'][outputData['DERIVED AJCC-6 STAGE GRP'].str.find('7') == 0] = '40'

# combine 3rd 6th Stage
outputData = outputData.rename(columns={'DERIVED AJCC-6 STAGE GRP':'Combine_Stage',
                                        'DERIVED AJCC-6 T':'Combine_T','DERIVED AJCC-6 N':'Combine_N','DERIVED AJCC-6 M':'Combine_M',})
outputData = outputData.drop(['T VALUE - BASED ON AJCC 3RD (1988-2003)','N VALUE - BASED ON AJCC 3RD (1988-2003)',
                              'M VALUE - BASED ON AJCC 3RD (1988-2003)','SEER MODIFIED AJCC STAGE 3RD ED (1988-2003)'], axis=1)

outputData2 = outputData2.rename(columns={'SEER MODIFIED AJCC STAGE 3RD ED (1988-2003)':'Combine_Stage',
                                        'T VALUE - BASED ON AJCC 3RD (1988-2003)':'Combine_T',
                                        'N VALUE - BASED ON AJCC 3RD (1988-2003)':'Combine_N',
                                        'M VALUE - BASED ON AJCC 3RD (1988-2003)':'Combine_M'})
outputData2 = outputData2.drop(['DERIVED AJCC-6 T','DERIVED AJCC-6 N','DERIVED AJCC-6 M','DERIVED AJCC-6 STAGE GRP'], axis=1)
print('------combine 3rd 6th Stage completely------,lenght -4')
data1988_2014 = pd.concat([outputData,outputData2], join='inner')
remain_columns = list(data1988_2014.columns)

'''check the year indeed between 04~14'''
#for i in range(len(outputData)):
#    if (outputData['YEAR OF DIAGNOSIS'][i]<2003):
#        print(np.where(outputData['YEAR OF DIAGNOSIS'][i]))
'''save file to csv'''
data1988_2014.to_csv('colon_ThreeSubset_radiation_1988.csv' ,index = False ,encoding='utf-8') #2004~2014資料
