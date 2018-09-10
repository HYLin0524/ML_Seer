# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:32:02 2018

@author: HIM_LAB
"""

import csv
import pandas as pd
import numpy as np

'''build a patient_id table'''
root_path = 'C:\\Users\\HIM_LAB\\Desktop\\SEER DATA dataset\\SEERDATA(unzip\\SEER_1973_2014_CUSTOM_TEXTDATA\\incidence\\'
sub_path = ['yr1973_2014.seer9\\','yr1992_2014.sj_la_rg_ak\\','yr2000_2014.ca_ky_lo_nj_ga\\']
file = [['Breast1','Breast2','Breast3'],['Colon1','Colon2','Colon3'],['Digestive1','Digestive2','Digestive3'],
        ['Female1','Female2','Female3'],['Lymphoma1','Lymphoma2','Lymphoma3'],['Male1','Male2','Male3'],
        ['Respiratory1','Respiratory2','Respiratory3'],['Urinary1','Urinary2','Urinary3'],['Other1','Other2','Other3']]
path = {'Breast1':root_path + sub_path[0] +'BREAST.TXT', 'Breast2':root_path + sub_path[1] +'BREAST.TXT',
        'Breast3':root_path + sub_path[2] +'BREAST.TXT',
        'Colon1':root_path + sub_path[0] +'COLRECT.TXT', 'Colon2':root_path + sub_path[1] +'COLRECT.TXT',
        'Colon3':root_path + sub_path[2] +'COLRECT.TXT', 
        'Digestive1':root_path + sub_path[0] +'DIGOTHR.TXT', 'Digestive2':root_path + sub_path[1] +'DIGOTHR.TXT',
        'Digestive3':root_path + sub_path[2] +'DIGOTHR.TXT',
        'Female1':root_path + sub_path[0] +'FEMGEN.TXT', 'Female2':root_path + sub_path[1] +'FEMGEN.TXT',
        'Female3':root_path + sub_path[2] +'FEMGEN.TXT',
        'Lymphoma1':root_path + sub_path[0] +'LYMYLEUK.TXT', 'Lymphoma2':root_path + sub_path[1] +'LYMYLEUK.TXT',
        'Lymphoma3':root_path + sub_path[2] +'LYMYLEUK.TXT',
        'Male1':root_path + sub_path[0] +'MALEGEN.TXT', 'Male2':root_path + sub_path[1] +'MALEGEN.TXT',
        'Male3':root_path + sub_path[2] +'MALEGEN.TXT',
        'Respiratory1':root_path + sub_path[0] +'RESPIR.TXT', 'Respiratory2':root_path + sub_path[1] +'RESPIR.TXT',
        'Respiratory3':root_path + sub_path[2] +'RESPIR.TXT',
        'Urinary1':root_path + sub_path[0] +'URINARY.TXT', 'Urinary2':root_path + sub_path[1] +'URINARY.TXT',
        'Urinary3':root_path + sub_path[2] +'URINARY.TXT',
        'Other1':root_path + sub_path[0] +'OTHER.TXT', 'Other2':root_path + sub_path[1] +'OTHER.TXT',
        'Other3':root_path + sub_path[2] +'OTHER.TXT',
        }


def fetch_PatientID(cancer):
    register = []
    for j in cancer['Origin']:
        register.append(j[0:8])
    return register

def fetch_C19_21(cancer):
    register = []
    for j in cancer['Origin']:
        register.append(j[42:46])
    return register

count = 0
Pid = []
alldata = []
for i in range(len(file)): #len = 9
    for k in range(len(file[i])): #len = 3
        test = pd.read_csv(path[file[i][k]], header=None)
        test.columns = ['Origin']
        alldata.append(test)
    cancer = pd.concat([alldata[0],alldata[1],alldata[2]])
    alldata = [] #清空
    Pid.append(fetch_PatientID(cancer))
#    Pid.append(fetch_C19_21(cancer))
    count += 1
    print(count)

all_ID = pd.DataFrame(Pid)
all_ID = all_ID.transpose()
all_ID.columns = ['Breast','Colon','Digestive','Female_Genital','Lymphoma',
                  'Male_Genital','Respiratory','Urinary','Other']
#all_ID.to_csv('all_cancers_patient_id.csv', index=False)