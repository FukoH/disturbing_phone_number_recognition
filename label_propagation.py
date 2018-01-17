# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:54:14 2017

@author: eda
"""

import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import StratifiedShuffleSplit

# 读入
positive = pd.read_csv('./positive.csv',encoding='gbk')
positive['label']= -1
negative = pd.read_csv('./negative.csv',encoding='gbk')
negative['label'] = -1
#去掉
to_drop = ['手机号码','漫游通话时长（单位：秒）','漫游通话次数']
l=[]
for i in range(25):
    l.append(str(i)+'点通话')  
    
to_drop.extend(l)

positive = positive.drop(to_drop,axis = 1 )
negative = negative.drop(to_drop,axis = 1 )

#过滤FN和FP
positive['avg']= positive.iloc[:, 17:27].sum(axis=1)/10
negative['avg']= negative.iloc[:, 17:27].sum(axis=1)/10
#positive = positive[positive['avg']>20]
positive.loc[positive['avg']>20,'label'] = 1
#negative = negative.sort_values('avg').iloc[:int(len(negative)*0.3),:]
negative.loc[negative['avg']<3,'label'] = 0
#negative.sort_values('avg').iloc[:int(len(negative)*0.3),'label'] = 0

df = pd.concat([positive,negative]) 

df['次均通话时长'] = df['月通话时长（单位：秒）']/df['月通话次数']
#df['次均漫游时长'] = df['漫游通话时长（单位：秒）']/df['漫游通话次数']
df = df.fillna(0)
df.replace([np.inf, -np.inf], 0)

#positive['次均通话时长'] = positive['月通话时长（单位：秒）']/positive['月通话次数']
#positive['次均漫游时长'] = positive['漫游通话时长（单位：秒）']/positive['漫游通话次数']


#negative['次均通话时长'] = negative['月通话时长（单位：秒）']/negative['月通话次数']
#negative['次均漫游时长'] = negative['漫游通话时长（单位：秒）']/negative['漫游通话次数']

#df = pd.concat([positive,negative]) 

#df = df.fillna(0)
y = df['label']

    
call_times=[]
for i in range(0,25):
    call_times.append(str(i)+'点通话次数')
    
df = df[['次均通话时长','在网时长（单位：秒）','当月活跃基站个数','交往圈数量','avg']]
#df= df.drop(call_times,axis = 1 )
#df = df[['月累计短信发送数量','月累计流量使用情况（单位：字节）']]


print('start pca...')
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(df)
reduced_X_1 = reduced_X[:,0]

reduced_X_2 = reduced_X[:,1]

conponent = pd.DataFrame({'p1':reduced_X_1,'p2':reduced_X_2,'label':y})

X = conponent[['p1','p2']]
y = conponent['label']
sss = StratifiedShuffleSplit(n_splits=3, train_size=0.0025,test_size=0.0025, random_state=0)
sss.get_n_splits(X, y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    label_prop_model = LabelPropagation(max_iter=5000)
    label_prop_model.fit(X_train, y_train)
    print(label_prop_model.score(X_test,y_test))
    
    



