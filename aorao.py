# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:54:14 2017

@author: eda
"""

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression


# 读入
positive = pd.read_csv('./positive_2.csv',encoding='gbk')
positive['label']= 1
negative = pd.read_csv('./negative_2.csv',encoding='gbk')
negative['label'] = 0
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
positive = positive[positive['avg']>5]
negative = negative.sort_values('avg').iloc[:int(len(negative)*0.8),:]

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

    
#call_times=[]
#for i in range(0,25):
#    call_times.append(str(i)+'点通话次数')
    
df = df[['次均通话时长','在网时长（单位：月）','当月活跃基站个数','交往圈数量']]
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
sns.pairplot(conponent,hue='label')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#clf = RandomForestClassifier(n_estimators=10, max_depth=None,
#     min_samples_split=2, random_state=0)

#scores = cross_val_score(clf, X, y,cv = 5,scoring='recall')
#print(scores.mean())

clf = neighbors.KNeighborsClassifier(15, weights='distance')
#clf3 = LogisticRegression()

print('start trainning...')
#accuracy is the default scoring metric
print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5).mean())
# use AUC as scoring metric
print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc').mean())
# use recall as scoring metric
print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=5, scoring = 'recall').mean())





y_score_lr = clf.fit(X_train, y_train).predict_proba(X_test)
y_score_lr = y_score_lr[:,1]
fpr_lr, tpr_lr, threshod = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()

#file = pd.read_csv('./201712.csv',encoding='gbk')
#file['次均通话时长'] = file['月通话时长（单位：秒）']/file['月通话次数']
##df['次均漫游时长'] = df['漫游通话时长（单位：秒）']/df['漫游通话次数']
#file = file.fillna(0)
#file.replace([np.inf, -np.inf], 0)
#predict_X = file[['次均通话时长','在网时长（单位：月）','当月活跃基站个数','交往圈数量']]
#pca_Predict_X = pca.transform(predict_X)
#predict_y = clf2.predict_proba(pca_Predict_X)