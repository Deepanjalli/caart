import numpy as np
from numpy import *
import sys
import os
import xlrd
import pandas as pd
import graphviz 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import timeit

start=timeit.default_timer()
data=pd.ExcelFile('wave500k.xls')
edata=data.parse(0)
fedata=np.array(edata)
clf = tree.DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=4,min_samples_split=4,min_samples_leaf=1)
#clf = tree.DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=8,min_samples_split=6,min_samples_leaf=3)
#clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=4,min_samples_split=4,min_samples_leaf=1)
#clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=8,min_samples_split=6,min_samples_leaf=3)


X=fedata[:,1:]
Y=fedata[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf=clf.fit(X_train,y_train)
clf.predict(X_test)
pr_lab=clf.predict(X_test)
acc=(float(sum(pr_lab==y_test))/len(y_test))*100
print acc
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("caart")
eval_mat=precision_recall_fscore_support(y_test, pr_lab, average='weighted')
eval_mat=np.array(eval_mat)
eval_mat=eval_mat.astype(float)
eval_mat=eval_mat*100
stop=timeit.default_timer()
time=stop-start
print time
with open('eval_mat.txt','ab') as f:
 f.write(b'\n')
 np.savetxt(f, eval_mat,delimiter=',',fmt='%2f', newline='')

