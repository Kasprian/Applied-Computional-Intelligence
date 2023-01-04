# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:57:12 2021

@author: Pjoter
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz

clfs = {
    '_momentum': MLPClassifier(hidden_layer_sizes=(16,),max_iter=500, momentum=0),
    'no_momentum': MLPClassifier(hidden_layer_sizes=(100,),max_iter=500, momentum=0.9),
    'sgd_momentum': MLPClassifier(hidden_layer_sizes=(16,),solver='sgd',max_iter=500, momentum=0),
    'sgd_no_momentum': MLPClassifier(hidden_layer_sizes=(100,),solver='sgd',max_iter=500, momentum=0.9),
    'lbfgs_momentum': MLPClassifier(hidden_layer_sizes=(16,),solver='lbfgs',max_iter=500, momentum=0),
    'lbfgs_no_momentum': MLPClassifier(hidden_layer_sizes=(100,),solver='lbfgs',max_iter=500, momentum=0.9),
}

def SimpleModel(X_train, X_val, y_train, y_val):
    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_val)
        print(accuracy_score(y_val, prediction))
        print(precision_score(y_val, prediction,pos_label=1))
        print(recall_score(y_val, prediction))
        
def createPreviousLoadData(raw_data, n):
    x = raw_data["Load"].to_numpy()
    req = raw_data.drop('Falha', axis=1).to_numpy()
    for i in range(n):
        requestN =np.roll(req[:,0],i+1)
        print(requestN.shape)
        x = np.vstack((x,requestN))
        
    return x.T
        
        
    
def main():
    raw_data = pd.read_csv("ACI21-22_Proj1IoTGatewayCrashDataset.csv")
    X = raw_data.drop('Falha', axis=1).to_numpy()
    y = raw_data['Falha'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False) # 0.25 x 0.8 = 0.2
    print("x",X_train[:,1])
    #SimpleModel(X_train, X_val, y_train, y_val)
    
    load = raw_data["Load"].to_numpy()
    request1 = np.roll(X[:,0],1)
    request2 = np.roll(X[:,0],2)
    X2 = np.concatenate(([load],[request1],[request2]),axis=0)
    print(X2.T)
    l= createPreviousLoadData(raw_data, 3)
    print(l)

    #new_column = np.concatenate((np.array([0,0]), X[2:]),axis=0)
    #print(new_column)
   # X2 = np.append(X2,new_column, axis = 1)
    
    

if __name__ == "__main__":
    main()