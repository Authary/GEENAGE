# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:21:31 2020

@author: abazin
"""

import subprocess
import os
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from subprocess import PIPE
import exMCDM
import copy
import time
import numpy as np
from collections import OrderedDict
from scipy.stats import chi2_contingency
from sklearn import model_selection, svm, ensemble, feature_selection, linear_model, metrics,neural_network,tree
from sklearn.feature_selection import mutual_info_classif,SelectKBest,chi2,f_regression
from sklearn.naive_bayes import GaussianNB



#Loads the data from a file
#Rows are individuals, columns are features
#The values should be numerical and separated by semicolons
#The first line should be the names of features
#The first column should be the class
def loadData(f):
    data = list()
    file = open(f,"r")
    for line in file:
        row = line.split("\n")[0]
        row = row.split(";")
        data.append(list(row))
    return np.array(data)


#Selects features and reduces the dataset
def selectFeatures(data,n,method):
    data_raw = data[1:,1:].astype(float)
    data_classes = data[1:,0].astype(float)
    if method == "corr":
        selector = SelectKBest(chi2, k = n)
        selector.fit(data_raw,data_classes)
        cols = selector.get_support(indices=True)
        cols2 = [0]+[x+1 for x in cols]
        data = data[:,cols2]
        
    return data



#Creates an array equal to array X with the kth column randomly shuffled 
def shuffle(k, X):
    Y = copy.deepcopy(X)
    np.random.shuffle(Y[:, k])
    return Y


'''
Scores computation
'''


#Computes the confusion matrix from the predicted and true classes
def confusion_matrix(Y_true, Y_pred):
    M = [[0, 0], [0, 0]]
    for k in range(len(Y_true)):
        if Y_pred[k] >= 0.5:
            if Y_true[k] >= 0.5:
                M[0][0] = M[0][0] + 1
            else:
                M[0][1] = M[0][1] + 1
        else:
            if Y_true[k] >= 0.5:
                M[1][0] = M[1][0] + 1
            else:
                M[1][1] = M[1][1] + 1
    return M



#Computes the sensitivity of the classifier from the confusion matrix
def sensitivity(confusion_matrix):
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    try:
        R = TP/(TP + FN)
    except:
        R = 0
    return R



#Computes the sensitivity of the classifier from the confusion matrix
def specificity(confusion_matrix):
    FP = confusion_matrix[0][1]
    TN = confusion_matrix[1][1]
    try:
        R = TN/(TN + FP)
    except:
        R = 0
    return R



#Computes the sensitivity of the classifier from the confusion matrix
def precision(confusion_matrix):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    try:
        R = TP/(TP+FP)
    except:
        R = 0
    return R



#Computes the sensitivity of the classifier from the confusion matrix
def npv(confusion_matrix):
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    try:
        R = TN/(TN+FN)
    except:
        R = 0
    return R



#Computes the sensitivity of the classifier from the confusion matrix
def fscore(confusion_matrix):
    prec = precision(confusion_matrix)
    sens = sensitivity(confusion_matrix)
    try:
        R = 2*(prec*sens)/(prec+sens)
    except:
        R = 0
    return R



#Computes the sensitivity of the classifier from the confusion matrix
def accuracy(confusion_matrix):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    Tot_pop = TP + FP + FN + TN
    try:
        R = (TP + TN)/Tot_pop
    except:
        R = 0
    return R



#Computes the sensitivity of the classifier from the confusion matrix
def error_rate(confusion_matrix):
    return 1 - accuracy(confusion_matrix)


'''
Classifiers functions
'''


#Trains classifiers on the data and returns the impacts of the features on the values of four performance measures : sensitivity, specificity, accuracy, fscore
def train_and_weight(arg):
    (list_features, classifier, data, permutations) = arg
    n = 50
    
    R = [[0]*6]*len(list_features)
    
    data_reduced = data
    
    Classifiers = []
    
    #Train classifiers
    if classifier == "RF":
        C = ensemble.RandomForestClassifier(n_estimators=50, min_samples_leaf=1, max_features=1195, n_jobs=1)
    if classifier == "NN":
        C = neural_network.MLPClassifier(solver='adam', hidden_layer_sizes=(10,10), max_iter = 1000, activation = 'relu', batch_size = 20)
    if classifier == "NB":
        C = GaussianNB()
    if classifier == "SVM":
        C = svm.SVC(kernel='poly', degree=4, gamma='scale')
    if classifier == "DT":
        C = tree.DecisionTreeClassifier(max_depth = 5)
        
    X_train = data_reduced[1:,1:].astype(float)
    Y_train = data_reduced[1:,0].astype(float)
    X_test = data_reduced[1:,1:].astype(float)
    Y_test = data_reduced[1:,0].astype(float)
    
    try:
        C.max_features = len(list_features)
    except:
        a = 0
            
    C.fit(X_train, Y_train)
        
    Classifiers.append(C)
        
    Y_pred = C.predict(X_test)
        
    sensi0 = sensitivity(confusion_matrix(Y_test, Y_pred))
    speci0 = specificity(confusion_matrix(Y_test, Y_pred))
    accu0 = accuracy(confusion_matrix(Y_test, Y_pred))
    fscor0 = fscore(confusion_matrix(Y_test, Y_pred))
    
    
    for k in range(len(list_features)):
    
        S1 = 0
        S2 = 0
        S3 = 0
        S4 = 0
        
        sensi = 0
        speci = 0
        accu = 0
        fscor = 0
        
        X_testS = copy.deepcopy(X_test)

        for p in permutations:
                    
            X_testS[:,k] = X_test[p,k]
                    
            Y_pred = C.predict(X_testS)
    
                    
            sensi = sensitivity(confusion_matrix(Y_test, Y_pred))
            speci = specificity(confusion_matrix(Y_test, Y_pred))
            accu = accuracy(confusion_matrix(Y_test, Y_pred))
            fscor = fscore(confusion_matrix(Y_test, Y_pred))
            S1 = S1 + (sensi-sensi0)          
            S2 = S2 + (speci-speci0)
            S3 = S3 + (accu-accu0) 
            S4 = S4 + (fscor-fscor0)
           
 
        S1 = S1/len(permutations)
        S2 = S2/len(permutations)
        S3 = S3/len(permutations)
        S4 = S4/len(permutations)
        
        R[k] = [S1,S2,S3,S4]
        
        
    return accu0,R



'''
Main functions
'''


#Identifies the features that are predictive and/or discriminant of the class feature
def preDisc(data,file = None,depth = 1, n_perm = 50):
    start_time = time.time()

    knowledge = [[["Predictive_pos"],["Predictive"]],[["Predictive_neg"],["Predictive"]],[["Discriminant"],["Interesting"]],[["Predictive"],["Interesting"]]]
    
    knowlCriteria = [["Predictive_pos"],["Predictive_neg"],["Discriminant"],["Discriminant"],["Predictive_pos"],["Predictive_neg"],["Discriminant"],["Discriminant"],["Predictive_pos"],["Predictive_neg"],["Discriminant"],["Discriminant"],["Predictive_pos"],["Predictive_neg"],["Discriminant"],["Discriminant"]]
    
    list_features = np.array(data[0, 1:])
       
    print("Data : ",data.shape[0]-1," objects and ",data.shape[1]-1," features")
    
    print("Training models...")
    
    permutations = []
    L = list(range(data.shape[0]-1))
    for i in range(n_perm):
        np.random.shuffle(L)
        permutations.append(L)
   
    AccRF,MoyRF = train_and_weight((list_features, "RF", data, permutations))
    AccNB,MoyNB = train_and_weight((list_features, "NB", data, permutations))
    AccNN,MoyNN = train_and_weight((list_features, "NN", data, permutations))
    AccSVM,MoySVM = train_and_weight((list_features, "SVM", data, permutations))
            
    Result = np.hstack((np.array(MoyRF),np.array(MoyNB),np.array(MoyNN),np.array(MoySVM)))

    print("Random Forest : ",AccRF," accuracy")
    print("Naive Bayes : ",AccNB," accuracy")
    print("Neural Network : ",AccNN," accuracy")
    print("Support Vector Machine : ",AccSVM," accuracy")
    
    for i in range(Result.shape[1]):
        Result[:,i] = exMCDM.Rank(Result[:,i]*-1)
    FP = exMCDM.ParetoDepth(Result,depth)
    print("==========================")    
    print(len(FP)," important features")
    print("==========================")    
    Result = Result[FP,:]
    list_features = list_features[FP]    
    
    print("Constructing the interpretation...")
    Interp = exMCDM.constructInterpretation(Result,knowlCriteria,knowledge,depth)
    
    if file != None:
        interpReduced2File(Interp, file, list_features,knowledge)
    
    print("Done ! (",time.time()-start_time,"seconds )")
    
    return Interp, list_features

    
    
#Writes the interpretation of selected features in a file
def interp2File(Interp,f,l_f):
    file = open(f,"w")
    
    for i in range(len(Interp)):
        file.write(str(l_f[i]))
        file.write(" : ")
        first = True
        for t in Interp[i]:
            if not first:
                file.write(" ")
            first = False
            file.write(str(t))
        file.write("\n")
    
    file.close()
    
 
    #Writes the reduced interpretation of selected features in a file
def interpReduced2File(Interp,f,l_f,knowledge):
    file = open(f,"w")
    
    for i in range(len(Interp)):
        file.write(str(l_f[i]))
        file.write(" : ")
        first = True
        for t in Interp[i]:
            add = True
            for z in Interp[i]:
                if t != z:
                    if t in exMCDM.logicalClosure([z],knowledge):
                        add = False
            if add:
                if not first:
                    file.write(" ")
                first = False
                file.write(str(t))
        file.write("\n")
    
    file.close() 
    
 
    