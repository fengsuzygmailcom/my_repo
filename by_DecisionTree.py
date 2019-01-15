#*************************************  load two files and trace booking id
# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
#from random import shuffle

import csv
import sys
import string
import time
#import numpy as np
import pandas as pd
import os
from pandas import *
#from pytz import timezone
from datetime import datetime



##################################  STEP 1: feature analysis
############################  1:  evaluate feature relevance

def evaluate_feature_relevance():
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    
    #pd_vars= list(data_raw.columns)
    #pd_vars.remove('LSP Name')
    pd_vars=['r_quartile','f_quartile','m_quartile']
    
    data=data_raw[pd_vars] #drop the lsp name
    
    
    for var in pd_vars:
        
        #make a copy
        new_data= data.drop([var], axis=1)
        
        #create feature series
        #new_features=pd.DataFrame(data.loc[:,var])
        new_feature=data.loc[:,var]
        
        #split the data into training and testing
        X_train, X_test, y_train, y_test= train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
        
        #instantiate
        dtr=DecisionTreeRegressor(random_state=42)
        
        #fit
        dtr.fit(X_train, y_train)
        
        #return R^2
        score=dtr.score(X_test, y_test)    
        print('R2 score for {} as dependent variable: {}'.format(var, score))
        

########################## 2: visualize fueature distributions
def visualize_feature_distributions():
    data=data_raw[['r_quartile','f_quartile','m_quartile']] #
    
    # scatter matrix
    pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
    display(data.corr())
    
    # Correlation Matrix
    import matplotlib.pyplot as plt 
    
    def plot_corr(df,size=10):
        '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    
        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot'''
    
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        cax = ax.matshow(df, interpolation='nearest')
        ax.matshow(corr)
        fig.colorbar(cax)
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);
    
    plot_corr(data)





#corr
def corr(data):
    from sklearn import preprocessing     
    scaler=preprocessing.MinMaxScaler()
    X=scaler.fit_transform(data)
    X=pd.DataFrame(X)
    display(X.corr())
    return X

#
def minMaxScaler(data):
    from sklearn import preprocessing 
    scaler=preprocessing.MinMaxScaler()
    X=scaler.fit_transform(data)
    return X

# Produce a scatter matrix for each pair of newly-transformed features
def scatter_matrix():
    from sklearn import preprocessing 
    data=data_raw[['r_quartile','f_quartile','m_quartile']]
    scaler=preprocessing.MinMaxScaler()
    X=scaler.fit_transform(data)
    X=pd.DataFrame(X)
    pd.scatter_matrix(X, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
    display(X.corr())
    return X




# decision tree regressor
def DT_regressor(X_train, X_test, y_train, y_test):  
    print 'DT_regressor' 
    from sklearn import tree
    clf=tree.DecisionTreeRegressor(random_state=42)
    clf.fit(X_train, y_train)
    y_predict=clf.predict(X_test)
    evaluation(y_test, y_predict)
    print y_predict
    print 'Mean Accuracy is'
    print clf.score(X_test, y_test) #Returns the mean accuracy on the given test data and labels.



 

#use decision tree classifier
def DT_classifier(X_train, X_test, y_train, y_test):
    print 'DT_classifier'
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_predict= clf.predict(X_test)
    y_proba= clf.predict_proba(X_test)
    evaluation(y_test, y_predict)
    print 'Mean Accuracy is'
    print clf.score(X_test, y_test) #Returns the mean accuracy on the given test data and labels.
    from sklearn.metrics import accuracy_score
    print accuracy_score(y_test, y_predict)




def evaluation(y_test, y_predict):
    from sklearn.metrics import mean_squared_error #MSE
    from sklearn.metrics import mean_absolute_error #MAE
    from sklearn.metrics import r2_score#R square
    from sklearn import metrics
    
    #
    print 'MSE   '
    print mean_squared_error(y_test,y_predict)
    
    #root_mean_squared_error=mean_squared_error**(0.5)

    print 'MAE   '
    print mean_absolute_error(y_test,y_predict)
    
    print 'R2 '
    print r2_score(y_test,y_predict)
    


        
def visulization(clf):
    tree.export_graphviz(clf, out_file='tree.dot')
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source('tree.dot') 
    graph.render("iris") 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph 

    
def voting_classifier(X,y):
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from itertools import product
    from sklearn.ensemble import VotingClassifier
    
    ## Loading some example data
    #iris = datasets.load_iris()
    #X = iris.data[:, [0,2]]
    #y = iris.target
    
    # Training classifiers
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
    
    #clf1 = clf1.fit(X,y)
    #clf2 = clf2.fit(X,y)
    #clf3 = clf3.fit(X,y)
    #eclf = eclf.fit(X,y)   
    print 'voting classifier'
    display(cross_val_score(clf1, X, y))    
    display(cross_val_score(clf2, X, y))  
    #display(cross_val_score(clf3, X, y))  
    #display(cross_val_score(eclf, X, y))     

def split_by_KFold(X, Y, k):
    X=np.array(X)
    Y=np.array(Y)
    from sklearn.model_selection import KFold
    #X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    #y = np.array([1, 2, 3, 4])
    #kf = KFold(n_splits=2)
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)    
    print(kf)      
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]    
 
def random_result():
    from sklearn import metrics
    yy=np.random.randint(2, size=len(y_test))
    print 'random accuracy'
    print metrics.accuracy_score(y_test, yy)

def my_cross_val_score(clf, X, Y):
    print 'cross validation score'
    from sklearn.model_selection import cross_val_score
    score= cross_val_score(clf, X, Y, cv=10)
    #print score
    #print score.mean()
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                              

def my_cross_val_score_2(clf, X, Y):
    from sklearn.cross_validation import cross_val_score # K折交叉验证模块 
    scores=cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
    #print scores
    #print (scores.mean())
    print 'cross validation score'
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
##plot two class
def plot_roc(y_test, result, str_name):
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds=metrics.roc_curve(y_test, result, pos_label=None, sample_weight=None, drop_intermediate=True)
    
    roc_auc = auc(fpr, tpr)
    #plot roc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(str_name))
    plt.legend(loc="lower right")
    plt.show()    


def use_svm(X, Y):
    from sklearn import svm
    clf=svm.SVC(C=1, kernel='linear')
    clf.fit(X,Y)
    print 'SVM'
    display(my_cross_val_score(clf, X, Y))
    
    
if __name__ == '__main__':
    
    data_raw=read_csv('.csv', sep=',', header=0)
    data=pd.get_dummies(data_raw[['dayofweek','schedule hour_mins','Mall ID_x']])
    X=corr(data)
    Y=data_raw['delay label'] 
      
    #data_raw=read_csv('.csv', sep =',',  header=0)
    #X=corr(data_raw[['r_quartile','f_quartile','m_quartile']])
    #Y=data_raw['rank']   

    
    from sklearn.cross_validation import train_test_split    
    X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.25, random_state=42)    

    from sklearn import tree
    DT_classifier(X_train, X_test, y_train, y_test)
    DT_regressor(X_train, X_test, y_train, y_test)
    
    print 'DecisionTreeClassifier'
    display(my_cross_val_score(tree.DecisionTreeClassifier(), X, Y))
    
    print 'DecisionTreeRegressor'
    display(my_cross_val_score(tree.DecisionTreeRegressor(), X, Y))
    
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    #X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)
    print 'random forestClassifier'
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    display(my_cross_val_score(clf, X, Y))

    print 'ExtraTreesClassifier'
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    display(my_cross_val_score(clf, X, Y))
    
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(X,Y)
    clf.feature_importances_ 
    print 'AdaBoostClassfier'
    display(my_cross_val_score(clf, X, Y))

    
    from sklearn.ensemble import GradientBoostingClassifier
    clf=GradientBoostingClassifier(n_estimators=10,learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(X,Y)
    clf.feature_importances_  
    print 'GradientBoostingClassifier'
    display(my_cross_val_score(clf, X, Y))
    
    from sklearn.neighbors import KNeighborsClassifier     
    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(X,Y)
    print 'KNeighborsClassifier'
    display(my_cross_val_score(clf, X, Y))
    
        
    #visulization()
    #voting_classifier(X, Y)
    print 'the end'
    random_result




