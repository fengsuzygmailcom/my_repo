#*************************************  load two files and trace booking id
# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
#from random import shuffle

import csv
import sys
import string
import time
import numpy as np
import pandas as pd
import os
from pandas import *
from datetime import datetime

path='.csv'

file='customer_rfmTable.csv' # booking 
file_2='booking_inout_concat_by_PYTHON.csv' # booking 

#data_raw=read_csv(path+file, sep =',',  header=0)
data_raw_2=read_csv(path+file_2, sep=',', header=0)

#dr=data_raw  #<== rfm table
df3=data_raw_2 #<== concat data


####   load concat data
def prepare_data_for_delay_forcasting(df3):
    df3.isnull().sum(axis=0)
    
    ### calculate recency
    import datetime as dt
    NOW= dt.datetime(2018, 04, 01)
    df3['In Time'] = pd.to_datetime(df3['In Time'])
    df3['Out Time'] = pd.to_datetime(df3['Out Time'])
    df3['Schedule Time'] = pd.to_datetime(df3['Schedule Time'])
    
    # mon~sun = 0, ~, 6
    df3['dayofweek']=df3['In Time'].apply(lambda x: x.dayofweek)
    
    ## add one col (timedelta64, Y=year, W=week, D=day, M=month, m=minutes, s=second, h=hour)
    df3['in mall time']=(df3['Out Time']-df3['In Time'] )/ np.timedelta64(1, 'm')
    
    ##calculate delay
    df3['delivery delay']=(df3['In Time']-df3['Schedule Time'] )/ np.timedelta64(1, 'm')
    
    ## arrive early means no delay
    df3['delivery delay']=df3['delivery delay'].apply(lambda x: 0 if x<0 else x) 
    
    #filter out delay> 10 hours i.e. 600mins
    df3=df3[df3['delivery delay']<=600]
    df3['delay label']=df3['delivery delay'].apply(lambda x: 0 if x<=2 else 1)
    
    # hour 24 hours
    df3['in hour']=df3['In Time'].apply(lambda x: x.hour)
    df3['in hour_mins']=df3['In Time'].apply(lambda x: x.hour*60+x.minute)
    df3['schedule hour']=df3['Schedule Time'].apply(lambda x: x.hour)
    df3['schedule hour_mins']=df3['Schedule Time'].apply(lambda x: x.hour*60+x.minute)
    
    #filter nan
    df3.isnull().sum(axis=0)
    df3=df3[pd.notnull(df3['in mall time'])] # find_out: when it has lsp name, it has lsp id as well.
    
    df3.to_csv('.csv', index=False, header=True)
    return df3

df3=prepare_data_for_delay_forcasting(df3) 
       
########################### STEP 1: processing data scaling
##
def plot_delay(data):
    close('all')
    for v in ['TM', 'BM','IMM','WG','BPP']:
        a=data[data['Mall ID_x']==v]['delay label'].value_counts()
        fig, ax = plt.subplots()
        a.plot.bar()
        #plt.show()
        plt.title(str(v))

###############  if use  normalization
def normalize(data):
    from sklearn import preprocessing
    X=sklearn.preprocessing.normalize(data, norm= 'l2')
    
    #conver to data frame
    X=pd.DataFrame(X)
    display(X.corr())
    return X

###############  if  use logarithm to normalize
def normalize_log(data):
    ## TODO: Scale the data using the natural logarithm
    log_data = np.log(data)    
    # Produce a scatter matrix for each pair of newly-transformed features
    pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');    
    display(log_data.corr())
    return log_data

############################    function to evaluate feature relevance
def evaluate_feature_relevance(data, pd_vars):
    #data dataframe
    #pd_var eg:pd_vars=['r_quartile','f_quartile','m_quartile']        
    for var in pd_vars:
        
        #make a copy
        new_data= data.drop([var], axis=1)
        
        #create feature series
        #new_features=pd.DataFrame(data.loc[:,var])
        new_feature=data.loc[:,var]
        
        #split the data into training and testing
        X_train, X_test, y_train, y_test= train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
        
        #instantiate
        from sklearn.tree import DecisionTreeRegressor
        dtr=DecisionTreeRegressor(random_state=42)
        
        #fit
        dtr.fit(X_train, y_train)
        
        #return R^2
        score=dtr.score(X_test, y_test)    
        print('R2 score for {} as dependent variable: {}'.format(var, score))
        

####
def hour_N_delay(data):
    a=data.groupby('in hour').agg({'in hour': lambda x: len(x), 
                    'delivery delay': lambda x: x.sum()/len(x) })
                    
    b=data.groupby('schedule hour').agg({'schedule hour': lambda x: len(x), 
                    'delivery delay': lambda x: x.sum()/len(x) })
    a.plot.bar() 
    b.plot.bar()



#  method 1: dummy the data and use sklearn.linear_model.LogisticRegression
def dummy_LogisticRegression(df3):
    data=df3[['in hour','dayofweek','Mall ID_x']]    
    data_dummy_feature=pd.get_dummies(data)
    data_dummy_label=df3['delay label']
    
    # regression 
    from sklearn import datasets, linear_model
    from sklearn.cross_validation import train_test_split
    
    X_train,X_test,y_train, y_test= train_test_split(data_dummy_feature, data_dummy_label, test_size=0.20, random_state=42)
    MODEL = linear_model.LogisticRegression().fit(X_train, y_train)
    ##predict new samples
    result=MODEL.predict(X_test)
    # score
    score=MODEL.score(X_test, y_test) 
    print 'accuracy='+str(score) #0.7674
    
    plot_roc(y_test, result,'  logistic_Regression')
    #plot_roc_multiClass(y_test, result)
    return result

    
def plot_roc_multiClass(y_test, result):    
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds=metrics.roc_curve(y_test, result, pos_label=None, sample_weight=None, drop_intermediate=True)
    fpr=dict() #false positive rate
    tpr=dict() #true positive rate
    roc_auc=dict()
    
    from sklearn.preprocessing import label_binarize
    yy=label_binarize(y_test, classes=[0, 1])
    num_class=yy.shape[1]
    
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_test, result)
        roc_auc[i] = auc(fpr[i], tpr[i])
        #plot roc
        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


## pca
def pca(X):
    import numpy as np
    from sklearn.decomposition import PCA
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
    print(pca.explained_variance_ratio_)  
    print(pca.singular_values_)  

    
    
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


#  method 2: dummy the data and use sklearn.linear_model.LogisticRegression
def dummy_multiMethods(df3):
    data=df3[['in hour','dayofweek','Mall ID_x']]    
    data_dummy_feature=pd.get_dummies(data)
    data_dummy_label=df3['delay label']
    
    # method_1: regression 
    from sklearn import datasets, linear_model
    from sklearn.cross_validation import train_test_split
        
    X_train,X_test,y_train, y_test= train_test_split(data_dummy_feature, data_dummy_label, test_size=0.20, random_state=42)
    MODEL = linear_model.LogisticRegression().fit(X_train, y_train)
    ##predict new samples
    result=MODEL.predict(X_test)
    # score
    score=MODEL.score(X_test, y_test) 
    print 'accuracy='+str(score) #0.7674   
    plot_roc(y_test, result, 'lr')
    
    # method_2:  random
    from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)
    from sklearn.pipeline import make_pipeline
    
    n_estimator = 10
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,random_state=0)
    rt_lm = linear_model.LogisticRegression()
    pipeline =make_pipeline(rt, rt_lm)  
    pipeline.fit(X_train, y_train)  
    y_pred_randomTree = pipeline.predict_proba(X_test)[:, 1]
    plot_roc(y_test, y_pred_randomTree,'randomTrees')
    score_RandomTrees=pipeline.score(X_test, y_test) 
    print 'randomTree accuracy='+str(score_RandomTrees)
    
    # method_3: Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    from sklearn.preprocessing import OneHotEncoder
    rf_enc = OneHotEncoder()
    rf_lm = linear_model.LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train)), y_train)    
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    plot_roc(y_test, y_pred_rf_lm,'randomForest')
    
    
    ## method_4: gradient boosting 
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_lm = linear_model.LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train)[:, :, 0]), y_train)  
    y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    #y_pred_grd_lm = grd_lm.predict(grd_enc.transform(grd.apply(X_test)[:, :, 0]))
    #a=y_pred_grd_lm==y_test 
    #accuraccy=a.sum()*1.0/len(a)
    plot_roc(y_test, y_pred_grd_lm,'Logistic regression')
    
    
    # method_5: The gradient boosted model by itself
    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    plot_roc(y_test, y_pred_grd,'Pure_GradientBoosting')
    
    
    # method_6: The random forest model by itself
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    plot_roc(y_test, y_pred_rf,'Pure_randomForest')
                            
                                                                


# use
result=dummy_regression(df3) # use multi regression 

#
#data=dr[['r_quartile','f_quartile','m_quartile']]
#data=dr[['ave_in_mall_time_monetary_cost','frequency_trans_interval','monetary_sum_trans_num','ave_delay_minutes','RFMScore_sum']]



#data=df3[['in hour','dayofweek','delivery delay','in mall time','delay label', 'LSP Name']]
data=df3[['in hour','dayofweek','delay label', 'LSP Name','Mall ID_x']]
hour_N_delay(data)
X=normalize(data)


data_dummy_feature=pd.get_dummies(data)
features=data_dummy_feature.columns
evaluate_feature_relevance(data_dummy_feature,features)


###########################  STEP 2: use cross validation
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt



############# method_1: Create linear regression object
#split the data into training and testing
from sklearn.cross_validation import train_test_split

#X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.25, random_state=42)
X_train,X_test= train_test_split(X, test_size=0.20, random_state=42)
MODEL = linear_model.LinearRegression().fit(X_train[[3]], X_train[4])
##predict new samples
result=MODEL.predict(X_test[[3]])
# plot
y=X_test[4]
close('all')
fig, ax = plt.subplots()
ax.scatter(y, result, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



############# method_2: cross_val_predict
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(linear_model.LinearRegression(), X[[0]], X[4], cv=10)
# plot
y=X[4]
fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
close('all')
plt.show()
