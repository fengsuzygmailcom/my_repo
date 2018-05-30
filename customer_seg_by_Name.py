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

path='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/'
file='booking_inout_concat_by_PYTHON.csv' # booking 
#file='booking_inout_concat_by_PYTHON_LSPID.csv' # booking 

df=read_csv(path+file, sep =',',  header=0)
df.head()
df1=df

df1['hour']=pd.to_datetime(df1['Schedule Time']).apply(lambda x: x.hour)


#df_WG=df1[df1['Mall ID_x']=='WG']
#df_TM=df1[df1['Mall ID_x']=='TM']
#df_BM=df1[df1['Mall ID_x']=='BM']
#df_IMM=df1[df1['Mall ID_x']=='IMM']
#df_BPP=df1[df1['Mall ID_x']=='BPP']


## drop num
df1.isnull().sum(axis=0)
df3=df1[pd.notnull(df1['LSP Name'])] # find_out: when it has lsp name, it has lsp id as well.
#df3=df1[pd.notnull(df1['LSP ID'])]  # find_out:if trans has no lsp name, no lsp id as well

## sort data by  time
df3=df3.sort_values('In Time',ascending=True)

print(df3['In Time'].min())
print(df3['In Time'].max())
time_weeks=51


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

###### create RFM table


rfmTable=df3.groupby('LSP Name').agg({
'In Time': lambda x:(NOW-x.max()).days, 
'Out Time': lambda x:((x-x.shift(1))/np.timedelta64(1, 'D')).sum()/len(x), 
'in mall time': lambda x: x.sum()/len(x),
'Booking ID': lambda x: len(x)
})

#check group
#xxx=df3.groupby('LSP Name').get_group('Cold Storage Singapore (1983) Pte Ltd')
#yyy=df3.groupby('LSP Name').get_group('Sea Trading/ ENC')

#change column name
rfmTable.rename(columns={'In Time': 'recency', 
                         'Out Time': 'frequency_trans_interval', 
                         'in mall time': 'ave_in_mall_time_monetary_cost',
                         'Booking ID': 'monetary_sum_trans_num'}, inplace=True)
                         
rfmTable['recency']=rfmTable['recency'].astype(int)  

#below is correct also
#a=df3.groupby('LSP Name').apply(lambda x: pd.Series(dict(    
#    frequency=len(x),
#    cost_monetary_value=(x['in mall time']).sum(),
#    week0=len(x[x['dayofweek'] == 0]),
#    week1=len(x[x['dayofweek'] == 1]),
#    week2=len(x[x['dayofweek'] == 2]),
#    week3=len(x[x['dayofweek'] == 3]),
#    week4=len(x[x['dayofweek'] == 4]),
#    week5=len(x[x['dayofweek'] == 5]),
#    week6=len(x[x['dayofweek'] == 6]),
#    )))
  
rfmTable['Mon']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 0]))
rfmTable['Tue']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 1]))
rfmTable['Wed']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 2]))
rfmTable['Thu']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 3]))
rfmTable['Fri']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 4]))
rfmTable['Sat']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 5]))
rfmTable['Sun']=df3.groupby('LSP Name').apply(lambda x:len(x[x['dayofweek'] == 6]))


##calculate delay
df3['delivery delay']=df3['delivery delay'].apply(lambda x: 0 if x<0 else x)
rfmTable['ave_delay_minutes']=df3.groupby('LSP Name').agg({'delivery delay': lambda x: x.sum()/len(x)})


# get sum of cancellation
#lsp_cancel_booking_num=read_csv(path+'lsp_info/lspID_cancel_booking_num.csv', sep=',') <== 
lsp_cancel_booking_num=read_csv(path+'lsp_info/lsp_cancel_booking_num.csv', sep=',')
rfmTable['LSP Name']=rfmTable.index
rfmTable=pd.merge(rfmTable, lsp_cancel_booking_num, on=['LSP Name'])
rfmTable['cancel_rate']=rfmTable['num_cancelled_booking']/rfmTable['monetary_sum_trans_num']


## split the metries
#percentile=rfmTable.rank(pct=True)
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()

## create a segmented RFM table
segmented_rfm=rfmTable

##The lowest recency, highest frequency and monetary amounts are our best customers.
# The lower the value is, the hi
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

##        
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
## 

## the lower the cost value , the higher the monetary score  
def cost_Score(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4     
        
# add segment numbers to the newly created segmented RFM table
segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency_trans_interval'].apply(RScore, args=('frequency_trans_interval', quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_sum_trans_num'].apply(FMScore, args=('monetary_sum_trans_num', quantiles,))

# calculate cost_quartile
segmented_rfm['cost_quartile'] = segmented_rfm['ave_in_mall_time_monetary_cost'].apply(cost_Score, args=('ave_in_mall_time_monetary_cost', quantiles,))

# calculate  punctuality_quartile
segmented_rfm['punctuality_quartile']=segmented_rfm['ave_delay_minutes'].apply(cost_Score, args=('ave_delay_minutes',quantiles,))

# segmented_rfm.head()

##Add a new column to combine RFM score: 111 is the highest score as we determined earlier.
#segmented_rfm['RFMScore_concat']= (segmented_rfm.r_quartile.map(str)+segmented_rfm.f_quartile.map(str)+segmented_rfm.m_quartile.map(str)
#                                +segmented_rfm.cost_quartile.map(str)+segmented_rfm.punctuality_quartile.map(str))


segmented_rfm['cacel_rate_quartile']=segmented_rfm['cancel_rate'].apply(cost_Score, args=('cancel_rate', quantiles,))


# sum f+m+cost+punctuality
segmented_rfm['RFMScore_sum']= (segmented_rfm.f_quartile
                            +segmented_rfm.m_quartile
                            +segmented_rfm.cost_quartile
                            +segmented_rfm.punctuality_quartile
                            +segmented_rfm.cacel_rate_quartile)

#segmented_rfm.head()

## who are the top 10 of our best customers
#segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('frequency', ascending=False).head(15)

## method 1: sort by score only
#segmented_rfm.sort_values('RFMScore_concat', ascending=True).head(10)
segmented_rfm=segmented_rfm.sort_values('RFMScore_sum', ascending=True)
segmented_rfm['rank']=segmented_rfm['RFMScore_sum']-segmented_rfm['RFMScore_sum'].min()

##


## method 2: if sore >7, sort by frequency
#segmented_rfm[segmented_rfm['RFMScore2']>7].sort_values('frequency_trans_perweek', ascending=False).head(15)
#segmented_rfm.to_csv('/Users/suzy/Documents/suzyWork/imda/data/dsq/csv/raw/customer_rfmTable.csv', header=True, index=True)
segmented_rfm.to_csv('/Users/suzy/Documents/suzyWork/imda/data/dsq/csv/raw/customer_rfmTable.csv', header=True, index=False)