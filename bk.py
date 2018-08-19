#************************************* booking data
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
import matplotlib.pyplot as plt

# fill nan and process and save
def fill_na(dsq):
    dsq=dsq.fillna('NA')
    return dsq
    
def func_rep(x):
    x=x.replace(',',';')
    return x

##


############### to process booking data
def process_booking(dsq):
    dsq=dsq.fillna('NA')  
    dsq['Created On']=dsq['Created On'].apply(lambda x: str(func2(x)))  
    dsq['Created On']=pd.to_datetime(dsq['Created On'])
    dsq['Modified On']=dsq['Modified On'].apply(lambda x: str(func2(x)))  
    dsq['Modified On']=pd.to_datetime(dsq['Modified On'])
    ## dsq['Schedule Time'] has only hour, has no date, need to combine with 'Schedule Date', as new'Schedule Time'
    dsq['Schedule Time']=dsq['Schedule Time'].apply(lambda x: ' '+str(x)+':00')    
    dsq['Schedule Date']=dsq['Schedule Date'].apply(lambda x: x.split(' ')[0])
    dsq['Schedule Time']=dsq['Schedule Date'].apply(lambda x: x.split(' ')[0])+dsq['Schedule Time']
    dsq['Selected Tenants']=dsq['Selected Tenants'].apply(lambda x: func_rep(str(x)))    

    
    return dsq
    

#dsq1=process_booking(dsq1)


# get one mall booking info
def one_mall(dsq, mall):
    dsq_tm=dsq[dsq['Mall ID']==mall]
    return dsq_tm

########## analysis booking behaivor according to 'created by'(LSP)
def count_by_createdBy(dsq, path):
    dsq=dsq.sort_values(by=['Schedule Time'], ascending=True)
    count_tel=0
    count_1booking=0
    count_2booking=0
    count_3booking=0
    count_3morebooking=0
    court_trick=0
    court_notrick=0
    dsq['day']=dsq['Schedule Date'].apply(lambda(x):x.split(' ')[0])
    for lsp_i, lsp_v in dsq.groupby(['Created By']): 
        #print(driver_v['Driver ID'].unique())
        #print lsp_i   
        #if lsp_i.split('65')[0]=='+':
        #   count_tel+=1     
        #directory='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/TM_booking_by_createdBy/'+str(lsp_i)+'/'
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #    os.chdir(directory)   
            for date_i, date_v in lsp_v.groupby(['day']):
                court_notrick+=1
                if len(date_v)==1:
                    count_1booking+=1
                if len(date_v)==2:
                    count_2booking+=1
                if len(date_v)==3:
                    count_3booking+=1
                if len(date_v)>3:
                    count_3morebooking+=1 
                if len(date_v)>1: 
                    if len(date_v['Delivery Status'].unique())>1:
                        if 'COMPLETED' in set(date_v['Delivery Status'].unique()):
                            if 'CANCELED' in set(date_v['Delivery Status'].unique()):
                                c=max(date_v[date_v['Delivery Status']=='COMPLETED']['Created On'])
                                a = max(date_v[date_v['Delivery Status']=='CANCELED']['Schedule Time'])
                                #b = min(date_v['Created On'])
                                
                                if (pd.to_datetime(c)>pd.to_datetime(a)):
                                    print (str(a)+' '+str(c))
                                    court_trick+=1
                                    directory=path+str(lsp_i)+'/'
                                    if not os.path.exists(directory):
                                        os.makedirs(directory)
                                        os.chdir(directory)                
                                    date_v.to_csv(str(date_i)+'.csv', header=True, index=False)
                                #a = date_v['Schedule Time'].shift(1)# shift down
                                    
                                #print b
                                #if a is not NaT:
                                #    if b is not NaT:
                                #        print (str(b-a))
            
    s=[{'1: '+str(count_1booking):count_1booking}, {'2: '+str(count_2booking):count_2booking}, {'3: '+str(count_3booking):count_3booking}, {'>3: '+str(count_3morebooking):count_3morebooking}]
    s = pd.DataFrame(s) # covert to dataframe
    s.plot.bar() 
    print count_tel
    print 'trick : ' 
    print str(court_trick) 
    print str(court_notrick)
    print str(court_trick*1.0/court_notrick)


def func(x): # for change booking data ['Created On']
    if x==' ':
        return x
    m=x.split(' ')[0].split('/')[0]
    d=x.split(' ')[0].split('/')[1]
    y=x.split(' ')[0].split('/')[2]
    t=x.split(' ')[1]
    dt=str(y+'-'+m+'-'+d+' '+t)
    return dt
    
def func2(x): # for change booking data ['Created On']
    if x==' ':
        return x
    print x
    m=x.split(' ')[0].split('/')[0]
    d=x.split(' ')[0].split('/')[1]
    y=x.split(' ')[0].split('/')[2]
    t=x.split(' ')[1]
    hour=t.split(':')[0]
    minute=t.split(':')[1]
    second=t.split(':')[2]
    
    halfday=x.split(' ')[2]
    if halfday=='PM':
        dt=str(y+'-'+m+'-'+d+' '+t)
    if halfday=='AM':
        hour=float(hour)+12
        if hour==24:
            hour=0
        #print hour
        dt=str(y+'-'+m+'-'+d+' '+str(hour)+':'+minute+':'+second)
    return dt
    

    

def ana_booking_time_by_mall(dsq, Mall_ID):
    tm_booking=dsq[dsq['Mall ID']==Mall_ID]
    #tm_booking=tm_booking[['Booking ID', 'Schedule Date','Schedule Time', 'Created On','Delivery Status','Mall ID','Created By']]
    tm_booking['Created On']=tm_booking['Created On'].apply(lambda x: str(func2(x)))
    tm_booking['Created On']=pd.to_datetime(tm_booking['Created On'])
    tm_booking['Created On hour']=tm_booking['Created On'].apply(lambda x: x.hour)
    tm_booking['Schedule Time']=pd.to_datetime(tm_booking['Schedule Time'])
    print(tm_booking['Created On hour'].value_counts().sort_index())
    
    return tm_booking

    
# get one all of yamato booking info
def one_mall_yamato(dsq, mall,yamato):
    dsq_tm=dsq[dsq['Mall ID']==mall]
    dsq_tm_ymt=dsq_tm[dsq_tm['Created By']==yamato]  
    return dsq_tm_ymt

#y=one_mall_yamato(dsq, 'BM','YAMATO')

def one_mall_yamato_oneday(dsq, mall,yamato,date):
    dsq_tm=dsq[dsq['Mall ID']=='TM']
    dsq_tm_ymt=dsq_tm[dsq_tm['Created By']==yamato]
    dsq_tm_ymt_oneday=dsq_tm_ymt[dsq_tm_ymt['Schedule Date']==date]
    
    return dsq_tm_ymt_oneday

#y=one_mall_yamato_oneday(dsq, 'TM','YAMATO', '2017-12-08')        
                


########################################################### read

path='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/raw_booking_inout/'
#file_name='Booking_Search_Oct_Jan.csv'
file_name='Booking_Search2017_2018Jan.csv'

dsq=read_csv(path+file_name, sep=',')
dsq1=process_booking(dsq)  
dsq1=get_unique_LSP_Name(dsq1)
# flter out the cancelled booking 
dsq1=dsq1[dsq1['Delivery Status']!='CANCELED']


dsq1['Created_hour']=dsq1['Created On'].apply(lambda x: x.hour)
dsq1['Created_dayofweek']=dsq1['Created On'].apply(lambda x: x.dayofweek)
#C = np.where(cond, A, B)
#defines C to be equal to A where cond is True, and B where cond is False.
#dsq1['Created_on']=np.where((dsq1['Created On']<=dsq1['Modified On']), dsq1['Modified On'], dsq1['Created On'])
#dsq1['Created_on']=pd.to_datetime(dsq1['Created_on'])
dsq1['Schedule Time']=pd.to_datetime(dsq1['Schedule Time'])
dsq1['early_booking']=(dsq1['Schedule Time']-dsq1['Created On']).astype('timedelta64[s]') 
dsq1.to_csv(str(path)+'booking_clean_data.csv', header=True, index=False)

col=['Mall ID','Booking Type','LSP Name','Tenant ID','Created_hour','Created_dayofweek','early_booking']
d=dsq1[col]
d=d[d['LSP Name']!='LSP']
d.to_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/'+'for_data_studio/for_data_studio_booking.csv', header=True, index=False)
#['Trans Date','Mall Name','AD - HOC Booking Remarks','Company Name','IMD Customer','Schedule Date','Driver Name','Delivery Status','Driver Mobile No']

a2=dsq1['LSP Name'].value_counts()
a2.to_csv(str(path)+'withno_cancelled_top_LSP_names.csv', header=True, index=True)
a3=dsq1['Company Name'].value_counts()
a3.to_csv(str(path)+'top_Company_names.csv', header=True, index=True)
a4=dsq1['Supplier Name'].value_counts()
a4.to_csv(str(path)+'top_Supplier_names.csv', header=True, index=True)


dsq1['aa']=dsq1['LSP Name']+'-'+ dsq1['Company Name'] +'-'+ dsq1['Supplier Name']
   
tm_booking=ana_booking_time_by_mall(dsq,'BM')
tm_booking=ana_booking_time_by_mall(dsq,'TM')
tm_booking=ana_booking_time_by_mall(dsq,'BPP')
tm_booking=ana_booking_time_by_mall(dsq,'IMM')
tm_booking=ana_booking_time_by_mall(dsq,'WG')

count_by_createdBy(tm_booking,'/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/BM_booking_by_createdBy/')


    


##process

dsq1['Schedule Time']=pd.to_datetime(dsq1['Schedule Time'])

#use




y=one_mall_yamato(dsq, 'BM','YAMATO')
y=one_mall_yamato_oneday(dsq, 'TM','YAMATO', '2017-12-08')

dsq['Schedule hour']=dsq['Schedule Time'].apply(lambda x: x.hour)
#dsq=dsq.drop(['Schedule Time'],1)

# want to know how early need to schedule ?  use dsq['Schedule Time']-dsq['Created On']
dsq['Created On']=pd.to_datetime(dsq['Created On'])
e=dsq['Schedule Time']-dsq['Created On']
e=pd.DataFrame(e.apply(lambda x: x.seconds))



## 'trans date' date is accurate, time is not accurate, use in&out data instead
#dsq['Trans Date']=pd.to_datetime(dsq['Trans Date'])
#dsq['Trans hour']=dsq['Trans Date'].apply(lambda x: x.hour) 
#dsq['Time delay']=dsq['Trans Date']-dsq['Schedule Time'] 
#dsq['Trans weekday']=dsq['Trans Date'].apply(lambda x: x.dayofweek)


#get basic info
headlist=dsq.columns

for head in headlist:
    #dsq['Delivery Status'].describe()
    #dsq['Delivery Status'].unique()
    #dsq['Delivery Status'].value_counts()
    print head   
    #print(dsq[head].describe()) 
    if head == 'Created By': # 0-Mon 1-Tue 3-Wed        
       #print(dsq['Delivery Status'].describe())
        s=dsq[head].value_counts()
        close()
        s.plot.bar()
        savefig('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/find_out/booking_search/'+head+'.png')
        print(s)



