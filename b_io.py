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


def func2(x): # for change booking data ['Created On']
    if x==' ':
        return x
    #print x
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

############### to process booking data
def process_booking(dsq):
    dsq['Created On']=dsq['Created On'].apply(lambda x: str(func2(x)))  
    dsq['Created On']=pd.to_datetime(dsq['Created On'])
    dsq['Schedule Time']=dsq['Schedule Time'].apply(lambda x: ' '+str(x)+':00')
    dsq['Schedule Date']=dsq['Schedule Date'].apply(lambda x: x.split(' ')[0])
    dsq['Schedule Time']=dsq['Schedule Date'].apply(lambda x: x.split(' ')[0])+dsq['Schedule Time']
    dsq['Modified On']=dsq['Modified On'].apply(lambda x: str(func2(x)))  
    dsq['Modified On']=pd.to_datetime(dsq['Modified On'])

    return dsq


############### define to process inout data
def process_inout(dsq):
    #'Trans Date ' has date, without time, need to combine with 'in time' as new 'in time' and 'out time'
    dsq['In Time']=dsq['Trans Date'].apply(lambda x: x.split(' ')[0])+str(' ')+dsq['In Time']
    #dsq['In Time']=dsq['Trans Date']
    dsq['Out Time']=dsq['Trans Date'].apply(lambda x: x.split(' ')[0])+str(' ')+dsq['Out Time']
    dsq['In Time']=pd.to_datetime(dsq['In Time'])
    dsq['Out Time']=pd.to_datetime(dsq['Out Time'])
    #dsq['stay']=dsq['Out Time']- dsq['In Time']  
    #dsq['stay']=dsq['stay'].astype('timedelta64[s]')  # convert to seconds
    return dsq



# get one mall booking info
def one_mall(dsq, mall):
    dsq_tm=dsq[dsq['Mall ID']==mall]
    return dsq_tm

# combine booking and inout
def concat_book_inout_byBooking_cal_delay(dsq1, dsq2):
    a=dsq1[['Booking ID','Schedule Time']]
    b=dsq2[['Booking ID', 'In Time']]
    #result = pd.concat([a, b], axis=1,join_axes=[a['Booking ID']])
    result=pd.merge(a, b, on=['Booking ID'])
    result['In Time']=pd.to_datetime(result['In Time'])
    result['Schedule Time']=pd.to_datetime(result['Schedule Time'])
    result['delay']=result['In Time']-result['Schedule Time']
    print(result['delay'].describe())
    print 'sum=  '+str(sum(result['delay']))
    return result
##
def get_unique_LSP_Name(dt):
    dt['LSP Name']=dt['LSP Name'].apply(lambda x: 'Tiong Nam Logistics' if x=='Tiong Nam Logistics (S) Pte Ltd' else x)
    dt['LSP Name']=dt['LSP Name'].apply(lambda x: 'Bollore Logistics' if x=='BOLLORE LOGISTICS SINGAPORE PTE LTD' else x)
    dt['LSP Name']=dt['LSP Name'].apply(lambda x: 'LF Logistics Pte Ltd' if x=='LF LOGISTICS SERVICES PTE LTD' else x)
    dt['LSP Name']=dt['LSP Name'].apply(lambda x: 'Schenker Singapore' if x=='SCHENKER SINGAPORE (PTE) LTD' else x)
    
    return dt


#use0 : concat booking and inout data and save it to new file

path='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/raw_booking_inout/'
#filelist=['trip_info_201711_dgs.csv']
file_1='Booking_Search2017_2018Jan.csv' # booking 
file_2='FOP0248R1V1_TrafficInOutHistory_2017_2018Jan.csv' # in and out

dsq1=read_csv(path+file_1, sep =',',  header=0)
dsq2=read_csv(path+file_2, sep =',',  header=0)

a=process_booking(dsq1)
a=get_unique_LSP_Name(a)
a.to_csv(str(path)+'processed_booking.csv', header=True)
b=process_inout(dsq2)
result=pd.merge(a, b, on=['Booking ID'])
#result.to_csv(str(path)+'concat.csv', header=True, index=False)
col=['Booking ID','Mall ID_x','Booking Type','Created On_x','Schedule Time','In Time','Out Time','LSP Name','Supplier Name','Company Name','Modified On_x']
result2=get_unique_LSP_Name(result)
result2.to_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/'+'concat.csv', header=True, index=False)

# now the concat data is got. run java to do ana

#######################################################################


#use1
dsq1_mall=one_mall(dsq1, 'BM')
dsq2_mall=one_mall(dsq2, 'BM')
result=concat_book_inout_byBooking_cal_delay(dsq1_mall, dsq2_mall)

#use2 


#subdsq=dsq1[dsq1['Delivery Status']!= 'CANCELED']
#dsq1=subdsq
booking_data=a
def get_num_cancel_lsp_day(booking_data):
    booking_data['schedule day']=booking_data['Schedule Time'].apply(lambda x: x.split(' ')[0])
    cancel=booking_data.groupby(['LSP Name','schedule day']).agg({'Delivery Status': lambda x:len(x=='CANCELED')})
    cancel.rename(columns={'Delivery Status':'num_cancel_theday'},inplace =True)
    cancel_df=pd.DataFrame(cancel,columns=['LSP Name','schedule day', 'Delivery Status'])    
    book_cancel=pd.merge(booking_data,cancel_df,on=['LSP Name','schedule day'])
    return book_cancel

def booking_num_cancel(booking_data):
    cancel=booking_data.groupby(['LSP Name']).agg({'Delivery Status': lambda x:len(x=='CANCELED')})
    cancel.rename(columns={'Delivery Status':'num_cancelled_booking'},inplace =True)
    cancel.to_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/lsp_info/lsp_cancel_booking_num.csv',header=True, index=True)
    return cancel



def booking_distribution_day(a):
    a=a[a['Delivery Status']=='COMPLETED']
    a['Schedule Time']=pd.to_datetime(a['Schedule Time'])
    a['Schedule dayofweek']=a['Schedule Time'].apply(lambda x: x.dayofweek)
    mall=['TM', 'BM', 'BPP','WG','IMM']
    for v in mall:
        print v
        aa=a[a['Mall ID']==v]
        num=aa.groupby('Schedule dayofweek').agg({'Delivery Status': lambda x: len(x)})
        num.rename(columns={'Delivery Status': 'num_booking'}, inplace=True)
        num.to_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/result_for_cudo/'+str(v)+'_booking_distribution.csv', header=True, index=True)

def booking_distribution_hour_mean(a):
    a=a[a['Delivery Status']=='COMPLETED']
    a['Schedule Time']=pd.to_datetime(a['Schedule Time'])
    a['Schedule dayofweek']=a['Schedule Time'].apply(lambda x: x.dayofweek)
    a['Schedule hour']=a['Schedule Time'].apply(lambda x: x.hour)
    mall=['TM', 'BM', 'BPP','WG','IMM']
    for v in mall:
        print v
        aa=a[a['Mall ID']==v]
        num=aa.groupby(['Schedule dayofweek','Schedule hour']).agg({'Delivery Status': lambda x: len(x)*1.0/55})
        num.rename(columns={'Delivery Status': 'num_booking'}, inplace=True)
        num.to_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/result_for_cudo/'+str(v)+'_booking_hour_distribution.csv', header=True, index=True)


def booking_distribution_hour_sum(a):
    a=a[a['Delivery Status']=='COMPLETED']
    a['Schedule Time']=pd.to_datetime(a['Schedule Time'])
    a['Schedule dayofweek']=a['Schedule Time'].apply(lambda x: x.dayofweek)
    a['Schedule hour']=a['Schedule Time'].apply(lambda x: x.hour)
    mall=['TM', 'BM', 'BPP','WG','IMM']
    for v in mall:
        print v
        aa=a[a['Mall ID']==v]
        num=aa.groupby(['Schedule dayofweek','Schedule hour']).agg({'Delivery Status': lambda x: len(x)/55})
        num.rename(columns={'Delivery Status': 'num_booking'}, inplace=True)
        num.to_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/result_for_cudo/'+str(v)+'_booking_hour_distribution.csv', header=True, index=True)



def write(myData, myFile): 
    myFile = open(myFile, 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)  
#




def cal_inmall_delivery(dsq2):
    d=dsq2['Booking ID'].dropna()
    
def count(dsq1, dsq2): 
    set_bookingid_book= set(dsq1['Booking ID'].unique())
    set_bookingid_inout= set(dsq2['Booking ID'].unique())   
    count =0
    count2=0
    set_booking_id_book_complete=set(dsq1[dsq1['Delivery Status']=='COMPLETED']['Booking ID']) # completed trans in booking data
    for v in set_bookingid_inout:
        #myFile1='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/tm_book_complete.csv'
        #myFile2='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/tm_book_inout.csv'
        if v in set_bookingid_book:
            print v
            count+=1
            if v in set_booking_id_book_complete:
                count2+=1
                #print (dsq1[dsq1['Booking ID']==v])
                #write(dsq1[dsq1['Booking ID']==v], myFile1)
                #write(dsq2[dsq2['Booking ID']==v], myFile2)
                Schedule_time= pd.to_datetime(dsq1[dsq1['Booking ID']==v]['Schedule Time'])
                #Schedule_time=dsq1[dsq1['Booking ID']==v]['Schedule Time']
                in_time= pd.to_datetime(dsq2[dsq2['Booking ID']==v]['In Time'])
                #in_time=dsq2[dsq2['Booking ID']==v]['In Time']
                print Schedule_time           
                print in_time
                
    print count
    print count2


