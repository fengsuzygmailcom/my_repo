#*************************************  in_out data trans seperate by lspid by date
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

def pic(y):
    N = len(y)
    #x = range(N)
    x=['1','2','3','4']
    width = 1/1.5
    plt.bar(x, y, width, color="blue")


#filelist=['trip_info_201711_dgs.csv']
file_i='Booking_Search_Oct_Jan.csv' # booking 
#file_i='FOP0248R1V1-TrafficIn-OutHistory_Oct_Jan.csv' # in and out


#folder=file_i.split('_')[2]    
path='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/'

# load data
#colnames=['id','type','start_ts','end_ts']
dsq=read_csv(path+file_i, sep =',',  header=0)

#dsq=dsq.fillna('NA')
#dsq.to_csv(str(path)+'inout.csv', header=True, index=False)

######### process data especially the date
def dsq_process(dsq):
    #'Trans Time ' has date, without time, need to combine with 'in time'
    dsq['Trans Date']=dsq['Trans Date'].apply(lambda x: x.split(' ')[0])+str(' ')+dsq['In Time']
    dsq['Trans Date']=pd.to_datetime(dsq['Trans Date'])
    dsq['Trans weekday']=dsq['Trans Date'].apply(lambda x: x.dayofweek)
    dsq['Trans hour']=dsq['Trans Date'].apply(lambda x: x.hour)
    dsq['Trans Date']=dsq['Trans Date'].apply(lambda x: x.split(' ')[0])
    
    # sort record by taxi_id and time
    dsq=dsq.sort_values(by=['Driver ID', 'Trans Date'], ascending=True)
    dsq['day']=dsq['Trans Date'].apply(lambda(x):x.split(' ')[0])
    return dsq
    
dsq=dsq_process(dsq)

########## analysis according to Vechile ID
def countby_vechile_id(dsq):
    count1=0
    count2=0
    count3=0
    count3more=0
    mall1=0
    mall2=0
    mall3=0
    mall3more=0
    for vehicle_i, vehicle_v in dsq.groupby(['Vechile ID']):       
        #directory='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/'+str(vehicle_i)+'/'
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #    os.chdir('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/'+str(vehicle_i)+'/')   
            for date_i, date_v in vehicle_v.groupby(['day']):
                if len(date_v)==1:
                    count1+=1
                if len(date_v)==2:
                    count2+=1                   
                if len(date_v)==3:                    
                    count3+=1
                if len(date_v)>3:
                    count3more+=1
                    #print len(date_v)
                    #print vehicle_i
                    #print date_i
                if len(date_v['Mall ID'].unique())==1:
                    mall1+=1
                    #print str(vehicle_i)+' '+ str(date_i)
                    #print len(date_v['Mall ID'].unique())
                if len(date_v['Mall ID'].unique())==2:
                    mall2+=1
                if len(date_v['Mall ID'].unique())==3:
                    mall3+=1  
                    print str(vehicle_i)+ ' '+str(date_i)
                if len(date_v['Mall ID'].unique())>3:
                    mall3more+=1  
                
                #date_v.to_csv(str(date_i)+'.csv', header=True, index=False)
    s=[{'1: '+str(count1):count1}, {'2: '+str(count2):count2}, {'3: '+str(count3):count3}, {'>3: '+str(count3more):count3more}]
    s = pd.DataFrame(s) # covert to dataframe
    s.plot.bar()
    
    m=[{'1: '+str(mall1):mall1}, {'2: '+str(mall2):mall2}, {'3: '+str(mall3):mall3}, {'>3: '+str(mall3more): mall3more}]
    m = pd.DataFrame(m) # covert to dataframe
    m.plot.bar()

########## analysis according to drivers
def count_by_driver(dsq):
    count1=0
    count2=0
    count3=0
    count3more=0
    for driver_i, driver_v in dsq.groupby(['Driver ID']): 
        #print(driver_v['Driver ID'].unique())
        #print driver_i        
        #directory='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/'+str(driver_i)+'/'
        #if not os.path.exists(directory):
            #os.makedirs(directory)
            #os.chdir('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/'+str(driver_i)+'/')   
            for date_i, date_v in driver_v.groupby(['day']):
                if len(date_v)==1:
                    count1+=1
                if len(date_v)==2:
                    count2+=1           
                if len(date_v)==3:                    
                    count3+=1
                if len(date_v)>3:
                    count3more+=1
                    print (str(driver_i) + ' '+ str(date_i))
                    print len(date_v)
                    print len(date_v['Mall ID'].unique()) # driver go to one mall only per day
                    
                    
                #date_v.to_csv(str(date_i)+'.csv', header=True, index=False)
            
    s=[{'1: '+str(count1):count1}, {'2: '+str(count2):count2}, {'3: '+str(count3):count3}, {'>3: '+str(count3more):count3more}]
    s = pd.DataFrame(s) # covert to dataframe
    s.plot.bar()

########## analysis according to 'created by'(LSP)
def count_by_createdBy(dsq):
    for lsp_i, lsp_v in dsq.groupby(['Created By']): 
        #print(driver_v['Driver ID'].unique())
        print lsp_i        
        directory='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/booking_by_createdBy/'+str(lsp_i)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chdir(directory)   
            for date_i, date_v in lsp_v.groupby(['day']):
                #print len(date_v)
                #print lsp_i
                date_v.to_csv(str(date_i)+'.csv', header=True, index=False)


###################### count by col name
def count_by_head(dsq):
    headlist = dsq.columns
    for head in headlist:
        print head    
        #print(dsq[head].describe())
        #print(log[head].unique())
        s=dsq[head].value_counts()
        print(s)

                        
##$$################# analysis by Type , i.e. shop customer
def write(myData): 
    myFile = open('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/perShop.csv', 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)  
        
        
def count_by_Type(dsq):
    write([['shop','weekday','num_of_days','ave_num_record']])
    for type_i, type_v  in dsq.groupby(['Type']):
        print type_i
        #directory='/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/processed/inNout_by_Type/'+str(type_i)+'/'
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #    os.chdir(directory)   

        for date_i, date_v in type_v.groupby(['Trans weekday']):            
            c=len(date_v['day'].unique())
            print str(date_i)+' '+str(c)+' '+str(len(date_v)*1.0/c)
            data=[[str(type_i), str(date_i), str(c), str(len(date_v)*1.0/c)]]
            write(data)

            
count_by_Type(dsq) 

          
                    
                                       


#dsq_bm['Created By'].value_counts()
#YAMATO_BM    6561
#YAMATO         88

#### analysis one mall one weekday cross shops
def ana_one_mall_one_weekday(mall, weekday):
    # ** get mall data
    dsq_bm=dsq[dsq['Mall ID']==mall]
    #** get one weekday data 0-6 = monday-sunday
    dsq_bm_weekday=dsq_bm[dsq_bm['Trans weekday']==weekday]
    #** get this weekday's customer distribution
    s=dsq_bm_weekday['Type'].value_counts()
    s.plot.bar()
    
ana_one_mall_one_weekday('BM', 2)


###  analysis one mall one shop cross weekday
def ana_one_mall_one_shop(mall, shop):
    close()
    # ** get mall data
    dsq_bm=dsq[dsq['Mall ID']==mall]
    # ** get customer data for this mall
    dsq_bm_pizzaHut=dsq_bm[dsq_bm['Type']==shop]
    #** get customer behavior distribution cross a week
    s=dsq_bm_pizzaHut['Trans weekday'].value_counts()
    s.plot.bar()
#USE
ana_one_mall_one_shop('BM', 'POPULAR BOOKSTORE')



###  analysis one mall one weekday cross 24 hours
def ana_one_mall_24_hour(mall, weekday):
    close()
    # ** get mall data
    dsq_bm=dsq[dsq['Mall ID']==mall]
    # ** get mall data at one weekday
    dsq_bm_mon=dsq_bm[dsq_bm['Trans weekday']==weekday]
    #** get trans distribution cross 24 hour
    s=dsq_bm_mon['Trans hour'].value_counts(sort=True, ascending=False)
    s.plot.bar()
    savefig('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/find_out/in_out/bedok_mall/bm_in_hour_weekday_'+str(weekday)+'.png')

#USE
for i in range(6):
    ana_one_mall_24_hour('BM',i)


## ana one mall every shop 
def ana_one_mall_per_shop(dsq, mall):
    # ** get mall data
    dsq_bm=dsq[dsq['Mall ID']==mall]
    # ** get customer data for this mall
    shoplist=dsq_bm['Type'].unique()
    for shop in shoplist:
        if shop is not nan:
            print shop
            dsq_bm_pizzaHut=dsq_bm[dsq_bm['Type']==str(shop)]
            #** get customer behavior distribution cross a week
            s=dsq_bm_pizzaHut['Trans weekday'].value_counts()
            s.plot.bar()           
            savefig('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/find_out/in_out/bedok_mall/shops/'+str(shop)+'_bm_weekday.png')
            close()

#use
ana_one_mall_per_shop(dsq, 'BM')
