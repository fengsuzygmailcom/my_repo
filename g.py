# -*- coding: utf-8 -*-
import numpy as np
import csv
import sys
import string
import time
#import numpy as np
import pandas as pd
import os
from datetime import datetime



file_i='gentry_Oct16_Jan17.csv' # booking 

path='/Users/suzy/Documents/suzyWork/IMDA/data/gantry/'

data_raw=read_csv(path+file_i, sep =',',  header=0)
#[u'entry_station', u'entry_time', u'iu_tk_no', u'card_mc_no',
       #u'trans_type', u'paid_time', u'exit_station', u'exit_time',
       #u'parked_time', u'parking_fee', u'paid_amt']
       
#dd=read_csv('/Users/suzy/Documents/suzyWork/IMDA/data/dsq/csv/raw/for_delay_forcasting/data_for_delay_forcasting.csv', sep=',', header=0)
#print 'data for delay facasting'
#print dd['In Time'].min()
                           
df=data_raw
print df.entry_time.min()
print df.exit_time.max()
df=df[df['entry_time']>='2017-01-01 00:00:00']
print df.entry_time.min()
print df.exit_time.max()

df.sort_values('entry_time',ascending=True)
df['entry_time']=pd.to_datetime(df['entry_time'])
df['entry_date']=df['entry_time'].apply(lambda x: x.date())
df['entry_hour']=df['entry_time'].apply(lambda x: x.hour)
df['entry_hour'].value_counts()

df['exit_time']=pd.to_datetime(df['exit_time'])
df['exit_date']=df['exit_time'].apply(lambda x: x.date())
df['exit_hour']=df['exit_time'].apply(lambda x : x.hour)
df['exit_hour'].value_counts()


entry=pd.DataFrame({'count_entry': df.groupby(['entry_date','entry_hour']).size()}).reset_index()
leave=pd.DataFrame({'count_leave': df.groupby(['exit_date','exit_hour']).size()}).reset_index()

entry.columns=['date','hour','count_entry']
leave.columns=['date','hour','count_leave']

#entry=df.groupby(['entry_day','entry_hour']).agg({'iu_tk_no': lambda x: len(x)}).reset_index()
#entry.rename(columns={'iu_tk_no': 'sum_num_entry'}, inplace=True)
#leave=df.groupby(['exit_date','exit_hour']).agg({'iu_tk_no': lambda x: len(x)}).reset_index()
#leave.rename(columns={'iu_tk_no': 'sum_num_exit'}, inplace=True)

entry_leave = pd.merge(entry, leave, on=['date','hour'], how='outer')
entry_leave=entry_leave.fillna(0)
entry_leave.sort_values(['date','hour'],ascending=True)

entry_leave['cumsum_entry']=entry_leave['count_entry'].cumsum()
entry_leave['cumsum_leave']=entry_leave['count_leave'].cumsum()
entry_leave['occupied']=entry_leave['cumsum_entry']-entry_leave['cumsum_leave']

entry_leave.to_csv('/Users/suzy/Documents/utilization.csv', sep=',', header=True, index=False)


####### utilise
data_inout=read_csv('/Users/suzy/Documents/suzywork/imda/data/dsq/csv/raw/clean/clean_inout.csv', sep=',', header=0)

dt=data_inout[['Mall ID','In Time','Out Time']]
dt.sort_values('In Time', ascending=True)
dt['In Time']=pd.to_datetime(dt['In Time'])
dt['In_date']=dt['In Time'].apply(lambda x: x.date())
dt['In_hour']=dt['In Time'].apply(lambda x: x.hour)

dt['Out Time']=pd.to_datetime(dt['Out Time'])
dt['Out_date']=dt['Out Time'].apply(lambda x: x.date())
dt['Out_hour']=dt['Out Time'].apply(lambda x: x.hour)

in_out=pd.DataFrame()

for mall in ['BM', 'BPP', 'IMM','TM','WG']:
    print mall
    di=dt[dt['Mall ID']==mall]
    IN=pd.DataFrame({'count_entry': di.groupby(['In_date','In_hour']).size()}).reset_index()
    OUT=pd.DataFrame({'count_leave': di.groupby(['Out_date','Out_hour']).size()}).reset_index()
    
    IN.columns=['date','hour','count_entry']
    OUT.columns=['date','hour','count_leave']
    
    IN_OUT = pd.merge(IN, OUT, on=['date','hour'], how='outer')
    IN_OUT=IN_OUT.fillna(0)
    IN_OUT['Mall ID']=mall
    
    IN_OUT.sort_values(['date','hour'],ascending=True)
    IN_OUT['cumsum_entry']=IN_OUT['count_entry'].cumsum()
    IN_OUT['cumsum_leave']=IN_OUT['count_leave'].cumsum()
    IN_OUT['occupied']=IN_OUT['cumsum_entry']-IN_OUT['cumsum_leave']
    print np.array(IN_OUT['occupied'])
    print len(IN_OUT)
    in_out=pd.concat([in_out, IN_OUT])

in_out['occupied']=in_out['occupied'].apply(lambda x: max(0, x))    
in_out.to_csv('/Users/suzy/Documents/utilization_all.csv', sep=',', header=True, index=False)
