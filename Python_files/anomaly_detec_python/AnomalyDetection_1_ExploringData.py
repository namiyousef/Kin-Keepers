#!/usr/bin/env python
# coding: utf-8

# # AnomalyDetection_1_ExploringData
# 
# The first part of the project (separating significant movements from non-significant ones) has been complete, with the following condition having been found:
# 
# $$ M =\begin{cases} 
#       -1 & g_ig_i > p\left(\frac{1}{\dot{\theta_i}\dot{\theta_i}}\right)\\
#       1 & g_ig_i\leq p\left(\frac{1}{\dot{\theta_i}\dot{\theta_i}}\right) \\
#    \end{cases}
# $$
# 
# $$p(x_i) = C_ix_i$$
# 
# $$ C_i = \begin{pmatrix}
# 0.7741697399557282\\
# -0.15839741967042406\\
# 0.09528795099596377\\
# -0.004279871380772796
# \end{pmatrix} \,\, \mathrm{and} \,\,  x_i = \begin{pmatrix}
# x^4\\
# x^2\\
# x\\
# 1
# \end{pmatrix} $$
# 
# On the assumption that this is a good model (ideally given more resources and time, more elaborate testing would have been carried out), the goal now is to find anomalies in time series of the significant movements.

# ## Libraries and Configuration

# In[1]:


""" Libraries """

#file / system libraries 
import os
import datetime as dt

# mathematical 

import numpy as np

# data exploration

import pandas as pd

# data visualization

import matplotlib.pyplot as plt

""" Configuration """

# pandas 

pd.set_option('display.max_columns', None)


# ## Functions

# In[2]:


def polynomial(x):
    """ takes an array and returns it after our polynomial function has been applied to it"""
    C = [0.7741697399557282,-0.15839741967042406,0.09528795099596377,-0.004279871380772796]
    y = C[0]*np.power(x,4)+C[1]*np.power(x,2)+C[2]*x+C[3]
    return y

def directory_to_df(paths, exclude = [None], filetype = '.csv',ignore_index = True, exception = '_repet'):
    """ concatenates all files in a directory into a dataframe
    components:
    path: path to the directory (must end with /)
    exclude: array of directories to excludes from the treatment
    filetype: a string of the file extension (must include .)
    ignore_index: boolean that tells pandas to ignore the index or not
    exception: takes a string. Any time a filename includes this string it is treated differently (for cases when you have
    more than one ) 
    """
    filenames = []
    file_column = []
    frames = []
    test_index = 1
    
    for path in paths:
        for filename in os.listdir(path):
            print(path)
            if filetype in filename and filename not in exclude:
                if exception in filename:
                    curr_df = pd.read_csv(path+filename)
                    curr_df = special_treatment(curr_df)
                    
                else:
                    curr_df = pd.read_csv(path+filename)                    
                frames.append(curr_df)
                filenames.append(filename.replace(filetype,''))
                for i in range(curr_df.shape[0]):
                    file_column.append(test_index)
                test_index+=1

    df = pd.concat(frames,ignore_index = ignore_index)
    df['files'] = file_column
    return df, filenames


def special_treatment(df):
    """ performs a custom operation on a dataframe
    components:
    df: dataframe to play on
    """
    columns = df.columns.values.tolist()
    columns.remove('date')
    df.drop('gyrZ',inplace = True, axis = 1)
    df.columns = columns
    df.reset_index(inplace = True)
    df.rename(columns= {'index':'date'},inplace = True)
    return df


# ## Data

# In[18]:


base = '/Users/yousefnami/KinKeepers/ProjectAI/Kin-Keepers/Data/{}'
paths = [base.format('Rohan/'),base.format('Ignacio/')]
frames = []

for index,path in enumerate(paths):
    frames.append(directory_to_df([path]))
    frames[index][0]['accTotal'] =  np.sqrt(np.power(frames[index][0][['accX','accY','accZ']],2).sum(axis = 1))
    frames[index][0]['gyrTotal'] =  np.sqrt(np.power(frames[index][0][['gyrX','gyrY','gyrZ']],2).sum(axis = 1))


    
df_rohan = frames[0][0]
df_ignacio = frames[1][0]


dfs = []
dfs.append(df_rohan)
dfs.append(df_ignacio)
names = ["rohan's data", "ignacio's data"]

for index,df in enumerate(dfs):
    dfs[index] = df[df.accTotal > polynomial(1/df.gyrTotal)]
    dfs[index].to_csv('/Users/yousefnami/KinKeepers/ProjectAI/Kin-Keepers/Data/{}_filtered.csv'.format(names[index][:-7]))
    
df_rohan = dfs[0]
df_ignacio = dfs[1]


# In[4]:


df_rohan


# In[5]:


df_ignacio


# In[6]:


for df in dfs:
    df.date = pd.to_datetime(df.date)
    times = []
    
    # this is good, but you must apply it for EACH day
    for index,time in enumerate(df.date.values):
        if index == 0:
            times.append((time - time)/np.timedelta64(1, 's'))
        else:
            times.append((time - df.date.values[0])/np.timedelta64(1, 's'))
    df['times'] = times
print('time',type(time))
print('value',type(df.date.values[0]))
df_ignacio.head()


# In[15]:


class seasonality():
    """ takes in a dataframe, outputting it with two extra columns: seasonality (but column name = seasonality
    inputted) and times, where 'times' is a plottable version of date with reference to a prespecified start time
    (day_start)
    Components:
    df: the dataframe, must have the dates column as 'date' and in np.datetime64 timeformat
    seasonality (optional): defaults to 'day'. This is the criteria for splitting the data
    day_start (optional): this signifies what is the 'start time' of the day (i.e. the 0 point on the x axis). Defaults
    for midnight.
    time_delta (optional): this defines the units for the time delta between data points. Defaults to seconds.
    EDIT THIS MSG
    NEED TO FIX THIS
    """ 
    def __init__(self,df,seasonality='day',day_start = '00:00:00', time_delta = 's'):
        
        if seasonality not in ['hour','day','month','year']:
            raise ValueError("you can only input the following for seasonality: 'day', 'month', or 'year'")
        self.df = df
        self.seasonality = 'seasonality_{}'.format(seasonality)
        try:
            self.day_start = dt.datetime.strptime(day_start,'%H:%M:%S')
        except:
            raise ValueError('Please enter your day_start in the correct format: "HH:MM:SS". "{}" is not acceptable'                             .format(day_start))
        self.time_delta = time_delta

    def find_seasonal_trends(self):
        if 'hour' in self.seasonality:
            self.df[self.seasonality] = self.df.date.dt.hour
        elif 'day' in self.seasonality:
            self.df[self.seasonality] = self.df.date.dt.day
        elif 'month' in self.seasonality:
            self.df[self.seasonality] = self.df.date.dt.month
        else:
            self.df[self.seasonality] = self.df.date.dt.year
            
        self.create_times()


        return self.df
    
    def create_times(self):
        times = []
        for season in self.df[self.seasonality].unique():
            temp_dates = self.df.date[self.df[self.seasonality] == season].values
            date = dt.datetime.strptime(str(temp_dates[0])[:-3], '%Y-%m-%dT%H:%M:%S.%f')
            # 'date' is wrong: this will not work for when you have a lower order seasonality.
            # it needs to adapt such that it starts recording when the beginning of the year
            start_day = dt.datetime(date.year,
                                    date.month,
                                    date.day,
                                    self.day_start.hour,
                                    self.day_start.minute,
                                    self.day_start.second)
            start_day = np.datetime64(start_day)
            
            for index, date in enumerate(temp_dates):
                times.append((date - start_day)/np.timedelta64(1, self.time_delta))
        self.df['times'] = times


# In[8]:



df_temp = df_ignacio
#df_temp.date = pd.to_datetime(df_temp.date)

myObj = seasonality(df_temp,time_delta = 's')


df_temp = myObj.find_seasonal_trends()

#df_ignacio = find_seasonal_trends(df_ignacio,seasonality = 'month')
#df_ignacio.date.dt.day
#df_ignacio.head()
df_temp.head()


# In[9]:


#df.date = pd.to_datetime(df_temp.date)
for index,df in enumerate(dfs):
    seasonal = seasonality(df)
    dfs[index] = seasonal.find_seasonal_trends()
    
df_rohan = dfs[0]
df_ignacio = dfs[1]


# In[10]:


df_rohan.head()


# In[11]:


df_ignacio.head()


# In[12]:


colors = ['r','b']
i = 1
for df,color in zip(dfs,colors):
    fig = plt.figure(figsize = (16,16))
    for season in df.seasonality_day.unique():
        df_temp = df[df.seasonality_day == season]
        fig.add_subplot(len(df.seasonality_day.unique()),len(dfs),i)
        print(i)
        plt.plot(df_temp.times,df_temp.accTotal,'{}.'.format(color))
        i+=1
        
    #plt.plot(df.times,df.accTotal,'{}.'.format(color))
    #print(df.times.max())


# In[14]:


fig = plt.figure(figsize = (16,16))
i = 1
for df,name in zip(dfs,names):
    fig.add_subplot(2,2,i)
    i+=1
    plt.plot(df.times,df.accTotal,'.')
    plt.title("{}, size = {}".format(name,df.shape))
    plt.xlabel('time')
    plt.ylabel('acceleration')
    
    fig.add_subplot(2,2,i)
    i+=1
    plt.plot(df.times,df.gyrTotal,'.')
    plt.title("{}, size = {}".format(name,df.shape))
    plt.xlabel('time')
    plt.ylabel('gyration')


# In[ ]:




