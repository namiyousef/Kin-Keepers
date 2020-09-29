#!/usr/bin/env python
# coding: utf-8

# # KKProject_2_MoreData
# 
# In this notebook, I'll explore a larger dataset, keeping in mind some of my conclusions from last time

# # Libraries

# In[2]:


# file management
import os

# mathematical
import numpy as np

# data exploration
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib

# modelling

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# # Dataset
# 
# Note the following:
# - acceleration is measured in units of 'g'
# - gyration (angular velocity) has units of degrees per second
# - data has only been recorded for one hour, and as such may not be representative
# - a 'resting' object should feel an acceleration of about 1 'g'

# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
path = '/Users/yousefnami/KinKeepers/ProjectAI/Data/Ignacio/'
frames = []
for filename in os.listdir(path):
    if '.csv' in filename:
        frames.append(pd.read_csv(path+filename))

df = pd.concat(frames,ignore_index = True)
df['accTotal'] = np.sqrt(np.power(df[['accX','accY','accZ']],2).sum(axis = 1))
df['gyrTotal'] = np.sqrt(np.power(df[['gyrX','gyrY','gyrZ']],2).sum(axis = 1))

df.head()


# # Data Exploration / Visualization Part I

# In[4]:


plt.plot(df.accTotal,df.gyrTotal,'.')
plt.show()


# In[5]:


plt.plot(df.index,df.accTotal,'r.')
plt.show()
plt.plot(df.index,df.gyrTotal,'b.')
plt.show()


# In[6]:


dates = matplotlib.dates.date2num(df.date.values)
plt.plot_date(dates, df.accTotal)

#note sure why, but the plot does not seem to be great

""" needs to be fixed """


# In[7]:


plots = ['X','Y','Z','Total']
plt.figure(figsize=(18,8))
i = 1
for parameter in ['acc','gyr']:
    for plot in plots:
        plt.subplot(int('24{}'.format(i)))
        if plot == 'Total':
            plt.plot(df.index,df['{}{}'.format(parameter,plot)],'r.')
        else:
            plt.plot(df.index,df['{}{}'.format(parameter,plot)],'.')
        plt.ylabel('{}{}'.format(parameter,plot))
        plt.xlabel('time (index not actual time!)')
        i+=1
plt.show()


# In[8]:


plots = ['X','Y','Z','Total']
plt.figure(figsize=(18,8))
i = 1
for parameter in ['acc','gyr']:
    for plot in plots:
        plt.subplot(int('24{}'.format(i)))
        if plot == 'Total':
            plt.hist(df['{}{}'.format(parameter,plot)],color = 'red')
        else:
            plt.hist(df['{}{}'.format(parameter,plot)])
        plt.ylabel('{}{}'.format(parameter,plot))
        plt.xlabel('time (index not actual time!)')
        i+=1


# Based on the exploration done here, it would appear that the data is difficult to deal with. Here are some thoughts:
# 
# - the gyration vs. accleration graph does not show any nicely shaped clusters
# - It might be worth determining whether gyration is in fact a parameter worthy of consideration for this exercise: does it really say much about whether the movement is significant? (Rohan)
# - None of the graphs (time series) show 'clear' clusters
# - The data is very much skewed: gyrations are mostly zero, and accelerations are centred around 0.96 (as this would appear to be 'stationary'). It is likely that a person spends most of their day sitting, so the data points for 'significant movement' will be low; the clustering technique used must account for this (perhaps using an outlier detection algorithm is best?)
# - It's worth noting that total acceleration should not have a 'normal distribution', since you cannot have accelerations lower than that of about 1 g (since that is 'stationary'). As such, it might be worth filtering the lower end of the data when it comes to this (this would lead to a right skewed histogram, which **could** be fixed by a log transform)

# # Data Exploration / Visualization Part II

# In[9]:


median = df.accTotal.median()
iqr = df.accTotal.quantile(0.75) -df.accTotal.quantile(0.25)

df2 = df.copy()
df2 = df2[df2.accTotal >= median-iqr]
df2.shape
print(median)


# In[10]:


plt.plot(df2.index,df2.accTotal,'r.')
plt.show()

plt.hist(df2.accTotal,color = 'r')
plt.show()
#appears to be more 'normal'


# # Modelling attempt using DBSCAN
# 
# Meaning of the metrics:
# - *eps* is the max distance between two samples for one to be considered a 'neighborhood' of the other (so kind of like radius of the cluster)
# - *min_samples* number of points ina  neighborhood to consider a central point the 'core point'
# - *metric* chooses the type of distance
# - *p* is the power parameter in the minkowski distance equation

# In[170]:


get_ipython().run_cell_magic('latex', '', '\\begin{align}\n\\mathrm{Minkowski \\,distance} = \\left( \n    \\sum_{i=1}^{n}|X_i - Y_i|^p\n    \\right)^\\frac{1}{p}\n\\end{align}')


# In[11]:


X = StandardScaler().fit_transform(np.asarray(df2.accTotal).reshape(-1,1))
model = DBSCAN(eps = 0.5,min_samples = 1000, metric = 'minkowski', p = 1.5).fit(X)
model.labels_
true_false = []
for item in model.labels_:
    if item == 0:
        true_false.append(False)
    else:
        true_false.append(True)
        
anomalies = df2[true_false]
actuals = df2[[not i for i in true_false]]


# In[12]:


plt.plot(anomalies.index,anomalies.accTotal,'r.')
plt.plot(actuals.index,actuals.accTotal,'b.')


# In[178]:


plt.plot(anomalies.accTotal,anomalies.gyrTotal,'r.')
plt.plot(actuals.accTotal,actuals.gyrTotal,'b.')
#this is a better plot than the scatter plot sketched before, it seems to suggest 1 important thing:
#gyration might actually be an important element to consider. There is a very large cluster between 0.95-1 on the
#x axis and 0 100 on the y axis? Let's explore this using Isolation forest, K means, DBSCAN, SVM

