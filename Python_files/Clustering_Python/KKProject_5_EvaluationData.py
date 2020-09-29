#!/usr/bin/env python
# coding: utf-8

# # KKProject_5_EvaluationData
# 
# In this notebook I'll be exploring defined movement data.

# # Libraries

# In[2]:


# file management
import os
import pickle

# mathematical
import numpy as np
from scipy.spatial.distance import cdist 

# data exploration
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

# preprocessing

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# modelling

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.ensemble import IsolationForest


# # Dataset
# 
# - there will be no explanations, see KKProject_3_Modelling for more information

# In[3]:


path = '/Users/yousefnami/KinKeepers/ProjectAI/Data/sampleWalk.csv'
df = pd.read_csv(path)
df['accTotal'] = np.sqrt(np.power(df[['accX','accY','accZ']],2).sum(axis = 1))
df['gyrTotal'] = np.sqrt(np.power(df[['gyrX','gyrY','gyrZ']],2).sum(axis = 1))
df.head()


# In[4]:


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


# In[5]:


plt.plot(df.accTotal,df.gyrTotal,'.')


# In[9]:


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


# ## iForest

# In[105]:


#outliers = red 

inliers = []
    #outliers = []
power = 1000
model = IsolationForest(n_estimators = power, max_features = X_normal.shape[1],contamination = 0.06,random_state = 0)
model.fit(X_normal)
output = model.predict(X_normal)
for result in output:
    if result == 1:
        inliers.append(True)
    else:
        inliers.append(False)
outliers = [not i for i in  inliers]
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.plot(df.accTotal[outliers],df.gyrTotal[outliers],'r.')
plt.plot(df.accTotal[inliers],df.gyrTotal[inliers],'b.')
plt.title('n_estimators = {}'.format(power))

plt.subplot(1,2,2)
plt.plot(df_normal.accTotal[outliers],df_normal.gyrTotal[outliers],'r.')
plt.plot(df_normal.accTotal[inliers],df_normal.gyrTotal[inliers],'b.')
plt.title('n_estimators = {}'.format(power))
plt.show()


# ## Gaussian Mixture

# In[34]:


# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
# define dataset
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(X_normal)
# assign a cluster to each example
yhat = model.predict(X)

outliers = []
for item in yhat:
    if item == 1:
        outliers.append(True)
    else:
        outliers.append(False)


inliers = [not i for i in outliers]
plt.plot(df.accTotal[inliers],df.gyrTotal[inliers],'r.')
plt.plot(df.accTotal[outliers],df.gyrTotal[outliers],'b.')

print(unique(yhat))


# ## Spectral clustering

# In[8]:


from sklearn.cluster import SpectralClustering
X = df[['accTotal','gyrTotal']].values
model = SpectralClustering(n_clusters=2)
from numpy import unique
yhat = model.fit(X)
# retrieve unique clusters
clusters = unique(yhat)
clusters

# not sure but this is not working
df['clusters_spectral'] = yhat
cmap = plt.cm.get_cmap('hsv', len(clusters))

for index, cluster in enumerate(clusters):
    plt.plot(df.accTotal[df.clusters_spectral == cluster],df.gyrTotal[df.clusters_spectral == cluster], c = cmap(cluster),marker = '.',linewidth = 0)


# ## Mean shift

# In[4]:


from numpy import unique
from matplotlib import cm
from sklearn.cluster import MeanShift

model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)


# In[13]:


clusters
df['clusters_mean_shift'] = yhat
cmap = plt.cm.get_cmap('hsv', len(clusters))

for index, cluster in enumerate(clusters):
    plt.plot(df.accTotal[df.clusters_mean_shift == cluster],df.gyrTotal[df.clusters_mean_shift == cluster], c = cmap(cluster),marker = '.',linewidth = 0)


# ## BIRCH

# In[33]:


from sklearn.cluster import Birch


model = Birch(threshold=0.001, n_clusters=2)
# fit the model
model.fit(X_normal)
# assign a cluster to each example
yhat = model.predict(X_normal)
# retrieve unique clusters
clusters = unique(yhat)
print(clusters)
print(yhat)
df['clusters_birch'] = yhat
print(df.clusters_birch.value_counts())



outliers = []
for item in yhat:
    if item == 1:
        outliers.append(True)
    else:
        outliers.append(False)


inliers = [not i for i in outliers]
plt.plot(df.accTotal[inliers],df.gyrTotal[inliers],'r.')
plt.plot(df.accTotal[outliers],df.gyrTotal[outliers],'b.')


# ## Agglomerative

# In[41]:


from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)

outliers = []
for item in yhat:
    if item == 1:
        outliers.append(True)
    else:
        outliers.append(False)


inliers = [not i for i in outliers]
plt.plot(df.accTotal[inliers],df.gyrTotal[inliers],'r.')
plt.plot(df.accTotal[outliers],df.gyrTotal[outliers],'b.')


# ## DBSCAN

# ## Experiment

# In[104]:


#experiment:

df_test = df[(df.accTotal > 1.03) | (df.gyrTotal > 100)]

plt.plot(df_test.accTotal,df_test.gyrTotal,'.')
print('New shape = {}'.format(df_test.shape), 'Old shape = {}'.format(df.shape))
print('Outlier percentage = {}%'.format((df_test.shape[0]/df.shape[0])*100))


# In[ ]:





# In[ ]:


# logistic regression unsupervised? 

