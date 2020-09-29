"""
Author: Yousef Nami
Date: 29.09.2020
"""

# necesary libraries

from clustering import findBestModel, evaluate, convert_to_hist
from file_reading import directory_to_df

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.optimize import curve_fit


# imports already created data !
from creating_filtered_data import df

import matplotlib.pyplot as plt

import pickle

# import test data

path = ['/Users/yousefnami/KinKeepers/ProjectAI/Kin-Keepers/Data/TestData/'] # as before, path MUST
# be placed in a list

df_test, files = directory_to_df(path,ignore_index = False)

columns = df_test.columns.values.tolist()

columns.remove('date')
df_test.drop('gyrZ',inplace = True, axis = 1)
df_test.columns = columns

df_test['accTotal'] = np.sqrt(np.power(df_test[['accX','accY','accZ']],2).sum(axis = 1))
df_test['gyrTotal'] = np.sqrt(np.power(df_test[['gyrX','gyrY','gyrZ']],2).sum(axis = 1))


# scale the data so that euclidea distance does't get skewed

significance = [1,-1,1,-1,1,-1,-1,-1,1] # -1 == sig, 1 == non sig
scaler = MinMaxScaler()
X_test = []
Y_test = []

# creates the testing y data, not how it is inherently inaccurate!
for filetype in df_test.files.unique():
    X_test.append(scaler.fit_transform(np.asarray(df_test[['accTotal','gyrTotal']][df_test.files == filetype])))
    Y_test.append(np.asarray([significance[filetype -1] for i in range(X_test[filetype-1].shape[0])]))
    
X_train = scaler.fit_transform(np.asarray(df[['accTotal','gyrTotal']]))


# find the model model
""" this process takes a long time, to skip it ensure that you comment out this line, and the iForest line
below it, and to uncomment "speeding process up" """
parameters = findBestModel(X_train,X_test,Y_test)


# check visually wha tthe best model looks like

iForest = IsolationForest(n_estimators = 100, max_features = parameters[0],\
                          contamination = parameters[1], random_state = 0).fit(X_train)

# for speeding process up
#parameters = [1, 0.058, 5.8]
#iForest = IsolationForest(n_estimators = 100, max_features = parameters[0],\
#                         contamination = parameters[1], random_state = 0).fit(X_train)


output = iForest.predict(X_train)
true_false = []
                
for item in output:
    if item == 1:
        true_false.append(False)
    else:
        true_false.append(True)
        
anomalies = df[true_false]
actuals = df[[not i for i in true_false]]
                
plt.figure(figsize = (16,8))
plt.suptitle('iForest with contamination={},max_features={} and a score of {}/9'.format(*parameters))            

plt.subplot(121)
plt.plot(anomalies.index,anomalies.accTotal,'r.')
plt.plot(actuals.index,actuals.accTotal,'b.')
plt.xlabel('index (note that each index represents a second)')
plt.ylabel('total acceleration (in units of g)')

plt.subplot(122)
plt.plot(anomalies.gyrTotal,anomalies.accTotal,'r.')
plt.plot(actuals.gyrTotal,actuals.accTotal,'b.')
plt.xlabel('total gyration (in units of degrees per second)')
plt.ylabel('total acceleraion (in units of g)')
plt.legend(['significant movements','noise'])

plt.show()


# if happy with outcome, then save the model
# with open ('/Users/yousefnami/KinKeepers/ProjectAI/Kin-Keepers/Models/iForest.pkl',"wb") as f:
#    pickle.dump(iForest,f)


# create new dataframe based on reciprocal

df_new = pd.DataFrame(data = {'x':1/(actuals.gyrTotal),'y':(actuals.accTotal)})
df_new.reset_index(inplace = True,drop = True)

x,y = convert_to_hist(df_new)
x = x[0:15]
y = y[0:15]
plt.plot(x,y)
plt.xlabel('reciprocal of tot gyration (seconds per degrees)')
plt.ylabel('total acceleration in g')
plt.show()

# create function based on your expectation of what represents the curve

def func(x,a,b,c,d):
    x = np.asarray(x)
    return d+c*x + b*(np.power(x,2)) + a*(np.power(x,4))

popt, pcov = curve_fit(func, x, y)
plt.plot(x, func(x, *popt))
print(popt)


# plot results
plt.plot(1/(anomalies.gyrTotal),(anomalies.accTotal),'r.')
plt.plot(1/(actuals.gyrTotal),(actuals.accTotal),'b.')
plt.plot(x,func(x,*popt),'white')
plt.xlabel('reciprocal of total gyration (in units of seconds per degrees)')
plt.ylabel('total acceleraion (in units of g)')
plt.legend(['significant movements','noise'])


print('the equation is:\n x^4:  {}\n x^2:  {}\n x:  {}\n constant:  {}'.format(*popt))