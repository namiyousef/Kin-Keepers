"""
Author: Yousef Nami
Date: 29.08.2020
"""
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def polynomial(x):
    """ takes an array and returns it after our polynomial function has been applied to it"""
    C = [0.7741697399557282,-0.15839741967042406,0.09528795099596377,-0.004279871380772796]
    y = C[0]*np.power(x,4)+C[1]*np.power(x,2)+C[2]*x+C[3]
    return y



def findBestModel(X_train, X_test, Y_test, model='iForest'):
    """ Function to find the best parameters to use for a given model 
    components: 
    X_train: numpy array of the input data
    X_test: list containing numpy arrays of different test data
    Y_test: list containing numpy array of different test outcomes (note that this is configured differently 
    for different algorithms,for iForest, each column must have -1 or 1. -1 --> the anomaly, if 1 --> not an anomaly)
    model: string to determine model type
    """
    if model == 'iForest':
        total_score = 0;
        parameters = [0,0,0,0]
        for max_features in range(1,X_train.shape[1]+1):
            for contamination in range(1,101):
                iForest = IsolationForest(n_estimators = 100, max_features = max_features,\
                                         contamination = contamination/1000, random_state = 0).fit(X_train)
                
                scores = []
                for x_test,y_test in zip(X_test,Y_test):
                    y_hat = iForest.predict(x_test)
                    score = evaluate(y_test,y_hat) # returns similarity percentage
                    scores.append(score)
                
                if sum(scores) > total_score:
                    total_score = sum(scores)
                    parameters[0] = max_features
                    parameters[1] = contamination/1000
                    parameters[2] = total_score
                    parameters[3] = scores
                    print(parameters, contamination)
    
    return parameters

def evaluate(y_test, y_hat):
    """ function to evaluate the score of a predicted array and a 'ground truth' array
    components:
    y_test:
    y_hat:
    """
    score = np.sum(y_test==y_hat)/len(y_test)
    return score


def convert_to_hist(df,nbins = 100,normalise = True):
    """ converts a scatter plot into a histogram. Note that for this to work best, your scatter plot must 'look'
    like a distribution that could be turned into a histogram, i.e. it must have some sort of hump.
    Components:
    df: the data that you are feeding the function. Must have two columns, x and y for the axes respectively
    nbins: number of bins to segement the data into
    normalise (optional): normalises the resulting histogram
    """
    x_values = []
    y_values = []
    x_unique = df.x.unique()
    x_max = df.x.max()
    x_min = df.x.min()
    x_values.append(x_min)
    print(x_min)
    y_values.append(df.y[df.x == x_min].tolist()[0])
    bins = np.linspace(x_min,x_max,nbins)
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111)
    for i in range(len(bins) - 1):
        y = df.y[(df.x > bins[i]) & (df.x < bins[i+1])]
        y_max = y.max()
        x_mid = (bins[i]+bins[i+1])/2
        y_values.append(y_max)
        x_values.append(x_mid)
        plt.plot(x_mid,y_max,'r.')
        
    ax.set_aspect('equal')
    return x_values,y_values