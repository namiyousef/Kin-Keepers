"""
Author: Yousef Nami
Date: 29.09.2020
"""

# necessary libraries

import pandas as pd

from anomaly_detection import moving_avg, average


# importing filtered data

path = '/Users/yousefnami/KinKeepers/ProjectAI/biometrics/' 
# this is the path to the repository parent dirctory, i.e. biometrics

path_to_data = 'movement_data_analysis/Data/Filtered_data/' # path to data directory

files = ['ignacio_filtered.csv','rohan_filtered.csv'] # name of the files of interest 

# read csv files
df_rohan = pd.read_csv( path + path_to_data + files [1])
df_ignacio = pd.read_csv( path + path_to_data + files [0])

# check outcome
print(df_rohan.head(), df_ignacio.head())


# instantiate the moving_avg() class ONLY once
moving_average_instance = moving_avg()


""" NOTE: either Rohan's data, OR ignacio's data must be uncommented. Uncommenting both will result in
mixed results for the dataset that comes second
"""


# for ignacio's data
"""for datapoint in df_ignacio[['accTotal','gyrTotal','date']].values:
    datapoint_instance = average(datapoint[0:2],datapoint[2])
    

moving_average_instance.plot(plot_original = True)"""



# for rohan's data

for datapoint in df_rohan[['accTotal','gyrTotal','date']].values:
    datapoint_instance = average(datapoint[0:2],datapoint[2])

moving_average_instance.plot(plot_original = True)


