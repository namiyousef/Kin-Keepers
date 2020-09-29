"""
Author: Yousef Nami
Date: 29.09.2020

Prerequisites:
    - you should know the filter thresholds that you want to apply
   
What this file will do:
    - this file will generate 'filtered' data from Ignacio's and Rohan's 
    data by applying thresholds found from clustering.py
"""

# important libraries 

from file_reading import directory_to_df, special_treatment
import numpy as np
from clustering import polynomial

path = ('/Users/yousefnami/KinKeepers/ProjectAI/biometrics/'
        'movement_data_analysis/Data/') #path to the data folder

subdir_names = ['Ignacio/','Rohan/'] # note, this must be a list!

for i in range(len(subdir_names)):
    subdir_names[i] = path + subdir_names[i]
    
# sanity check
print(subdir_names)

df, filenames = directory_to_df(subdir_names)
print(filenames) # names of the files that were read

print(df.files.unique()) # the filenumber corresponds to the filename. So for example:
# file 1 --> 14092020_repet.csv

# create total acceleration and gyration vectors

df['accTotal'] = np.sqrt(np.power(df[['accX','accY','accZ']],2).sum(axis = 1))
df['gyrTotal'] = np.sqrt(np.power(df[['gyrX','gyrY','gyrZ']],2).sum(axis = 1))

print(df.shape)
# filter the data based on the thresholds defined in polynomial !

df_filtered = df[df.accTotal > polynomial(1/df.gyrTotal)]
print(df_filtered.shape)

# df_filtered.to_csv(path) # to save the data as a CSV file

# note that if you want to calculate filtered data separately, i.e. for ignacio and rohan, 
# then you must have subdir_names be a list of length == 1 with the correct name on it,
# i.e. for Ignacio:
# subdir_names = ['Ignacio/'].
# The reason why they have been combined here is for the purposes of 'clustering_guide.py'`


