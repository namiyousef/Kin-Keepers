# Kin-Keepers
A repository that contains work done for two AI projects at Kin Keepers. 
**Note:** this was not initially designed to be a repository, so within the notebooks, paths specified may no longer be valid.

# Contents of repository:

**Data:** folder containing all the data used for either research, flow detection project or the movement data project

**Models:** includes ML models saved as .pkl files

**Research:** includes work done while I had no access to data (i.e. preparation for when I did actually get the data). Note that some of the work is incomplete.

**Movement Detection:** Problem definition: there exists an Arduino that constantly sends movement data (accelerometer and gyroscopic) to be written onto a CSV file. Much of this data is uninformative, since it just shows that the user is stationary. The first stage of the project involved actually clustering this data to separate 'significant' movements from stationary datapoints. The second stage of the project was to use the 'filtered' for anomaly detection (multivariate time-series)

**Flow Detection:** not my main project, but I worked on this for a bit providing some valuable insight. Problem definition: you have access to simulation data that shows water flow inside a dwelling as a function of time. Can you determine anomalies in this data? (underflow, leak or open tap).

