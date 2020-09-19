# Kin-Keepers
A repository that contains work done for two AI projects at Kin Keepers.
**Note:** this was not initially designed to be a repository, so within the notebooks, paths specified may no longer be valid.

# Problem Description:

There currently exists an Arduino that constantly sends movement data (accelerometer and gyroscopic) to be written onto a CSV file. Much of this data is uninformative, since it just shows that the user is stationary.
The goals of this project are as follows:
- find thresholds that partition movement data into 'significant' and 'non-significant' (using machine learning techniques or otherwise)
- use this 'filtered' data to find anomalies in time-series acceleration and gyration data (i.e. if the user has lower acceleration / gyration *on average* then one should trigger an alarm) using machine learning methods or otherwise

A vote was taken on what movements were 'significant' and which were non-significant. The results are shown below:

**Significant data:**
- standing up and sitting down straight after
- fetching remote control that is not at an arm's reach
- lifting and lowering one's lower leg to stimulate blood circulation
- walking
- falling

**Non-significant data:**
- crossing one's arms
- crossing one's legs
- switching seating positions
- moving from a sitting position to a lying one

Note that the above were used as 'testing criteria' for the clustering model developed.

# Contents of repository:

**Data:** folder containing all the data used for either research, flow detection project or the movement data project

**Models:** includes ML models saved as .pkl files

**Research:** includes work done while I had no access to data (i.e. preparation for when I did actually get the data). Note that some of the work is incomplete.

**Movement Clustering movement data:** This folder The first stage of the project involved actually clustering this data to separate 'significant' movements from stationary datapoints. The first couple of notebooks are exploratory, then there is modelling, verification and finally finding numerical thresholds to be incorporated on the arduino itself (so that there is no constant data transfer).

**Anomaly Detection in significant movements:** This folder is currently empty


**Flow Detection:** not part of the problem description from above. Brief: you have access to simulation data that shows water flow inside a dwelling as a function of time. Can you determine anomalies in this data? (underflow, leak or open tap). Much of my work here is exploratory as I was in fact providing insight into another team member's problems.
