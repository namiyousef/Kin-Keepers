"""

Author: Yousef Nami
Version control:
    v.0.0.0.    created file

"""

# Dependencies:

# class 'moving_avg'

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt



class moving_avg:

    """
    a class used to store a moving average values, parameters and methods
    
    Dependencies:
    -------------
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    

    Attributes:
    -----------
    
    data ( class var ): [*[*float]]
        stores all the datapoints for each window
        
    time_frame_start ( class var ): [datetime]
        the start of the moving average window
    
    time_stamps ( class var ): [*[*datetime]]
        stores the timestamps for each data point within it's window
        
    averages ( class var ): [*float]
        stores the values of the moving average for each window    
        
    time_frame ( optional - 5 ): int
        the length of the moving window in units of hours
        
    weight ( optional - (0.0, 0.75) ): (float, float)
        weight to apply to numbers greater than the specified quartile
    
        
    Methods:
    --------
    __init__( self, time_frame = 5, weight = (0.0, 0.75)):
    initialises class based on inputs; converts 'time_frame' to seconds

    
    average( self ):
        calculates the averages for each moving window
        
    plot( self, figsize = (16,8), labels = ('gyrTotal', ' accTotal') ): 
        plots the averages against the start time of the moving moving
        
    create_times ( self ):
        creates a list of the correct plotting index (i.e. real continuous time) with data gaps
    """
    data = [[]]
    time_frame_start = []
    time_stamps = [[]]
    all_data = [[]]
    
    # note there is a danger in using class variables because they 'save' every instantiations values!
    # note, your reshapes will need to change so that they can adapt for more than 2D data!!!
    def __init__( self, time_frame = 5, weight = (0.0, 0.75)):
        self.time_frame = time_frame*3600
        self.weight = weight
        self.averages = []

    def plot( self, figsize = (16,8), labels = ('gyrTotal', ' accTotal'), plot_original = False ):
        # plot original tells it to plot the original as well!
        
        averages = np.asarray(self.averages).reshape((-2,2))
        fig = plt.figure(figsize = figsize)
        # need create times here
        real_time = self.create_times(self.time_frame_start)
        all_data = np.asarray(self.all_data).reshape((-2,2))


        for i in range(averages.shape[1]): # this defines the columns
            for row in range(2):
                
                fig.add_subplot(2,averages.shape[1],averages.shape[1]*row+i+1)

                    
                if row%2 == 0:
                    if plot_original == True:
                        plt.plot([j for j in range(len(self.time_stamps[-1]))],all_data[:,i],'.')
                        
                    plt.plot([j for j in range(len(self.time_frame_start))],averages[:,i],'.')
                    plt.xlabel('discontinuous time, $\hat{t}$, in seconds')


                else:
                    if plot_original == True:
                        real_time_all = self.create_times(self.time_stamps[-1])
                        plt.plot(real_time_all,all_data[:,i],'.')

                        
                    plt.plot(real_time,averages[:,i],'.', markersize = 2)
                    plt.xlabel('continuous time, ${t}$, in seconds')                



                plt.ylabel('average {} in a {} hour window'.format(labels[i],int(self.time_frame/3600)))
                # need to add xticks
                

                


        plt.show()
        
    def create_times ( self, time_array ):
        real_time = [0]
        for time in time_array[1:]:
            real_time.append((time - time_array[0]).total_seconds())
        return real_time
            
        
    def average( self ):   
        for window in self.data:
            window = np.asarray(window).reshape(-2,2)
            self.averages.append([
                window[:,index].mean() for index in range(window.shape[1])
            ])
            
        if (len(self.averages) > 1):
            if (self.averages[-1][0] < self.averages[-2][0]):
                print('risk')

class average(moving_avg):
    """
    
    Dependencies:
    -------------
    moving_avg (class)
    
    Attributes:
    -----------
    
    datapoint: [*float]
        datapoint to be considered for averaging, length --> degrees of freedom
    
    time: str
        time data point is recorded in the format 'YYYY-mm-dd HH:MM:SS'
        
    Methods:
    --------
    
    __init__(self, datapoint, time):
        initilises class; converts time to datetime; stores new datapoint and time;
        if new time exceeds average window, creates new storage location
        
    """



    
    def __init__(self,datapoint,time):
        super().__init__() # is this necessary?

        self.datapoint = datapoint
        self.time_stamps[-1].append(dt.datetime.strptime(time,'%Y-%m-%d %H:%M:%S'))
        
        self.all_data[-1].append(datapoint)
        
        if not self.time_frame_start:
            self.time_frame_start.append(self.time_stamps[-1][-1])
            
        if (self.time_stamps[-1][-1] - self.time_frame_start[-1]).total_seconds() < self.time_frame:
            pass
        else:
            counter = 0
            for i,time in enumerate(self.time_stamps[-1]):
                if time not in self.time_frame_start:
                    counter += 1
                    self.data.append(
                        #[[0,0] for j in self.data[-1][:i]] + self.data[-1][i:] # this is correct, but something else is 
                        # amiss!!!!
                        # this used to just be: self.data[-1][1:]
                        [[0,0] for j in self.data[-1][:counter]] + self.data[-1][counter:]
                    )
                    self.time_frame_start.append(time)
                    
            
            self.time_frame_start.append(self.time_stamps[-1][-1])
            self.data.append([])
            
            
            
        self.data[-1].append([
            point for point in datapoint
        ]) # should account for the 'weights' that you've specified here, might require moving the average method
        
        #self.average() # this average only needs to average the most recent index, otherwise your scripts will
        # take ages to complete
        
# what the class is still missing is the 'decision making process', so if an average is lower than 
        
        
        
