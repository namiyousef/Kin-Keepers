import pandas as pd
import os

def directory_to_df(paths, exclude = [None], filetype = '.csv',ignore_index = True, exception = '_repet'):
    """ 
    function for concatenating all the files in a directory into a dataframe
    
    Dependecnies:
    -------------
    import os
    import pandas as pd
    
    
    Attributes:
    -----------
    paths: [*str]
        a list that contains path(s) to the desired directories
    
    exclude ( optional - [None] ): [*str]
        a list that contains filenames to exclude
    
    filetype ( optional - '.csv' ): str
        the file extension for the files that you are interested in
        
    ignore_index ( optional - True): bool
        tells pandas to ignore index from the file or not
    
    exception ( oprtional - '_repet' ): str
        any time a filename includes this string, it is treated differently (note that this was done because
        the data for this particular project came in odd forms due to changes in the movement API)

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
    
    """
    Function that does a specific taks on a dataframe (to be used with 'directory_to_df'). In this case, it
    deals with the repetition of the last column that some .csv files had, which led to a mislabelling of the
    column names.
    
    Dependencies:
    -------------
    import pandas as pd
    

    Attributes:
    -----------
    
    df: pd.dataframe
        a dataframe of items that are read incorrectly, and thus require special treatment to get them
        in the correct format
    """
    
    columns = df.columns.values.tolist()
    columns.remove('date')
    df.drop('gyrZ',inplace = True, axis = 1)
    df.columns = columns
    df.reset_index(inplace = True)
    df.rename(columns= {'index':'date'},inplace = True)
    return df
    
