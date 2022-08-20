# main reference:
# notebook: https://www.kaggle.com/kkanda/analyzing-uci-crime-and-communities-dataset
# UCI: https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
# kaggle: https://www.kaggle.com/kkanda/analyzing-uci-crime-and-communities-dataset
import pandas as pd, numpy as np
from utils.util import *

crimedata = pd.read_csv("dataset/crimedata.csv",sep='\s*,\s*',
                        encoding='latin-1',engine='python',na_values=["?"])
crimedata = crimedata.rename(columns={'Êcommunityname':'communityName'})

# first 5 columns are non-predictive columns
nonpredlist = list(crimedata.columns)[0:5]
crimedata = crimedata.drop(nonpredlist, axis = 1)

# impute medians for missing values
missing_info = report_missing_data(crimedata)
missing_attr = list(missing_info[missing_info['Total'] > 0].index)
crimedata.fillna(crimedata[missing_attr].median(), inplace=True)

crimedata.head()
crimedata.tail()
crimedata.info()
crimedata.describe()

data.to_pickle('dataset/crime_df.pkl')
# output = pd.read_pickle('dataset/crime_df.pkl')


# predictors and response
X = crimedata.drop("ViolentCrimesPerPop", axis = 1)
Y = crimedata["ViolentCrimesPerPop"].to_frame()

# build pipelines and preporcess predictors
num_attribs = list(X.columns)

num_pipeline = Pipeline([
               ('selector',DataFrameSelector(num_attribs)),      
               ('std_scaler',StandardScaler()), 
                ]) 

# concatenate all the transforms using "FeatureUnion"
pipelines = FeatureUnion(transformer_list=
                             [ 
                              ('num_pipeline',num_pipeline)
                             ])

# prepared predictors as numpy.ndarray()
X = pipelines.fit_transform(X)
X = np.hstack((np.ones([X.shape[0], 1]), X))
np.save('dataset/crime_X.npy', X)
# output = np.load('dataset/crime_X.npy')

# build pipelines and preporcess response
num_attribs = list(Y.columns)

num_pipeline = Pipeline([
               ('selector',DataFrameSelector(num_attribs)),      
               ('std_scaler',StandardScaler()), 
                ]) 

# concatenate all the transforms using "FeatureUnion"
pipelines = FeatureUnion(transformer_list=
                             [ 
                              ('num_pipeline',num_pipeline)
                             ])
# prepared response as numpy.ndarray()
Y = pipelines.fit_transform(Y)
np.save('dataset/crime_Y.npy', Y)
# output = np.load('dataset/crime_Y.npy')
