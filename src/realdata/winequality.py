# main reference: 
# Notebook: https://www.kaggle.com/sgus1318/wine-quality-exploration-and-analysis
# UCI: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# Kaggle: https://www.kaggle.com/sgus1318/winedata
# import pandas as _pd, numpy as _np, matplotlib.pyplot as _plt
from utils.util import *


data_red = _pd.read_csv('dataset/winequality-red.csv', sep = ";")
data_white = _pd.read_csv('dataset/winequality-white.csv', sep = ";")
data_red['color'] = 'R'
data_white['color'] = 'W'
data = _pd.concat([data_red, data_white], axis = 0)

data.head()
data.tail()
data.info()
data.describe()

report_missing_data(data)

data.to_pickle('dataset/wine_df.pkl')
# output = _pd.read_pickle('dataset/wine_df.pkl')


# features and labels
X = data.drop("quality", axis = 1)
Y = data["quality"].copy().to_frame()


# build pipelines and preporcess predictors
cat_attribs = ['color']
num_attribs = list(X.drop("color", axis=1).columns)

num_pipeline = Pipeline([
               ('selector',DataFrameSelector(num_attribs)),      
               ('std_scaler',StandardScaler()), 
                ]) 

# build categorical pipeline
cat_pipeline = Pipeline([
                  ('selector',DataFrameSelector(cat_attribs)),
                  ('cat_encoder',CategoricalEncoder(encoding='onehot-dense')),
              ])

# concatenate all the transforms using "FeatureUnion"
pipelines = FeatureUnion(transformer_list=
                             [ 
                              ('num_pipeline',num_pipeline),
                              ('cat_pipeline',cat_pipeline),
                             ])

# prepared predictors as numpy.ndarray()
X = pipelines.fit_transform(X)
X = _np.hstack((_np.ones([X.shape[0], 1]), X))
_np.save('dataset/wine_X.npy', X)
# output = _np.load('dataset/wine_X.npy')

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
_np.save('dataset/wine_Y.npy', Y)
# output = _np.load('dataset/wine_Y.npy')
