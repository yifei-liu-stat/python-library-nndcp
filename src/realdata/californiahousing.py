# main reference:
# notebook: https://jmyao17.github.io/Kaggle/California_Housing_Prices.html
# from Luis Torgo's page: https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
# kaggle: https://www.kaggle.com/camnugent/california-housing-prices
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from utils.util import *

data = pd.read_csv("dataset/housing.csv")

data.head()
data.info()
data.describe()

# predictors hava various spread and scale
plt.figure()
data.hist(bins = 50, figsize = (20, 15))
plt.savefig("dataset/test.png")

# house values depend heavily on locations and population
plt.figure()
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3,
         s=data['population']/100, label='population',   # set symbol size on population
         c=data['median_house_value'],                  #  set symbol color on house value    
         cmap=plt.get_cmap('jet'),      
         colorbar=True,
         figsize=(10,7))
plt.legend()
plt.savefig("dataset/test.png")

# pairwise feature correlation
plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.savefig("dataset/test.png")

# most correlateted features
# (to house value: income, total_rooms, house age)
corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# missing data
report_missing_data(data)
# deal with it
median = data['total_bedrooms'].median()
data["total_bedrooms"].fillna(median, inplace=True)
report_missing_data(data)

data.to_pickle('dataset/calhousing_df.pkl')
# output = pd.read_pickle("a_file.pkl")

# features and labels
X = data.drop("median_house_value", axis = 1)
Y = data["median_house_value"].copy().to_frame()


# build pipelines and preporcess predictors
cat_attribs = ['ocean_proximity']
num_attribs = list(X.drop("ocean_proximity", axis=1).columns)

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
X = np.hstack((np.ones([X.shape[0], 1]), X))
np.save('dataset/calhousing_X.npy', X)
# output = np.load('dataset/calhousing_X.npy')

# build pipelines and preporcess response
cat_attribs = []
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
np.save('dataset/calhousing_Y.npy', Y)
# output = np.load('dataset/calhousing_Y.npy')


