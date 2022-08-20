import numpy as _np, pandas as _pd
import pkg_resources as _pr

def load_wine(): 
    """Returns processed Wine Quality Data Set.

    See https://archive.ics.uci.edu/ml/datasets/Wine+Quality for detailed descritpion.


    Returns:
        dict: 
        A dictionary with the following keys:

        * wine_df (*pandas.core.frame.DataFrame*)
            A preprocessed pandas dataframe;
            Compared to the original ones, we merge the two datasets (white wine and red wine) and impute missing values with medians.

        * X (*_np.ndarray*)
            Design matrix with all numerical predictors standardized, and all categorical predictors one-hot encoded.
            Also, the first column is all-one vector.

        * Y (*_np.ndarray*)
            Standardized response.

    Examples:
        >>> from data import load_wine
        >>> wine_df = load_wine()["wine_df"]
        >>> X = load_wine()["X"]
        >>> Y = load_wine()["Y"]
    """ 

    wine_df = _pd.read_pickle(_pr.resource_stream(__name__, 'realdata/wine_df.pkl'))
    X = _np.load(_pr.resource_stream(__name__, 'realdata/wine_X.npy'))
    Y = _np.load(_pr.resource_stream(__name__, 'realdata/wine_Y.npy'))
    return {"wine_df": wine_df, "X": X, "Y": Y}


def load_crime():
    """Returns processed Communities and Crime Data Set.

    See https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime for detailed descritpion.


    Returns:
        dict: 
        A dictionary with the following keys:

        * crime_df (*pandas.core.frame.DataFrame*)
            A preprocessed pandas dataframe;
            Compared to the original ones, we delete some non-predictive variables and impute missing values with medians.

        * X (*_np.ndarray*)
            Design matrix with all numerical predictors standardized, and all categorical predictors one-hot encoded.
            Also, the first column is all-one vector.

        * Y (*_np.ndarray*)
            Standardized response.

    Examples:
        >>> from data import load_crime
        >>> crime_df = load_crime()["crime_df"]
        >>> X = load_crime()["X"]
        >>> Y = load_crime()["Y"]
    """    
    crime_df = _pd.read_pickle(_pr.resource_stream(__name__, 'realdata/crime_df.pkl'))
    X = _np.load(_pr.resource_stream(__name__, 'realdata/crime_X.npy'))
    Y = _np.load(_pr.resource_stream(__name__, 'realdata/crime_Y.npy'))
    return {"crime_df": crime_df, "X": X, "Y": Y}



def load_calhousing():
    """Returns processed California Housing Data Set. 

    See https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html for detailed descritpion.


    Returns:
        dict: 
        A dictionary with the following keys:

        * calhousing_df (*pandas.core.frame.DataFrame*)
            A preprocessed pandas dataframe;
            Compared to the original ones, we impute missing values with medians.

        * X (*_np.ndarray*)
            Design matrix with all numerical predictors standardized, and all categorical predictors one-hot encoded.
            Also, the first column is all-one vector.

        * Y (*_np.ndarray*)
            Standardized response.

    Examples:
        >>> from data import load_calhousing
        >>> calhousing_df = load_calhousing()["calhousing_df"]
        >>> X = load_calhousing()["X"]
        >>> Y = load_calhousing()["Y"]
    """    
    calhousing_df = _pd.read_pickle(_pr.resource_stream(__name__, 'realdata/calhousing_df.pkl'))
    X = _np.load(_pr.resource_stream(__name__, 'realdata/calhousing_X.npy'))
    Y = _np.load(_pr.resource_stream(__name__, 'realdata/calhousing_Y.npy'))
    return {"calhousing_df": calhousing_df, "X": X, "Y": Y}

