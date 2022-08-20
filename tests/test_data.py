from data import *
import pandas, numpy

def test_load_dataset():
    wine = load_wine()
    crime = load_crime()
    housing = load_calhousing()
    assert list(wine.keys()) == ['wine_df', 'X', 'Y']
    assert isinstance(crime['crime_df'], pandas.core.frame.DataFrame)
    assert isinstance(housing['X'], numpy.ndarray)