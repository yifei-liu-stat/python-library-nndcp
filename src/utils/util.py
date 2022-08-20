import pandas as _pd, numpy as _np, matplotlib.pyplot as _plt
import cvxpy as _copy, scs as _scs, mosek as _mosek
import copy as _copy

import torch as _torch

from torch.distributions.multivariate_normal import MultivariateNormal as _MultivariateNormal
from torch.distributions.normal import Normal as _Normal

from torch.utils.data.dataloader import DataLoader as _DataLoader
from torch.utils.data import random_split as _random_split

from scipy import sparse as _sparse
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.utils import check_array as _check_array
from sklearn.preprocessing import LabelEncoder as _LabelEncoder

def relunn(X, Ulist, alpha):
    """Output of a ReLU NN, that is

    .. math::
        ((\ldots (X U_1)_+ \ldots)_+ U_L)_+ \\alpha

    Args:
        X (numpy.ndarray): design matrix of shape (n, p)
        Ulist (list): list of weight matrices of NN, starting from the first hidden layer, that is [U_1, U_2, ..., U_L].
        alpha (numpy.ndarray): weights of output layers, of shape (d_L, d_{L+1})

    Returns:
        numpy.ndarray: output matrix of shape (n, d_{L+1}) with d_{L+1} as the dimension of output.
    """    
    y = _copy.deepcopy(X)
    for U in Ulist:
        y = _np.maximum(0, y @ U)
    
    y = y @ alpha
    
    return y 

def eloss(outputs, labels, q, mean = True):
    """Calculate average elementwise losses measured by (take matrix for example)

    .. math::
        \\frac{1}{N} \sum_{i, j} |X_{ij} - Y_{ij}|^q

    That is, we flatten two inputs as vectors, take the difference, calculate vector q-norm^q, and take the average as the final measurement.
    When q is 1, we have mean absolute loss (MAE); when q is 2, we have mean squared loss (MSE).

    Args:
        outputs (numpy.ndarray): a numpy array, can have any shape
        labels (numpy.ndarray): a numpy array, can have any shape
        q (int): exponent for the vector norm
        mean (bool, optional): whether to take average over all scalar elements. Defaults to True.

    Returns:
        float: if ``mean = True``, then return the average elementwise loss; otherwise, return the sum of elementwise losses.
    
    Examples:
        >>> from utils import util
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4]])
        >>> Y = np.array([[1, 2], [3, 5]])
        >>> util.eloss(X, Y, 1, mean = True)
        0.25
        >>> util.eloss(X, Y, 1, mean = False)
        1.0
    """    
    s = outputs.flatten()
    y = labels.flatten()

    if mean == True:
        return _np.linalg.norm(s - y, q) ** q / s.shape[0]
    else:
        return _np.linalg.norm(s - y, q) ** q


# a_+^\T b = f(a, b) - g(a, b)
def f(a, b):
    """Evaluate the value of `f(a, b)` at `(a, b)`, where

    .. math::
        f(a, b) = \\frac{1}{2}(||a_+ + b_+||^2 + ||a_+||^2 + ||b_-||^2) 

    Here both *a* and *b* are both vectors with same dimension.
    Note that function *f(a, b)* is used to construct the DC decomposition of the following

    .. math::
        a_+^{\mathsf T} b = f(a, b) - g(a, b)

    *g(a, b)* is defined in the following way

    .. math::
        g(a, b) = \\frac{1}{2}(||a_+ + b_-||^2 + ||a_+||^2 + ||b_+||^2) 

    Args:
        a (numpy.ndarray): first argument, will be flattened 
        b (numpy.ndarray): second argument of the same size, will also be flattened

    Returns:
        float: function value of *f* at *(a, b)*
    """     
    x = a.flatten()
    y = b.flatten()
    temp = 0
    temp = temp + _np.linalg.norm(_np.maximum(0, x) + _np.maximum(0, y), 2) ** 2
    temp = temp + _np.linalg.norm(_np.maximum(0, x), 2) ** 2
    temp = temp + _np.linalg.norm(_np.maximum(0, -y), 2) ** 2
    temp = temp / 2

    return temp

def g(a, b):
    """Evaluate the value of `g(a, b)` at `(a, b)`

    Args:
        a (numpy.ndarray): first argument, will be flattened 
        b (numpy.ndarray): second argument of the same size, will also be flattened

    Returns:
        float: function value of `g` at `(a, b)`
    """     
    x = a.flatten()
    y = b.flatten()
    temp = 0
    temp = temp + _np.linalg.norm(_np.maximum(0, x) + _np.maximum(0, -y), 2) ** 2
    temp = temp + _np.linalg.norm(_np.maximum(0, x), 2) ** 2
    temp = temp + _np.linalg.norm(_np.maximum(0, y), 2) ** 2
    temp = temp / 2

    return temp


# subgradient of f and g w.r.t. a and b
def fa(a, b):
    """Calculate subgradient of `f(a, b)` w.r.t. `a`

    Args:
        a (numpy.ndarray): first argument, will be flattened 
        b (numpy.ndarray): second argument of the same size, will also be flattened

    Returns:
        numpy.ndarray: 1d numpy array
    """       
    x = a.flatten()
    y = b.flatten()
    return 2*_np.maximum(0, x) + _np.multiply(_np.sign(_np.maximum(0, x)), _np.maximum(0, y))

def fb(a, b):
    """Calculate subgradient of `f(a, b)` w.r.t. `b`

    Args:
        a (numpy.ndarray): first argument, will be flattened 
        b (numpy.ndarray): second argument of the same size, will also be flattened

    Returns:
        numpy.ndarray: 1d numpy array
    """       
    x = a.flatten()
    y = b.flatten()
    return y + _np.multiply(_np.sign(_np.maximum(0, y)), _np.maximum(0, x))

def ga(a, b):
    """Calculate subgradient of `g(a, b)` w.r.t. `a`

    Args:
        a (numpy.ndarray): first argument, will be flattened 
        b (numpy.ndarray): second argument of the same size, will also be flattened

    Returns:
        numpy.ndarray: 1d numpy array
    """       
    x = a.flatten()
    y = b.flatten()
    return 2*_np.maximum(0, x) + _np.multiply(_np.sign(_np.maximum(0, x)), _np.maximum(0, -y))

def gb(a, b):
    """Calculate subgradient of `g(a, b)` w.r.t. `b`

    Args:
        a (numpy.ndarray): first argument, will be flattened 
        b (numpy.ndarray): second argument of the same size, will also be flattened

    Returns:
        numpy.ndarray: 1d numpy array
    """       
    x = a.flatten()
    y = b.flatten()
    return y - _np.multiply(_np.sign(_np.maximum(0, -y)), _np.maximum(0, x))



# constant matrix (and vectors) in component (3)
A31 = lambda d1: _np.vstack(
  [
  _np.hstack([1, _np.zeros(3 * d1)]), \
  _np.hstack([_np.zeros((d1, 1)), _np.identity(d1), _np.identity(d1), _np.zeros((d1, d1))]), \
  _np.hstack([_np.zeros((d1, 1)), _np.identity(d1), _np.zeros((d1, 2 * d1))]), \
  _np.hstack([_np.zeros((d1, 1 + 2 * d1)), _np.identity(d1)])
  ]
)

b31 = lambda d1: _np.hstack([1 / 4, _np.zeros(3 * d1)])

A32 = lambda d1: _np.vstack(
  [
  _np.hstack([1, _np.zeros(3 * d1)]), \
  _np.hstack([_np.zeros((d1, 1)), _np.identity(d1), _np.zeros((d1, d1)), _np.identity(d1)]), \
  _np.hstack([_np.zeros((d1, 1)), _np.identity(d1), _np.zeros((d1, 2 * d1))]), \
  _np.hstack([_np.zeros((d1, 1 + d1)), _np.identity(d1), _np.zeros((d1, d1))])
  ]
)

b32 = lambda d1, c: _np.hstack([2 * c + 1 / 4, _np.zeros(3 * d1)])





# # Chuan's version on functions above, used for real dataset
# # three-layer ReLU map
# ReLU = lambda X, W1, W2, w3:  w3 @ _np.maximum( W2 @ _np.maximum( W1 @ X , 0) , 0 )
# # MAE for three-lay ReLU map
# l1_loss = lambda X, y, W1, W2, w3: _np.linalg.norm(ReLU(X, W1, W2, w3) - y, 1) / y.shape
# # gradient info: w^\T z = f(w, z) - g(w, z)
# fw = lambda w, z: w + _np.multiply(_np.sign(_np.maximum(0, w)), _np.maximum(0, z))
# fz = lambda w, z: 2*_np.maximum(0, z) + _np.multiply(_np.sign(_np.maximum(0, z)), _np.maximum(0, w))
# gw = lambda w, z: w - _np.multiply(_np.sign(_np.maximum(0, -w)), _np.maximum(0, z))
# gz = lambda w, z: 2*_np.maximum(0, z) + _np.multiply(_np.sign(_np.maximum(0, z)), _np.maximum(0, -w))
# # matrix to construct SOC constraints
# A = lambda d: _np.vstack(
#     [
#     _np.hstack([_np.identity(d), _np.zeros((d,d)), _np.identity(d), _np.zeros((d,1))]), \
#     _np.hstack([_np.zeros((d,d)), _np.identity(d), _np.zeros((d,d+1))]), \
#     _np.hstack([_np.zeros((d,2*d)), _np.identity(d), _np.zeros((d,1))]), \
#     _np.hstack([_np.zeros(3*d), 1])
#     ]
# )



######################################
# On SGD training
######################################


def normal_nn(n, p, nnmodel, sigma):
    """Generate a dataset with the following underlying model,

    .. math::
        y = NN(x) + \sigma * e

    .. math::
        x \sim \mathcal N(0, I_p) \\text{ and } e \sim \mathcal N(0, 1)

    Args:
        n (int): sample size
        p (int): number of features
        nnmodel (_torch.nn.modules.container.Sequential): a neutral network model used for generating the dataset. It should be compatible (in the first layer) with ``p``.
        sigma (float): strength of the noise

    Returns:
        list: a dataset in the form of 
        [(sample_1, label_1), ..., (sample_n, label_n)]

    Examples:
        >>> from utils import util
        >>> import torch
        >>> # size of the problem
        >>> n = 100
        >>> p = 15
        >>> width = 20
        >>> sigma = 0.5
        >>> truemodel = torch.nn.Sequential(
        ... torch.nn.Linear(p, width),
        ... torch.nn.ReLU(),
        ... torch.nn.Linear(width, 1),
        ... torch.nn.Flatten(0, 1)
        ... )
        >>> dataset = util.normal_nn(n, p, truemodel, sigma)
    """

    m1 = _MultivariateNormal(_torch.zeros(p), _torch.eye(p))
    m2 = _Normal(_torch.tensor([0.0]), _torch.tensor([sigma]))

    X = m1.sample([n]) # n\times p input
    e = m2.sample([n]) # n\times 1 error 

    output = nnmodel(X).reshape(n, 1) # scalar output
    output = output.detach() # IMPORTANT, ONLY WANT TENSORS WITHOUT GRAD!!!
    Y = output + e

    dataset = list((X[i, :], Y[i, 0]) for i in range(n))
    return dataset

# # an example for normal_nn()
# n = 10
# p = 3
# width = 3
# sigma = 1
# model = _torch.nn.Sequential(
#     _torch.nn.Linear(p, width),
#     _torch.nn.ReLU(),
#     _torch.nn.Linear(width, 1),
#     _torch.nn.Flatten(0, 1)
# )
# normal_nn(n, p, model, sigma)

def todataset(features, labels):
    """Given design matrix *X* and labels *Y*, return a list-form dataset.

    Args:
        features (numpy.ndarray or _torch.Tensor): design matrix of shape (n, d)
        labels (numpy.ndarray or _torch.Tensor): response matrix of shape (n, p) (usually p = 1)

    Returns:
        list: a dataset in the form of 
        [(sample_1, label_1), ..., (sample_n, label_n)]

    Examples:
        >>> from utils import util
        >>> import numpy as np
        >>> X = np.ones([5, 2])
        >>> Y = np.zeros([5, 1])
        >>> util.todataset(X, Y)
        [(tensor([1., 1.]), tensor(0.)), (tensor([1., 1.]), tensor(0.)), (tensor([1., 1.]), tensor(0.)), (tensor([1., 1.]), tensor(0.)), (tensor([1., 1.]), tensor(0.))]
    """    
    if isinstance(features, _np.ndarray):
        X = _torch.tensor(features, dtype = _torch.float)
    if isinstance(labels, _np.ndarray):
        Y = _torch.tensor(labels, dtype = _torch.float)

    n = X.shape[0]
    dataset = list((X[i, :], Y[i, 0]) for i in range(n))
    return dataset



def extract(dataset):
    """Extract features and labels from a list-formated dataset.

    Args:
        dataset (list): a dataset like [(input_1, label_1), ..., (input_n, label_n)]

    Returns:
        tuple:
        a tuple consisting of the following elements in sequence:

        * X (*_np.ndarray*)
            Design matrix *X* of shape (n, p)

        * Y (*_np.ndarray*)
            Output *Y* of shape (n, 1)
    """    
    n = len(dataset)
    p = dataset[0][0].shape[0]

    X = _np.zeros([n, p])
    Y = _np.zeros([n, 1])
    for i in range(n):
        X[i, :] = dataset[i][0].numpy()
        Y[i, 0] = dataset[i][1].numpy()
    
    return X, Y


def splitdataset(dataset, train_pct = 0.8):
    """Split a dataset into training part and validation part.

    Args:
        dataset (list): dataset like [(input_1, label_1), ..., (input_n, label_n)]. For example, a dataset returned by normal_nn()
        train_pct (float, optional): percentage of training samples. Defaults to 0.8.

    Returns:
        tuple:
        a tuple consisting of the following elements in sequence

        * train_ds (*list*)
            Training dataset in the same form as ``dataset``
        * val_ds (*list*)
            Validation dataset in the same form as ``dataset``            
    """    
    # split dataset into training and validation samples
    sample_size = len(dataset)
    train_size = int(train_pct * sample_size)
    val_size = sample_size - train_size   
    train_ds, val_ds = _random_split(dataset, [train_size, val_size])

    return train_ds, val_ds


def wholeloss(wholeloader, model, loss_fn = _torch.nn.MSELoss()):
    """Calculate loss based on a neural network model and a specified dataset (as a dataloader).

    Args:
        wholeloader (_torch.utils.data.dataloader._DataLoader): a dataloader with only ONE batch specify the whole dataset
        model (_torch.nn.modules.container.Sequential): a neutral network model
        loss_fn (_torch.nn.modules.loss, optional): a loss function specified by _torch. Defaults to _torch.nn.MSELoss().

    Returns:
        float: loss based on the specified dataset

    Examples:
        >>> from utils import util
        >>> from torch.utils.data.dataloader import DataLoader
        >>> # run examples from util.normal_nn() to get train_ds and truemodel
        >>> train_size = len(train_ds)
        >>> wtrain_loader = DataLoader(train_ds, train_size)
        >>> wholeloss(wtrain_loader, truemodel)

    """    
    inputs, labels = next(iter(wholeloader))
    pred = model(inputs)
    return loss_fn(pred, labels).item()



######################################
# Data preprocessing pipeline
######################################

class CategoricalEncoder(_BaseEstimator, _TransformerMixin):
       
    def __init__(self, encoding='onehot', categories='auto', dtype=_np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = _check_array(X, dtype=_np.object, accept_sparse='csc', _copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [_LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = _np.in1d(Xi, self.categories[i])
                if not _np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = _np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = _np.array(_np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        X = _check_array(X, accept_sparse='csc', dtype=_np.object, _copy=True)
        n_samples, n_features = X.shape
        X_int = _np.zeros_like(X, dtype=_np.int)
        X_mask = _np.ones_like(X, dtype=_np.bool)

        for i in range(n_features):
            valid_mask = _np.in1d(X[:, i], self.categories_[i])

            if not _np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = _np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, _copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = _np.array([0] + n_values)
        indices = _np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = _np.repeat(_np.arange(n_samples, dtype=_np.int32),
                                n_features)[mask]
        data = _np.ones(n_samples * n_features)[mask]

        out = _sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
        
class DataFrameSelector(_BaseEstimator, _TransformerMixin):
    def __init__(self,feature_names):
        self.feature_names = feature_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.feature_names].values

# missing data
def report_missing_data(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = dataset.isnull().sum()/total 
        
    missing_data = _pd.concat([total, percent], axis=1, keys=['Total', 'Percent(%)'])
    # missing_data.plot(kind='bar',y='Total',figsize=(10,6),fontsize=20)
    return missing_data
