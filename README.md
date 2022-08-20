# Neural Network Training with Difference of Convex Programming (DCP)

This is an ongoing research project (in joint work with Chuan He) featuring neural network training with difference of convex probramming (DCP). When training neural networks, SGD usually stagnates at a bad local minima after some training. We aim to use DCP to avoid trapping in such solutions. For this purpose, we proposed three versions of the algorithm specifically for training neural network, aiming for different kinds of situations:

* One-pass DC algorithm: deals with small DC subproblem
* Two-pass DC algorithm: deals with large DC subproblem
* Stochastic DC algorithm: scales to large training size

We have derived specific forms for each of them, but up to this point, only the first version is documented in this Python project. 

## Documentation and examples

The API documentation (all exported functions) of this package can be found in [index.html](https://github.umn.edu/liu00980/nndcp/tree/master/docs/_build/html/index.html). For an illustration of basic usage for some key functions, check [examples.ipynb](https://github.umn.edu/liu00980/nndcp/tree/master/examples.ipynb) in the root directory. I haven't found a place to host the documentation privately, so in order to view [index.html](https://github.umn.edu/liu00980/nndcp/tree/master/docs/_build/html/index.html), one will have to download it and view it in a web browser.

## Installation

Run the following to install Python pakcage `nndcp` locally:

```bash
# on a virtualenv or conda environment
git clone https://github.umn.edu/liu00980/nndcp.git
cd nndcp
pip install -e .[dev]
```

## Usage

`nndcp` currently provides four modules:

### `data`

Module `data` contains three processed real datasets [Communities and Crime Data Set](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime), [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) and [California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html), ready for training. We provide three forms for each dataset: `pandas` dataframe with missing values imputed as medians, predictors (the first colum is all-one vector) with numerical variables standardized and categorical variables one-hot encoded, and standardized response. To get them, call:

```python
calhousing = load_calhousing()
# pandas dataframe
calhousing_df = calhousing["calhousing_df"]
# predictors
X = calhousing["X"]
# response
Y = calhousing["Y"]
```

### `utils.util`

Subpackage `utils` is designed for storing some small but useful functions. For now, it only has one module `utils.util` consisting of functions for transformation between different Python objects, intermediate calculation and etc. Please refer to the documentation of `nndcp` to check functions provided by this module.

### `SGDtraining`

This module provides a pipe line for training neural network with SGD. It contains one and only one function `trainnn_sgd` with the following semantics:

```python
# example of using trainnn_sgd
trainnn_sgd(
    train_ds = train_dataset,
    val_ds = validation_dataset,
    batch_size = batch_size,
    train_model = train_model,
    loss_fn = torch.nn.MSELoss(),
    nepochs = 100,
    lr = 0.1,
    verbose = True,
    show = 10
)
```

For now, it only supports plain SGD, which uses fix learning rate for training. In future, customized optimizer and learning rate scheduler will be added to the features, and this will allow us to use SGD variants, adaptivive learning rates, and regularizations when training neural networks.


### `DCshallow`

This module implement the one-pass DC algorithm with shallow ReLU network, in which case, the DC subproblem is small. To performt the algorithm, call:

```python
# example of using trainnn_dcshallow
trainnn_dcshallow(
    train_ds = train_dataset,
    val_ds = validation_dataset,
    width = width,
    train_model = train_model,
    lmda = 10.0,
    iterations = 20,
    verbose = True,
    solver = 'MOSEK'
)
```

The starting poing can either be a SGD-trained network model, or simply initialization. When the network is deep, one-pass algorithm might not be suitable due to computation efficiency.

## Developing `nndcp`

To install `nndcp`, along with the tools you need to develop and run tests, run the following in your virtualenv or conda environment:

```bash
pip install -e .[dev]
```

This command builds a symbolic link to your Python project and install your package locally. Compared to usual `pip install`, this avoids transporting between your local project and the one in `site-packages`. It is helpful if you are constantly trying and testing the package.


## Testing with `pytest`

One can test (as a developer) the usage of `trainnn_dcshallow` and `trainnn_sgd` with `pytest`. First, we need to install `pytest`:

```bash
pip install pytest
```

Then under `tests/`, run the following to check two existed test files:

```bash
pytest -s
```
