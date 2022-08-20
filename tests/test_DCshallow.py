from DCshallow import *
import torch

# size of the problem
n = 50
p = 4
width = 4
sigma = 0.1
train_pct = 0.8
train_size = int(n * train_pct)
val_size = n - train_size

# true underlying nn model
# torch.manual_seed(0)
truemodel = torch.nn.Sequential(
    torch.nn.Linear(p, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, 1),
    torch.nn.Flatten(0, 1)
)

# prepare dataset, dataloaders for model training
dataset = normal_nn(n, p, truemodel, sigma)
train_ds, val_ds = splitdataset(dataset, train_pct = train_pct)   

# train shallow NN with DC algorithm
print("\n strat DC training for 10 iterations with lambda = 10.0")
U0, alpha0, train_dcloss, val_dcloss, exactpenalty = trainnn_dcshallow(train_ds, val_ds, width, truemodel, iterations = 10)



# test whether trainnn_sgd() gives the right format
def test_trainnn_dcshallow_format():
    assert isinstance(train_dcloss, list)
    assert isinstance(val_dcloss, list)
    assert isinstance(exactpenalty, float)

# test whether training loss and validation loss match with the output
def test_trainnn_dcshallow_match():
    X, y = extract(train_ds)
    X_val, y_val = extract(val_ds)
    assert eloss(relunn(X, [U0], alpha0), y, 1) == train_dcloss[-1]
    assert eloss(relunn(X_val, [U0], alpha0), y_val, 1) == val_dcloss[-1]