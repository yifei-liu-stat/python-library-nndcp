from SGDtraining import *
from torch.utils.data.dataloader import DataLoader
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


train_model = torch.nn.Sequential(
    torch.nn.Linear(p, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, 1),
    torch.nn.Flatten(0, 1)
)
batch_size = 10
print("\nstart SGD training with learning rate 0.1 for 10 epochs")
train_loss, val_loss, train_model = trainnn_sgd(train_ds, val_ds, batch_size, train_model, nepochs = 10, show = 1)
# to continue training
print("\nstart SGD training for the second round with learning rate 0.01 for another 10 epochs")
train_loss2, val_loss2, train_model = trainnn_sgd(train_ds, val_ds, batch_size, train_model, lr = 0.01, nepochs = 10, show = 1)
train_loss = train_loss + train_loss2
val_loss = val_loss + val_loss2



# test whether trainnn_sgd() gives the right format
def test_trainnn_sgd_format():
    assert isinstance(train_loss, list)
    assert isinstance(val_loss, list)
    assert isinstance(train_model, torch.nn.modules.container.Sequential)

# test whether training loss and validation loss match the trained model
def test_trainnn_sgd_match():
    wtrain_loader = DataLoader(train_ds, train_size)
    wval_loader = DataLoader(val_ds, val_size)
    assert wholeloss(wtrain_loader, train_model, loss_fn = torch.nn.MSELoss()) == train_loss[-1]
    assert wholeloss(wval_loader, train_model, loss_fn = torch.nn.MSELoss()) == val_loss[-1]