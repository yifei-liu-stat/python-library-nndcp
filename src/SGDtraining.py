# train a neural network with plain SGD
from utils.util import *
import torch as _torch
from torch.utils.data.dataloader import DataLoader as _DataLoader

def trainnn_sgd(train_ds, val_ds, batch_size, train_model, loss_fn = _torch.nn.MSELoss(), nepochs = 100, 
                lr = 0.1, verbose = True, show = 10):
    """Train a neural network model with SGD.

    Args:
        train_ds (list): training dataset in form of [(input_1, label_1), ..., (input_n, label_n)]
        val_ds (list): validation dataset in the same form as ``train_ds``
        batch_size (int): batch size for SGD training
        train_model (torch.nn.modules.container.Sequential): a neural network model used for training
        loss_fn (torch.nn.modules.loss, optional): loss function used in SGD training. Defaults to torch.nn.MSELoss().
        nepochs (int, optional): number of epochs to be trained. Defaults to 100.
        lr (float, optional): learning rate. Defaults to 0.01.
        verbose (bool, optional): whether to print the training loss and validation loss during the training process. Defaults to True.
        show (int, optional): frequency to print the losses. Defaluts to 10 (every 10 epochs).

    Returns:
        tuple:
        a tuple consisting of the following elements in sequence:

        * train_loss (*list*)
            Training loss per SGD epoch.
        
        * val_loss (*list*)
            Validation loss per SGD epoch.

        * train_model (*torch.nn.modules.container.Sequential*)
            Trained model. Note that the function call will change the input ``train_model``.
    """    
    # default optimizer: SGD with lr = 1e-2
    optimizer = _torch.optim.SGD(train_model.parameters(), lr = lr)

    # prepare dataloaders
    train_size = len(train_ds)
    val_size = len(val_ds)
    wtrain_loader = _DataLoader(train_ds, train_size)
    train_loader = _DataLoader(train_ds, batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    val_loader = _DataLoader(val_ds, val_size, num_workers = 4, pin_memory = True)

    # initial losses
    train_loss = [wholeloss(wtrain_loader, train_model, loss_fn)]
    val_loss = [wholeloss(val_loader, train_model, loss_fn)]
    if verbose == True:
        print("epoch:", 0, ";", "training error:", "%0.8f" % train_loss[0],
            ";", "prediction error:", "%0.8f" % val_loss[0]) 
    
    # training process
    for epoch in range(nepochs):      
        for inputs, labels in train_loader:
            outputs = train_model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

        # keep track of train/validation loss
        train_loss.append(wholeloss(wtrain_loader, train_model, loss_fn))
        val_loss.append(wholeloss(val_loader, train_model, loss_fn)) 
        if verbose == True:
            if epoch % show == (show - 1):
                print("epoch:", epoch + 1, ";", "training error:", "%0.8f" % train_loss[-1],
                    ";", "prediction error:", "%0.8f" % val_loss[-1])    

    return train_loss, val_loss, train_model



if __name__ == '__main__':

    import torch
    # size of the problem
    n = 100
    p = 15
    width = 20
    sigma = 0.5

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
    train_ds, val_ds = splitdataset(dataset, train_pct = 0.8)   


    train_model = torch.nn.Sequential(
        torch.nn.Linear(p, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
        torch.nn.Flatten(0, 1)
    )
    batch_size = 10
    train_loss, val_loss, train_model = trainnn_sgd(train_ds, val_ds, batch_size, train_model)
    # to continue training
    train_loss2, val_loss2, train_model = trainnn_sgd(train_ds, val_ds, batch_size, train_model, lr = 0.05)
    train_loss = train_loss + train_loss2
    val_loss = val_loss + val_loss2