# shallow nn with DC algorithm without introducing copies
from utils.util import *
import cvxpy as _cp


def trainnn_dcshallow(train_ds, val_ds, width, train_model, lmda = 10.0, iterations = 20, verbose = True, solver = "MOSEK"):
    """Train a shallow neural network with DC algorithm, with mean absolute error (MAE) as the loss function.

    Args:
        train_ds (list): training dataset in form of [(input_1, label_1), ..., (input_n, label_n)]
        val_ds (list): validation dataset in the same form as ``train_ds``
        width (int): number of nerons in the unique hidden layer
        train_model (torch.nn.modules.container.Sequential): a neural network model, providing initialization of DC algorithm
        lmda (float, optional): exact penalty parameter. Defaults to 10.0.
        iterations (int, optional): number of DC iterations to be performed. Defaults to 20.
        verbose (bool, optional): whether to print training and validation DC losses. Defaults to True.
        solver (str, optional): solver used for solving SOCP. Can be one of 'ECOS', 'MOSEK' and 'SCS'. Defaults to 'MOSEK'.

    Returns:
        tuple:
        a tuple consisting of the following elements in sequence:

        * U0 (*np.ndarray*)
            Weight matrix of the first hidden layer of shape (d_0, d_1).

        * alpha0 (*np.ndarray*)
            Weight matrix of the output layer of shape (d_1, ).

        * wine_train_dcloss (*list*)
            Training loss (MAE) per DC iteration.

        * wine_val_dcloss (*list*)
            Validation loss (MAE) per DC iteration.

        * exactpenalty (*float*)
            Exact penalty of introducing artifitial variable (linear output of the hidden layer).

    """    
    ## Extract training samples and validation samples
    X, y = extract(train_ds)
    y = y.flatten()
    X_val, y_val = extract(val_ds)

    n = X.shape[0]  # training size
    d = X.shape[1]  # number of features
    d1 = width      # number of neurons


    # Define variables to solve each SOCP subproblem
    U = _cp.Variable((d, d1))
    Z = _cp.Variable((n, d1))
    alpha = _cp.Variable(d1)
    T1 = _cp.Variable((n, d1)) # t for component (1)
    T3 = _cp.Variable(n) # t for component (3)
    T31 = _cp.Variable((n, d1)) # \tau_1 for (3)
    T32 = _cp.Variable((n, d1)) # \tau_2 for (3)
    T33 = _cp.Variable((n, d1)) # \tau_3 for (3)

    U0 = train_model.state_dict()['0.weight'].numpy().transpose()
    alpha0 = train_model.state_dict()['2.weight'].numpy().flatten()
    Z0 = X @ U0

    count = 0
    wine_train_dcloss = [eloss(relunn(X, [U0], alpha0), y, 1)]
    wine_val_dcloss = [eloss(relunn(X_val, [U0], alpha0), y_val, 1)]
    if verbose == True:   
        print("DC iteration:", count, ";", "training error:", "%0.8f" % wine_train_dcloss[-1],
            ";", "prediction error:", "%0.8f" % wine_val_dcloss[-1])  

    # dc training for shallow ReLu neural network
    while count <= iterations - 1:
        count = count + 1

        # Define objective function
        objective = _cp.Minimize(
            lmda * _cp.sum(T1) + _cp.sum(T3) - \
            _cp.sum([Z[i, :] @ (fa(Z0[i, :], alpha0) + ga(Z0[i, :], alpha0)) + alpha @ (fb(Z0[i, :], alpha0) + gb(Z0[i, :], alpha0)) for i in range(n)])
        )

        # Define constraints
        constraints = [T1 >= X @ U - Z, T1 >= Z- X @ U] + \
                    [T31[i, :] >= 0 for i in range(n)] + [T31[i, :] >= Z[i, :] for i in range(n)] + \
                    [T32[i, :] >= 0 for i in range(n)] + [T32[i, :] >= alpha for i in range(n)] + \
                    [T33[i, :] >= 0 for i in range(n)] + [T33[i, :] >= -1 * alpha for i in range(n)] + \
                    [_cp.SOC(T3[i] + 1 / 4, A31(d1) @ _cp.hstack([T3[i], T31[i,:], T32[i,:], T33[i,:]]) - b31(d1)) for i in range(n)] + \
                    [_cp.SOC(T3[i] - 2 * y[i] + 1 / 4, A32(d1) @ _cp.hstack([T3[i], T31[i,:], T32[i,:], T33[i,:]]) - b32(d1, y[i])) for i in range(n)]

        # Define problem for each iteration
        prob = _cp.Problem(objective, constraints)
        prob.solve(solver = solver)

        U0 = U.value
        Z0 = Z.value
        alpha0 = alpha.value

        # Record the training loss and validation loss
        wine_train_dcloss.append(eloss(relunn(X, [U0], alpha0), y, 1))
        wine_val_dcloss.append(eloss(relunn(X_val, [U0], alpha0), y_val, 1))

        if verbose == True:   
            print("DC iteration:", count, ";", "training error:", "%0.8f" % wine_train_dcloss[-1],
                ";", "prediction error:", "%0.8f" % wine_val_dcloss[-1])  

    # check exact penalty
    exactpenalty = eloss(X @ U0, Z0, 1, mean = False)
    return U0, alpha0, wine_train_dcloss, wine_val_dcloss, exactpenalty



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

    # train shallow NN with DC algorithm
    trainnn_dcshallow(train_ds, val_ds, width, truemodel)