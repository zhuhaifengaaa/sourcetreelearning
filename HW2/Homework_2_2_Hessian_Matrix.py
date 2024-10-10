import numpy as np
from math import pi
from collections import defaultdict
from autograd_lib import autograd_lib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore")

student_id = 'zhu123z'  # fill with your student ID
# assert student_id != 'zhu123', 'Please fill in your student_id before you start.'

# find the key from student_id
import re

key = student_id[-1]
if re.match('[0-9]', key) is not None:
    key = int(key)
else:
    key = ord(key) % 10


class MathRegressor(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


# load checkpoint and data corresponding to the key
model = MathRegressor()
autograd_lib.register(model)

data = torch.load('data.pth')[key]
model.load_state_dict(data['model'])
# train, target = data['data']  # 或者如下
train = torch.Tensor([[0.2400],  # 然后feature和target(label)也加载进来
                      [0.2500],
                      [0.2600],
                      [0.2700],
                      [0.2800],
                      [0.2900]])
target = torch.Tensor([[-0.1559],
                       [-0.1801],
                       [-0.1981],
                       [-0.2101],
                       [-0.2162],
                       [-0.2168]])


# function to compute gradient norm
def compute_gradient_norm(model, criterion, train, target):
    model.train()
    model.zero_grad()
    output = model(train)
    loss = criterion(output, target)
    loss.backward()

    grads = []
    for p in model.regressor.children():
        if isinstance(p, nn.Linear):
            param_norm = p.weight.grad.norm(2).item()
            grads.append(param_norm)
    grad_mean = np.mean(grads)  # compute mean of gradient norms
    return grad_mean


# source code from the official document https://github.com/cybertronai/autograd-lib

# helper function to save activations
def save_activations(layer, A, _):
    '''
    A is the input of the layer, we use batch size of 6 here
    layer 1: A has size of (6, 1)
    layer 2: A has size of (6, 128)
    '''
    activations[layer] = A


# helper function to compute Hessian matrix
def compute_hess(layer, _, B):
    '''
    B is the backprop value of the layer
    layer 1: B has size of (6, 128)
    layer 2: B ahs size of (6, 1)
    '''
    A = activations[layer]
    BA = torch.einsum('nl,ni->nli', B, A)  # do batch-wise outer product

    # full Hessian
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)  # do batch-wise outer product, then sum over the batch


# function to compute the minimum ratio
def compute_minimum_ratio(model, criterion, train, target):
    model.zero_grad()
    # compute Hessian matrix
    # save the gradient of each layer
    with autograd_lib.module_hook(save_activations):
        output = model(train)
        loss = criterion(output, target)

    # compute Hessian according to the gradient value stored in the previous step
    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(output, loss='LeastSquares')

    layer_hess = list(hess.values())
    minimum_ratio = []

    # compute eigenvalues of the Hessian matrix
    for h in layer_hess:
        size = h.shape[0] * h.shape[1]
        h = h.reshape(size, size)
        h_eig = torch.symeig(
            h).eigenvalues  # torch.symeig() returns eigenvalues and eigenvectors of a real symmetric matrix
        num_greater = torch.sum(h_eig > 0).item()
        minimum_ratio.append(num_greater / len(h_eig))

    ratio_mean = np.mean(minimum_ratio)  # compute mean of minimum ratio

    return ratio_mean


# the main function to compute gradient norm and minimum ratio
def main(model, train, target):
    criterion = nn.MSELoss()

    gradient_norm = compute_gradient_norm(model, criterion, train, target)
    minimum_ratio = compute_minimum_ratio(model, criterion, train, target)

    print('gradient norm: {}, minimum ratio: {}'.format(gradient_norm, minimum_ratio))


if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(0)

    # reset compute dictionaries
    activations = defaultdict(int)
    hess = defaultdict(float)

    # compute Hessian
    main(model, train, target)
