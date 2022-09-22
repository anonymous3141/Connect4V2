# Quick analysis of why NN didnt learn
from models.Architectures import *
from models.NNModel import NNModel
from connect4 import ConnectFour
import torch

def testGradientsDied(A):
    X = ConnectFour()
    X.play(0)
    X.play(1)
    X.play(0)
    X.play(1)
    X.play(0)
    X.play(1)
    r1 = A(torch.tensor(X.toNPArray()).double().unsqueeze(0)).item()
    X.play(5)
    r2 = A(torch.tensor(X.toNPArray()).double().unsqueeze(0)).item()
    X.play(1)
    r3 = A(torch.tensor(X.toNPArray()).double().unsqueeze(0)).item()
    print(r1,r2,r3, r1==r2) # true means gradients died

if __name__ == "__main__":
    A = Architecture1(100).double()
    A.load_state_dict(torch.load("param_files/arc1-100-v5.pth"))
    for name, param in A.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    testGradientsDied(A)