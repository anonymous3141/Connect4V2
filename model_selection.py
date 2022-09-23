from trainer import Trainer 
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Architectures import *
import numpy as np

benchmarker = Trainer("",\
     "", Architecture1(100), [], "param_files/arc1-100-v8.pth",
     False, 10, 10000, 0.05, 1, 0.0001)

opponent1 = MCTSModel(50)
res = 0
GAMES = 100
for i in range(GAMES):
    res += benchmarker.runOneGame(opponent1, 0.05, True)
print(f"Won {res} points out of {GAMES} against MCTS50")

opponent2 = NNModel()
opponent2.set_position_scorer(Architecture1(100),"param_files/arc1-100-exponential.pth")

res = 0
GAMES = 100
for i in range(GAMES):
    res += benchmarker.runOneGame(opponent2, 0.05, True)
print(f"Won {res} points out of {GAMES} against NN v0")