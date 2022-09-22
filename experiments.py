from trainer import Trainer 
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Architectures import *
opponent1 = OneMoveLookAhead()
opponent2 = MCTSModel(50)
opponent3 = NNModel()
opponent3.set_position_scorer(Architecture1(100), "param_files/arc1-100-exponential.pth")

opponents = [opponent1, opponent2, opponent3]

experiment6 = Trainer("results_log/arch1-100-v5.txt",\
     "param_files/arc1-100-v5.pth", Architecture1(100), opponents, "param_files/arc1-100-exponential.pth",
     False, 1, 1000, 0.01, 0.2, 0.0001)
experiment6.trainLoop()

