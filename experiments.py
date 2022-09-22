from trainer import Trainer 
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Architectures import *
opponent1 = OneMoveLookAhead()

opponents = [opponent1]

experiment6 = Trainer("results_log/arch1-100-v7.txt",\
     "param_files/arc1-100-v7.pth", Architecture1(100), opponents, "",
     False, 2, 10000, 0.075, 1, 0.0001)
experiment6.trainLoop()

