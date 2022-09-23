from trainer import Trainer 
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Architectures import *
import numpy as np
opponent1 = OneMoveLookAhead()

opponents = [opponent1]
np.random.seed(42)
experiment6 = Trainer("results_log/arch1-100-v8.txt",\
     "param_files/arc1-100-v8.pth", Architecture1(100), opponents, "",
     False, 10, 10000, 0.05, 1, 0.0001)
experiment6.trainLoop()

