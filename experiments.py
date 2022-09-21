from trainer import Trainer 
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Architectures import *

# compare 4 architectures to figure best one out
# use default params (0.01 min eps, 10k games)
NUM_GAMES = 1000
experiment1 = Trainer("results_log/arch1-32v1.txt",\
     "param_files/arc1-32v1.txt", Architecture1(32),NUM_GAMES)
experiment2 = Trainer("results_log/arch1-64v1.txt",\
     "param_files/arc1-64v1.txt", Architecture1(64),NUM_GAMES)
experiment3 = Trainer("results_log/arch2-32v1.txt",\
     "param_files/arc2-32v1.txt", Architecture2(32),NUM_GAMES)
experiment4 = Trainer("results_log/arch2-64v1.txt",\
     "param_files/arc2-64v1.txt", Architecture2(64),NUM_GAMES)

experiment1.trainLoop()
experiment2.trainLoop()
experiment3.trainLoop()
experiment4.trainLoop()