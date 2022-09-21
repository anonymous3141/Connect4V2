from trainer import Trainer 
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Architectures import *

# compare 4 architectures to figure best one out
# use default params (0.01 min eps, 10k games)
NUM_GAMES = 10000
"""
experiment1 = Trainer("results_log/arch1-32v1.txt",\
     "param_files/arc1-32v1.pth", Architecture1(32),NUM_GAMES,False)

experiment2 = Trainer("results_log/arch1-64v1.txt",\
     "param_files/arc1-64v1.pth", Architecture1(64),NUM_GAMES)
experiment3 = Trainer("results_log/arch2-32v1.txt",\
     "param_files/arc2-32v1.pth", Architecture2(32),NUM_GAMES)
experiment4 = Trainer("results_log/arch2-64v1.txt",\
     "param_files/arc2-64v1.pth", Architecture2(64),NUM_GAMES)
"""
experiment5 = Trainer("results_log/arch1-100-exponential-bootstrap.txt",\
     "param_files/arc1-100-exponential-bootstrap.pth", Architecture1(100), NUM_GAMES, True)
experiment5.trainLoop()
#experiment6 = Trainer("results_log/arch1-128-exponential.txt",\
#     "param_files/arc1-128-exponential.pth", Architecture1(128), NUM_GAMES)
#experiment6.trainLoop()
"""
experiment2.trainLoop()
experiment3.trainLoop()
experiment4.trainLoop()
"""

"""

# debug dead neurons

for name, param in experiment2.modelToTrain.position_scorer.named_parameters():
    if param.requires_grad:
        print(name, param.data)

X = ConnectFour()
X.play(1)
X.play(2)
X.play(4)
print(experiment2.modelToTrain.position_scorer(torch.tensor(X.toNPArray()).double().unsqueeze(0)))
X.play(5)
#print(torch.tensor(X.toNPArray()).double().unsqueeze(0))
print(experiment2.modelToTrain.position_scorer(torch.tensor(X.toNPArray()).double().unsqueeze(0)))
"""
