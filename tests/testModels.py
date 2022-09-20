import os, sys

# add parent directory
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from connect4 import ConnectFour

from models.OneMoveLookAhead import OneMoveLookAhead
from models.RandomPlayer import RandomPlayer
from models.MCTS import MCTSModel
from models.NNModel import NNModel 
from models.Architectures import *
import numpy as np

def testModel(model, sequence, expectedMoves, nn=None):
    print("------------")
    env = ConnectFour()
    for i in range(len(sequence)):
        env.play(sequence[i])
    
    if nn != None:
        # duck typing OP
        model.set_position_scorer(nn)

    # test if next move is expected
    res = model.move(env)
    if res not in expectedMoves:
        print(f"Wrong Answer: Expected something in {expectedMoves}, got {res}")
    else:
        print(f'Accepted, received {res}')

    if nn != None:
        # test backward, prematurely end
        model.gameOver(env.getResult())
        print('Backprop Ran Successfully')
        
np.random.seed(42)
# test OneMoveLookAhead
testModel(OneMoveLookAhead(), [0,1,0,1,0,1], [0])
testModel(OneMoveLookAhead(), [0,1,0,1,0,1,2],[1])

# test MCTS
# just gotta accept some WA np rng is kinda shitty
testModel(MCTSModel(), [0,1,0,1,0,1], [0])
testModel(MCTSModel(), [0,1,0,1,0,1,2],[1])
testModel(MCTSModel(), [0,1,1,2,2,3,2,3,3,0], [3])
testModel(MCTSModel(), [3,3,2,3,4,3], [1,5])

# test NN
testModel(NNModel(), [3,3,2,3,4,3], [0,1,2,3,4,5,6], Architecture1()) # test nn not crash on train
testModel(NNModel(), [0,1,0,1,0,1], [0,1,2,3,4,5,6], Architecture1()) # test nn not crash on backward