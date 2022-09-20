import os, sys

# add parent directory
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from connect4 import ConnectFour

from models.OneMoveLookAhead import OneMoveLookAhead
from models.RandomPlayer import RandomPlayer
from models.MCTS import MCTSModel

def testModel(model, sequence, expectedMoves):
    env = ConnectFour()
    for i in range(len(sequence)):
        env.play(sequence[i])
    res = model.move(env)
    if res not in expectedMoves:
        print(f"Wrong Answer: Expected something in {expectedMoves}, got {res}")
    else:
        print(f'Accepted, received {res}')

testModel(OneMoveLookAhead(), [0,1,0,1,0,1], [0])
testModel(OneMoveLookAhead(), [0,1,0,1,0,1,2],[1])
testModel(MCTSModel(), [0,1,0,1,0,1], [0])
testModel(MCTSModel(), [0,1,0,1,0,1,2],[1])
testModel(MCTSModel(), [0,1,1,2,2,3,2,3,3,0], [3])
testModel(MCTSModel(), [3,3,2,3,4,3], [1,5])