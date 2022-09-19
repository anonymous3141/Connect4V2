from .IModel import IModel
import numpy as np
from .header import *

class RandomPlayer(IModel):
    def __init__(self):
        np.random.seed(1)
    def move(self, gameState):
        assert(not gameState.isTerminal())
        return np.random.choice(gameState.getValidMoves()).item()

