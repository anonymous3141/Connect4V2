from .IModel import IModel
import numpy as np
from .header import *

class RandomPlayer(IModel):
    def __init__(self):
        pass

    def move(self, gameState):
        assert(not gameState.hasGameEnded())
        return np.random.choice(gameState.getValidMoves())

