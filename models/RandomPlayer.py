from .IModel import IModel
import numpy as np

class RandomPlayer(IModel):
    def __init__(self):
        pass
    def move(self, gameState, eps = 0):
        assert(not gameState.isTerminal())
        return np.random.choice(gameState.getValidMoves()).item()

