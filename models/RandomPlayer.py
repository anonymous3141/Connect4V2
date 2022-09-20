from .IModel import IModel
import numpy as np

class RandomPlayer(IModel):
    def __init__(self):
        np.random.seed(42)
    def move(self, gameState):
        assert(not gameState.isTerminal())
        return np.random.choice(gameState.getValidMoves()).item()

