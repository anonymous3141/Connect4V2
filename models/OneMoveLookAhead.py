from .IModel import IModel
import numpy as np

class OneMoveLookAhead(IModel):
    def __init__(self):
        pass
    
    def move(self, gameState):
        # plays winning move if able
        assert(not gameState.isTerminal())
        for i in range(7):
            if gameState.canPlay(i):
                dup = gameState.duplicate()
                _, res, _ = dup.play(i)
                if res != 0 and\
                     ((gameState.getTurn() == 1 and res == 1) or\
                         (gameState.getTurn() == 2 and res == -1)):
                    return i

        return np.random.choice(gameState.getValidMoves()).item()

