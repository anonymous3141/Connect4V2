class IModel:
    def __init__(self):
        pass
    
    # gameState should be a copy of current gameState
    def move(self, gameState, eps = 0):
        pass
    
    # apply backprop etc
    def gameOver(self, reward, inference_mode = False, bootstrapping=False):
        pass
