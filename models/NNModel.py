import torch
import torch.nn

class NNModel(IModel):
    def __init__(self):
        self.model = None
        self.gameStates = []

    def set_model(model):
        self.model = model

    def move(self, gameState):
        pass
    
    # apply backprop to list of cached data
    def gameOver(self, reward):
        pass

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
