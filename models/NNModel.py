import torch
import torch.nn as nn
from .IModel import IModel
import numpy as np

DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.0003
LOSS_FN = nn.MSELoss()

class NNModel(IModel):
    def __init__(self):
        self.position_scorer = None
        self.optimizer = None
        self.gameStates = []

    def set_position_scorer(self, model):
        self.position_scorer = model.double()
        self.optimizer = torch.optim.Adam(\
            self.position_scorer.parameters(),\
            lr = LEARNING_RATE)

    def move(self, gameState):
        # input: duplicated gameState
        with torch.inference_mode():
            best_move = -1
            best_move_score = -100000
            turn_sign = 1 if gameState.getTurn() == 1 else -1 # p1 maximise, p2 minimise
            for move in gameState.getValidMoves():
                newState = gameState.duplicate()
                newState.play(move)
                move_score = self.position_scorer(torch.tensor(newState.toNPArray()).unsqueeze(0).double())
                if turn_sign * move_score > turn_sign * best_move_score: # minimax
                    best_move = move 
                    best_move_score = move_score 
            
            # append both input state and state after chosen move
            self.gameStates.append(gameState)
            dup = gameState.duplicate()
            dup.play(best_move)
            self.gameStates.append(dup)

            # return best move
            return best_move 
            
    
    # apply backprop to list of cached data
    def gameOver(self, reward):
        # apply exponentially weighted final reward as a ground truth
        # single batch
        training_input = np.zeros((len(self.gameStates), 2, 6, 7))
        ground_truth = [reward]

        for i in range(len(self.gameStates)):
            training_input[i,:] = self.gameStates[i].toNPArray() # np syntax nice
        
        for i in range(len(self.gameStates)-1):
            ground_truth.append(ground_truth[-1]*DISCOUNT_FACTOR)
        
        outputs = self.position_scorer(torch.tensor(training_input).double()).squeeze(-1)
        loss = LOSS_FN(outputs, torch.tensor(ground_truth[::-1]).double()) # compute loss
        #print(outputs, torch.tensor(ground_truth[::-1]).double())
        loss.backward() # generate gradients
        self.optimizer.step() # make a optimization step

    # save neural net model
    def save(self, filename):
        torch.save(self.position_scorer.state_dict(), filename)
