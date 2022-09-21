import torch
import torch.nn as nn
from .IModel import IModel
import numpy as np

DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.0001
LOSS_FN = nn.MSELoss()

class NNModel(IModel):
    def __init__(self):
        self.position_scorer = None
        self.optimizer = None
        self.gameStates = []
        self.gameStateValues = []
        self.gameNumber = 0
    def set_position_scorer(self, model, filename = ""):
        self.position_scorer = model.double()

        if filename != "":
            self.position_scorer.load_state_dict(torch.load(filename))
        self.optimizer = torch.optim.Adam(\
            self.position_scorer.parameters(),\
            lr = LEARNING_RATE, weight_decay=0.001)

    # exploration-exploitation parameter

    def move(self, gameState, eps = 0):
        # input: duplicated gameState, eps for eps greedy
        best_move = -1
        if np.random.random() < eps:
            best_move = np.random.choice(gameState.getValidMoves()).item()
        else:
            with torch.inference_mode():
                turn_sign = 1 if gameState.getTurn() == 1 else -1 # p1 maximise, p2 minimise
                best_move_score = -turn_sign*2 # minimax
                
                for move in gameState.getValidMoves():
                    newState = gameState.duplicate()
                    newState.play(move)
                    move_score = self.position_scorer(torch.tensor(newState.toNPArray()).unsqueeze(0).double()).item()
                    #print(move_score, move, best_move_score)
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

    # sometimes need to add a terminal state if opponent makes last move    
    def addTerminalState(self, gameState):
        self.gameStates.append(gameState)

    # apply backprop to list of cached data
    def gameOver(self, reward, inference_mode = False, bootstrapping=False):
        # apply exponentially weighted final reward as a ground truth
        # single batch

        if inference_mode:
            self.gameStates.clear()
            return 
        training_input = np.zeros((len(self.gameStates), 2, 6, 7))
        ground_truth = []

        for i in range(len(self.gameStates)):
            training_input[i,:] = self.gameStates[i].toNPArray() # np syntax nice
        
        outputs = self.position_scorer(torch.tensor(training_input).double()).squeeze(-1)
        
        if bootstrapping == False:
            # use monte carlo learning to generate targets
            ground_truth.append(reward)
            for i in range(len(self.gameStates)-1):
                ground_truth.append(ground_truth[-1]*DISCOUNT_FACTOR)
            ground_truth = ground_truth[::-1] #reverse
        else:
            # use the on policy variant of the TD(n) algorithm, i.e SARSA
            # change target to be 'optimal successor' according to cur policy
            ground_truth = [0] * len(self.gameStates)
            ground_truth[-1] = reward
            N = 15
            for i in range(len(self.gameStates)-2, -1, -1):
                if i+N>=len(self.gameStates):
                    ground_truth[i] = reward*(DISCOUNT_FACTOR**(len(self.gameStates)-i))
                else:
                    ground_truth[i] = outputs[i+N]*(DISCOUNT_FACTOR**(N-i))
        
        loss = LOSS_FN(outputs, torch.tensor(ground_truth).double()) # compute loss
        #print(outputs, torch.tensor(ground_truth[::-1]).double())
        loss.backward() # generate gradients
        self.gameStates.clear()
        self.optimizer.step() # make a optimization step

    # save neural net model
    def save(self, filename):
        torch.save(self.position_scorer.state_dict(), filename)
    
