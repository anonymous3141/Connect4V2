from connect4 import ConnectFour
from models.OneMoveLookAhead import OneMoveLookAhead
from models.NNModel import NNModel
import numpy as np


"""
Train models against baseline 
- currently set up so baseline doesnt learn
"""

class Trainer:
    def __init__(self, performance_file, params_file, nn_object,\
        num_games=10000, min_eps=0.01, baseline = OneMoveLookAhead()):
        self.NUM_TRAIN_GAMES = num_games
        self.baseline = baseline
        self.modelToTrain = NNModel()
        self.modelToTrain.set_position_scorer(nn_object, "")
        self.MODEL_PERFORMANCE_FILE = performance_file # file to log history
        self.MODEL_PARAMS_FILE = params_file # file to save params
        self.MIN_EPS = min_eps
    def epsilon_scheduler(self, gameNumber):
        # epsilon scheduler: using linear decay to min threshold
        # after fixed proportion of train games, eps is decreased to min
        MAX_EPS = 1
        THRESHOLD = 0.6 
        return self.MIN_EPS +(MAX_EPS - self.MIN_EPS) *\
                    max(0, 1 - gameNumber / (self.NUM_TRAIN_GAMES * THRESHOLD))
    
    def runOneGame(self, gameNumber, inference=False):
        # run the training loop for one game
        nn_goes_first = np.random.random() < 0.5
        env = ConnectFour()
        eps = self.epsilon_scheduler(gameNumber)

        if inference:
            eps = 0
        
        if nn_goes_first: 
            env.play(self.modelToTrain.move(env.duplicate(), eps))
        
        while not env.isTerminal():
            env.play(self.baseline.move(env.duplicate()))

            if not env.isTerminal():
                env.play(self.modelToTrain.move(env.duplicate(), eps))
            else:
                self.modelToTrain.addTerminalState(env.duplicate())
        
        self.modelToTrain.gameOver(env.getResult())
        self.baseline.gameOver(env.getResult()) # todo: this can be another model
        
        # return 1 if model won, 1/2 if draw, 0 if baseline won
        sign = -(-1)**(nn_goes_first)
        result = env.getResult()
        if result * sign == 1:
            return 1
        elif result == 0:
            return 0.5
        else:
            return 0
    
    def trainLoop(self):
        results = []
        BENCHMARK = 100
        DEBUG = True
        for i in range(self.NUM_TRAIN_GAMES):
            if DEBUG and i%BENCHMARK == 0 and i >= BENCHMARK:
                print(f"It {i}: Model won {sum(results[-BENCHMARK:])}\
                     of last {BENCHMARK} points")
            results.append(self.runOneGame(i))

        self.modelToTrain.save(self.MODEL_PARAMS_FILE)
        open(self.MODEL_PERFORMANCE_FILE, "w").write(\
            " ".join([str(c) for c in results]))
        
        res = 0
        for i in range(BENCHMARK):
            res += self.runOneGame(i, inference=True)
        print(f"Final result: Won {res} out of {BENCHMARK} points against baseline")


