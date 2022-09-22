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
        num_games=10000, bootstrap = False, min_eps=0.01, max_eps=1, baseline = OneMoveLookAhead(), pretrained_model = ""):
        self.NUM_TRAIN_GAMES = num_games
        self.baseline = baseline
        self.modelToTrain = NNModel()
        self.modelToTrain.set_position_scorer(nn_object, pretrained_model)
        self.MODEL_PERFORMANCE_FILE = performance_file # file to log history
        self.MODEL_PARAMS_FILE = params_file # file to save params
        
        # epsilon decay, assume > 0
        self.MAX_EPS = max_eps
        self.MIN_EPS = min_eps
        self.THRESHOLD = 0.8
        self.decay = (self.MIN_EPS/self.MAX_EPS)**(1/(self.THRESHOLD * num_games))


        self.bootstrap = bootstrap
    
    def runOneGame(self, eps, inference_mode = False):
        # run the training loop for one game
        nn_goes_first = np.random.random() < 0.5
        env = ConnectFour()

        if inference_mode:
            eps = 0
        
        if nn_goes_first: 
            env.play(self.modelToTrain.move(env.duplicate(), eps))
        
        while not env.isTerminal():
            env.play(self.baseline.move(env.duplicate()))

            if not env.isTerminal():
                env.play(self.modelToTrain.move(env.duplicate(), eps))
            else:
                self.modelToTrain.addTerminalState(env.duplicate())
        
        self.modelToTrain.gameOver(env.getResult(), inference_mode, self.bootstrap)
        self.baseline.gameOver(env.getResult()) # todo: this can be selfplay
        
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
        eps = self.MAX_EPS
        for i in range(self.NUM_TRAIN_GAMES):
            if DEBUG and i%BENCHMARK == 0 and i >= BENCHMARK:
                print(f"It {i}: Model won {sum(results[-BENCHMARK:])} of last {BENCHMARK} points")
            eps = max(self.MIN_EPS, eps * self.decay)
            results.append(self.runOneGame(eps))

        self.modelToTrain.save(self.MODEL_PARAMS_FILE)
        open(self.MODEL_PERFORMANCE_FILE, "w").write(\
            " ".join([str(c) for c in results]))
        
        res = 0
        for i in range(BENCHMARK):
            res += self.runOneGame(0, True)
        print(f"Final result: Won {res} out of {BENCHMARK} points against baseline")


