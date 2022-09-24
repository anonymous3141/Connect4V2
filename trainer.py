from connect4 import ConnectFour
from models.OneMoveLookAhead import OneMoveLookAhead
from models.NNModel import NNModel

from copy import deepcopy
import numpy as np


"""
Train models by partial self play in style similar
to how alpha go is trained

CANDIDATE_OPPONENTS = [SOME BASELINES]
REPEAT N times:
    INITIALISE CURRENT AGENT FROM PREVIOUS AGENT PARAMS
    FOR M LEARNING ITERATIONS
        DECAY EPSILON
        USE NAIVE MONTE-CARLO LEARNING:
            PLAY CURRENT AGENT AGAINST RANDOM PREVIOUS CANDIDATE
    DECAY LEARNING RATE
    ADD CURRENT AGENT TO OPPONENT LIST

baseline params:
N = 5
M = 10K
LR0 = 0.0001, SGD optimiser
EPS0 = 0.1, decaying down to 0.01 throughout course of each game
DECAY LR = halve after each iteration
DECAY EPS = exponential schedule

"""

class Trainer:
    def __init__(self, performance_file, params_file, nn_object,
        baselines = [OneMoveLookAhead()], pretrained_model = "", 
        bootstrap = False, num_its=1, num_games=10000,
         min_eps=0.01, max_eps=1, init_learn_rate = 0.001):

        self.NUM_TRAIN_GAMES = num_games
        self.opponents = baselines
        self.modelToTrain = NNModel(learning_rate=init_learn_rate)
        self.modelToTrain.set_position_scorer(nn_object, pretrained_model)
        self.MODEL_PERFORMANCE_FILE = performance_file # file to log history
        self.MODEL_PARAMS_FILE = params_file # file to save params
        self.init_learn_rate = init_learn_rate
        self.num_its = num_its

        # epsilon decay, assume > 0
        self.MAX_EPS = max_eps
        self.MIN_EPS = min_eps
        self.THRESHOLD = 0.8
        self.decay = (self.MIN_EPS/self.MAX_EPS)**(1/(self.THRESHOLD * num_games * num_its))


        self.bootstrap = bootstrap
    
    def runOneGame(self, opponent, eps, inference_mode = False, nn_goes_first=None):
        # run the training loop for one game
        if nn_goes_first == None:
            nn_goes_first = np.random.random() < 0.5
        env = ConnectFour()
        
        if nn_goes_first: 
            env.play(self.modelToTrain.move(env.duplicate(), eps))
        elif not inference_mode:
            # slightly hacky, append empty board
            self.modelToTrain.gameStates.append(env.duplicate())

        while not env.isTerminal():
            env.play(opponent.move(env.duplicate()))

            if not env.isTerminal():
                env.play(self.modelToTrain.move(env.duplicate(), eps))
            elif not inference_mode:
                self.modelToTrain.addTerminalState(env.duplicate())
        
        self.modelToTrain.gameOver(env.getResult(), inference_mode, self.bootstrap)
        opponent.gameOver(env.getResult(), True) 
        
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
        BENCHMARK_SIZE = 100
        DEBUG = True
        LR_DECAY = 0.5
        LR_CUR = self.init_learn_rate
        eps = self.MAX_EPS
        for it in range(self.num_its):
            for i in range(self.NUM_TRAIN_GAMES):
                if i%BENCHMARK_SIZE == 0:
                    print(f"It{i} of {self.NUM_TRAIN_GAMES}")
                eps = max(self.MIN_EPS, eps * self.decay)
                self.runOneGame(np.random.choice(self.opponents), eps)

            res = 0
            for i in range(BENCHMARK_SIZE):
                res += self.runOneGame(self.opponents[-1], 0.02, True)
            print(f"Final result: Won {res} out of {BENCHMARK_SIZE} points against last model")
            
            if res/BENCHMARK_SIZE > 0.4: #increase this to 70%?
                self.opponents.append(deepcopy(self.modelToTrain))
            LR_CUR *= LR_DECAY 
            new_model = NNModel(LR_CUR)
            new_model.set_position_scorer(deepcopy(self.opponents[-1].position_scorer))
            self.modelToTrain = new_model 
        

        self.modelToTrain.save(self.MODEL_PARAMS_FILE)
        open(self.MODEL_PERFORMANCE_FILE, "w").write(\
            " ".join([str(c) for c in results]))
        
       


