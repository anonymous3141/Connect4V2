from trainer import Trainer
from connect4 import ConnectFour
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.OneMoveLookAhead import OneMoveLookAhead
from models.Negamax import Negamax
from models.Architectures import *
import numpy as np

class Benchmarker:
    
    def __init__(self, models, games=100):
        self.trainers = {}
        self.results = {}
        self.model_names = list(models.keys())
        self.num_games = games
        for model_name in self.model_names:
            # very hacky code reuse
            model_trainer = Trainer("","",Architecture1()) 
            model_trainer.modelToTrain = models[model_name]
            self.trainers[model_name] = model_trainer

    def playOff(self, trainer1, trainer2):
        trainer1_points = 0 
        for i in range(self.num_games):
            print(i)
            trainer1_points += trainer1.runOneGame(trainer2.modelToTrain, 0, True)
        return trainer1_points
    
    def selectionLoop(self):
        for m1 in range(len(self.model_names)):
            for m2 in range(m1+1, len(self.model_names)):
                name1 = self.model_names[m1]
                name2 = self.model_names[m2]
                res = self.playOff(self.trainers[name1], self.trainers[name2])
                print(f"Playing {name1} against {name2}")
                print(f"{name1} won {res} of {self.num_games} points against {name2}")
    
NN = Architecture1(100)
NN.load_state_dict(torch.load("param_files/arc1-100-v8.pth"))
NN.double()
opponent0 = MCTSModel(300)
opponent1 = MCTSModel(300, OneMoveLookAhead(), NN, 0.25)
opponent2 = MCTSModel(300, OneMoveLookAhead(), NN, 0.5)
opponent3 = MCTSModel(300, OneMoveLookAhead(), NN, 0.75)
opponent4 = Negamax(NN)
opponent_dict = {"MCTS300-Vanilla": opponent0,
                "MCTS300-0.25": opponent1,
                "MCTS300-0.5": opponent2,
                "MCTS300-0.75": opponent3,
                "Negamax-depth3": opponent4
                }

benchmarker = Benchmarker(opponent_dict, 100)
benchmarker.selectionLoop()