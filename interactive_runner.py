from connect4 import ConnectFour
from models.Architectures import Architecture1
from models.OneMoveLookAhead import OneMoveLookAhead
from models.RandomPlayer import RandomPlayer
from models.MCTS import MCTSModel
from models.NNModel import NNModel
from models.Negamax import Negamax
import torch
"""
This file enables benchmarking of agent by manually playing against it
"""


def play_input(model, ai_goes_first = True):
    # get human to play model
    env = ConnectFour()
    done = False
    reward = 0
    if ai_goes_first:
        print("AI GOING FIRST")
        move = model.move(env)
        print(move)
        _, _, _ = env.play(move)
        
    while not done:
        env.displayBoard()
        move = -1
        while True:
            move = input("Make a move (cols 0-6): ")

            try:
                move = int(move)
            except:
                print("Invalid move. Try again")
                continue
            if env.canPlay(move):
                _, reward, done = env.play(move)
                break
            else:
                print("Invalid move. Try again")

        if not done:
            response = model.move(env.duplicate())
            print(response)
            _, reward, done = env.play(response)
        
        # not an else
        if done:
            if reward == 1:
                print("Player 1 Wins")
            elif reward == -1:
                print("Player 2 Wins")
            else:
                print("Draw")


#play_input(MCTSModel(), True)
#"""

def makeMCTSOpponent():
    NN = Architecture1(100)
    NN.load_state_dict(torch.load("param_files/arc1-100-v8.pth"))
    NN.double()
    return MCTSModel(300, OneMoveLookAhead(), NN, 0.25)

def makeNegamaxOpponent():
    nn = Architecture1(100)
    nn.load_state_dict(torch.load("param_files/arc1-100-v8.pth"))
    nn.double()
    return Negamax(nn)

def makeNNOpponent():
    nn_model = NNModel()
    nn_model.set_position_scorer(Architecture1(100),
    "param_files/arc1-100-v8.pth")
    return nn_model
play_input(makeMCTSOpponent(), False)
#"""