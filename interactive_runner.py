from connect4 import ConnectFour
from models.Architectures import Architecture1
from models.OneMoveLookAhead import OneMoveLookAhead
from models.RandomPlayer import RandomPlayer
from models.MCTS import MCTSModel
from models.NNModel import NNModel
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
nn = Architecture1(100)


model = NNModel()
model.set_position_scorer(nn, "param_files/arc1-100-v3.pth")
play_input(model, True)
#"""