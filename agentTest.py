from connect4 import ConnectFour
from models.RandomPlayer import RandomPlayer
"""
This file enables benchmarking of agent by manually playing against it
"""


def play_input(model, ai_goes_first = True):
    # get human to play model
    env = ConnectFour()
    done = False

    if ai_goes_first:
        # cant be done on first move
        move = model.move(env, env.getTurn())
        state, _, _ = env.play(move)

    while not done:
        env.displayBoard()
        move = -1
        while True:
            #print("AI Suggested move:")
            #print(get_move(model_type, model, state))
            move = input("Make a move (cols 0-6): ")

            try:
                move = int(move)
            except:
                print("Invalid move. Try again")
                continue
            if env.canPlay(move):
                tmp_state, reward, done = env.play(move)
                break
            else:
                print("Invalid move. Try again")

        if not done:
            response = model.move(env)
            print(response)
            env.play(response)
        
        if reward == 1:
            print("Player 1 Wins")
        elif reward == -1:
            print("Player 2 Wins")

#play_input("policy", "connect4PolicyVer4.pth")
#play_input("Q", "connect4QVer7.pth")
play_input(RandomPlayer(), None)
