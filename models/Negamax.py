from .IModel import IModel
import torch
class Negamax(IModel):

    def __init__(self, value_fn, depth = 3):
        self.value_fn = value_fn
        self.depth = depth
    
    def negamax(self, env, depth):
        # input: enviroment, depth (initial call must be > 0)
        # return: best objective value for turn player, move to achieve score
        sign = 1 if env.getTurn() == 1 else -1
        if env.isTerminal():
            return env.getResult() * sign, -1
        elif depth == 0:
            return self.value_fn(torch.tensor(env.toNPArray()).double().unsqueeze(0)).item()*sign , -1
        else:
            best_value = -2
            best_move = -1
            for move in env.getValidMoves():
                succ = env.duplicate()
                succ.play(move)
                res, _ = self.negamax(succ, depth-1) 
                if -res > best_value:
                    best_value = -res
                    best_move = move
            return best_value, best_move

    def move(self, gameState, eps = 0):
        _, move = self.negamax(gameState, self.depth)
        #print(move)
        return move 
