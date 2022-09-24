from flask import g
import numpy as np
from .IModel import IModel
from .RandomPlayer import RandomPlayer
from .OneMoveLookAhead import OneMoveLookAhead
import torch
# Baseline solution: Used to pretrain neural nets
# I lost to this (250 iteration version) while actually trying lmao

class Node:
    """
    Node in Search Tree

    selectAction(): chooses action 
    updAction(action, result): upd score of action by result
    isChildCreated(action): is child node created
    createChild(action): makes the child if not already
    """
    def __init__(self, state, parent, action):
        self.state = state.duplicate()
        self.W = [0]*7 # score per action
        self.N = [1]*7 # num passes on node,  kind of laplace smoothed I guess
        self.action_taken = action # action to get to node
        self.successors = [None]*7
        self.parent = parent
        self.isTerminal = self.state.isTerminal()
        self.validMoves = state.getValidMoves()
    def selectAction(self, EPS_MCTS = 0.2):
        # epsilon greedy
        # implicitly contains randomised rollout policy upon reaching 
        # leaf of the expanded tree
        if np.random.random()  < EPS_MCTS:
            return np.random.choice(self.validMoves).item()
        
        bestMoves = [self.validMoves[0]]

        # maximise if turn = 1, minimise if turn = 2
        sign = 1 if self.state.getTurn() == 1 else -1
        for move in self.validMoves[1:]:
            if sign * self.N[move] * self.W[bestMoves[0]] < sign * self.N[bestMoves[0]] * self.W[move]:
                bestMoves = [move]
            elif self.N[move] * self.W[bestMoves[0]] == self.N[bestMoves[0]] * self.W[move]:
                bestMoves.append(move)
        #print(bestMoves, self.N, self.W)
        return np.random.choice(bestMoves).item()

    def updAction(self, action, result):
        self.N[action] += 1
        self.W[action] += result

    def isChildCreated(self, a):
        return self.successors[a] != None

    def createChild(self, a):
        assert(self.state.canPlay(a) and (not self.isTerminal))
        if not self.isChildCreated(a):
            newState = self.state.duplicate()
            newState.play(a)
            self.successors[a] = Node(newState, self, a)
    

class MCTSModel(IModel):
    """
    rollout() -> execute rollout policy
    """
    def __init__(self, num_its = 250, rollout_policy=OneMoveLookAhead(), 
    auxillary_scorer = None, scorer_weight = 0.5):
        self.NUM_ITERATIONS = num_its
        self.rolloutPolicy = rollout_policy
        self.aux_scorer = auxillary_scorer
        self.scorer_weight = scorer_weight
    def rollout(self, gameState):
        # input: gameState is copy of cur game state
        # use randomised rollout to simulate game state
        # to end (returns cur result if already terminal state)
        # todo: add 1 move lookahead for insta-wins?
        dup = gameState.duplicate()
        player = self.rolloutPolicy
        while not gameState.isTerminal():
            gameState.play(player.move(gameState.duplicate()))

        final_score = gameState.getResult()
        if self.aux_scorer != None:
            aux_move_score = self.aux_scorer(\
                        torch.tensor(dup.toNPArray()).double().unsqueeze(0))
            final_score = self.scorer_weight * aux_move_score +\
                        gameState.getResult() * (1 - self.scorer_weight)
        
        return final_score
    
    def move(self, gameState, eps = 0):
        # eps required by interface
        root = Node(gameState, None, None)
        for _ in range(self.NUM_ITERATIONS):
            # Execute one MCTS pass per loop iteration

            curNode = root # pointer
            last_action = -1 # action taken to get to curNode
            
            while not curNode.isTerminal:
                last_action = curNode.selectAction()
                if curNode.isChildCreated(last_action):
                     curNode = curNode.successors[last_action]
                else:
                     curNode.createChild(last_action)
                     break
            
            # make curNode is node on which we make lastAction
            if curNode.isTerminal:
                curNode = curNode.parent
            result = self.rollout(curNode.successors[last_action].state.duplicate())
            
            while curNode != None:
                curNode.updAction(last_action, result)
                last_action = curNode.action_taken
                curNode = curNode.parent
        #print(root.N, root.W, root.validMoves)
        return root.selectAction(EPS_MCTS=0) # take best action


