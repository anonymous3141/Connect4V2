import numpy as np
from .header import *
from .IModel import IModel
from .RandomPlayer import RandomPlayer
from .OneMoveLookAhead import OneMoveLookAhead
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
        self.N = [0]*7 # num passes on node
        self.action_taken = action # action to get to node
        self.successors = [None]*7
        self.parent = parent
        self.isTerminal = self.state.isTerminal()
        self.validMoves = state.getValidMoves()
       
        
    def selectAction(self, EPS_MCTS = 0):
        # epsilon greedy
        # implicitly contains randomised rollout policy upon reaching 
        # leaf of the expanded tree
        if np.random.random() < EPS_MCTS:
            return np.random.choice(self.validMoves).item()
        
        bestMoves = [self.validMoves[0]]

        for move in self.validMoves:
            if self.N[move] * self.W[bestMoves[0]] > self.N[bestMoves[0]] * self.W[move]:
                bestMoves = [move]
            elif self.N[move] * self.W[bestMoves[0]] == self.N[bestMoves[0]] * self.W[move]:
                bestMoves.append(move)
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
    def __init__(self):
        np.random.seed(0)
    
    def rollout(self, gameState):
        # input: gameState is copy of cur game state
        # use randomised rollout to simulate game state
        # to end (returns cur result if already terminal state)
        # todo: add 1 move lookahead for insta-wins?
        player = OneMoveLookAhead()
        while not gameState.isTerminal():
            gameState.play(player.move(gameState))
        return gameState.getResult()
    
    def move(self, gameState):
        root = Node(gameState, None, None)
        for _ in range(NUM_MCTS_ITERATIONS):
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
        
        return root.selectAction(EPS_MCTS=0) # take best action
                    
                            
                        

