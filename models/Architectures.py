import torch.nn as nn 
import torch
"""
List of NN architectures.

Each architecture approximates

f: Position -> score
score = 1 in favour of p1
score = 0 means drawish
score = -1 in favour of p2

Position T is encoded as a 6 by 7 grid with 2 channels
T[0][i][j]: 1 if occupied by p1, 0 else
T[1][i][j]: 1 if occupied by p2, 0 else

Intuition: Need 4 by 4 conv filters as its connect4
Architecture 1: 1 conv layer
64 4 by 4 conv filters + 2 fully connected layers of 64
+ connect to output

FC(x,y) ~ xy params

Cost = 4*4*2*64 conv params + FC(4*3*64, 64) + FC(64, 64) + FC(64,1)
approx = 56000 params

Architecture 2: 2 conv layers
- The intuition is that you need as much filters
as you did before so that information isnt lost
- Another conv layer shaves off params
64 4 by 4 conv filters + 64 2 by 2 conv filters
+ 1 fully connected layer of 64 + output layer

Cost = 4*4*2*64 + 2*2*64*64 + 3*2*64*64 + 64
approx = 43000 params

And observe the values roughly scale quadratically 
with varying 64

Intended Training Idea:
- Min Squared error metric
"""

NUM_NODES = 32

class Architecture1(nn.Module):

  def __init__(self):
    super(Architecture1, self).__init__() 
    self.feature_stack = nn.Sequential(nn.Conv2d(2,NUM_NODES,4), #in channels, out channels, filter size
                                      nn.ReLU(inplace=True),
                                      nn.Flatten(start_dim=1, end_dim=-1))
    self.linear_stack = nn.Sequential(nn.Linear(NUM_NODES * 3 * 4,NUM_NODES),
                                      nn.ReLU(),
                                      nn.Linear(NUM_NODES,NUM_NODES),
                                      nn.ReLU(),
                                      nn.Linear(NUM_NODES, 1))
  def forward(self, x):
    return self.linear_stack(self.feature_stack(x))

class Architecture2(nn.Module):

  def __init__(self):
    super(Architecture2, self).__init__() 
    self.feature_stack = nn.Sequential(nn.Conv2d(2,NUM_NODES,4),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(NUM_NODES,NUM_NODES,2),
                                      nn.ReLU(inplace=True),
                                      nn.Flatten(start_dim=1, end_dim=-1))
    self.linear_stack = nn.Sequential(nn.Linear(NUM_NODES * 2 * 3,NUM_NODES),
                                      nn.ReLU(),
                                      nn.Linear(NUM_NODES, 1))
  def forward(self, x):
    return self.linear_stack(self.feature_stack(x))