import os, sys

# add parent directory
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from connect4 import ConnectFour
"""
Unit Tests
"""

def testSequence(sequence,display=True,checkNPConversion=False):
    print('-----------')
    print('Running sequence', sequence)
    x = ConnectFour()
    for i in range(len(sequence)):
        x.play(sequence[i])
        if i != len(sequence)-1:
            assert(not(x.getResult() or x.isTerminal()))
    if display: 
        x.displayBoard()
    if checkNPConversion:
        print(x.toNPArray())
    print(x.isTerminal(), x.getResult())
    print(x.numMoves, x.turn, x.mask1, x.mask2)

# test basic functionality
testSequence([1,1,1,1,1,1])
try:
    testSequence([1,1,1,1,1,1,1]) # should error
except:
    print('error spotted')
testSequence([0,1,1,2,2,2,3,3,3,3])

# test check win
print("Testing Check Win")
testSequence([0,1,0,1,0,1,0]) # player 1 vertical win
testSequence([0,1,2,1,0,1,2,1]) # player 2 vertical win
testSequence([0,6,1,6,2,6,3]) # player 1 horizontal win
testSequence([0,1,1,2,2,3,2,3,3,0,3]) # player 1 diagonal win

# some longer tests
long_test = '44444135555354137672222221571771661333766'
seq = [int(c)-1 for c in long_test]
testSequence(seq)

diagonal_test = '12233434414'
seq2 = [int(c)-1 for c in diagonal_test]
testSequence(seq2)

diagonal_test_yellow = '4534332222'
seq3 = [int(c)-1 for c in diagonal_test_yellow]
testSequence(seq3)

# test duplicate
def testDup():
    env = ConnectFour()
    env.play(0)
    dup = env.duplicate()
    env.play(0)
    assert(env.getTurn() == 1 and dup.getTurn() == 2)
testDup()
# test canPlay()
def testCanPlaySequence(sequence, expect):
    print('-----------')
    print('Running sequence', sequence)
    x = ConnectFour()
    for i in range(len(sequence)):
        x.play(sequence[i])
        if i != len(sequence)-1:
            assert(not(x.getResult() or x.isTerminal()))
    for i in range(7):
        if x.canPlay(i) != expect[i]:
            print('Error', i)
testCanPlaySequence([0,0,0,0,0,0,6,6,6,6,6,6],[0,1,1,1,1,1,0])

# Test NP array conversion
testSequence([0,1,1,2,2,3,2,3,3,0,3], False, True) # player 1 diagonal win