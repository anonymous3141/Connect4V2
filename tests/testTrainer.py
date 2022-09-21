import os, sys
# add parent directory
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from models.Architectures import Architecture1


from connect4 import ConnectFour
from trainer import Trainer

# test eps
trainer = Trainer("test.txt","test2.txt",Architecture1())
print(trainer.epsilon_scheduler(0),\
    trainer.epsilon_scheduler(1500),\
    trainer.epsilon_scheduler(3000),\
    trainer.epsilon_scheduler(5000))