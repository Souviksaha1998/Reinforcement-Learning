import torch.nn as nn
import torch
import numpy as np
import itertools
import random
from custom_env_class import SpaceGame
from collections import deque

# setting hyperparameter
GAMMA = 0.98
LR = 0.00001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.00001
TARGET_UPDATE_FREQUENCY = 2500
MIN_REPLAY_SIZE = 2000
BATCH = 64
BUFFER_SIZE = 100000

space_game = SpaceGame(120)
actions = len(space_game.actions_())
observation , _  , _= space_game.reset()


replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0],maxlen=100)
episode_reward = 0.0

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        in_feat = int(np.prod(np.array(observation).reshape(-1).shape))

        self.net = nn.Sequential(
            nn.Linear(in_feat,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,actions)
        )
    
    def forward(self,x):
        return self.net(x)
    
    def act(self,x):
        obs_t = torch.as_tensor(x,dtype=torch.float32).to('cuda')
  
        q_val = self(obs_t.unsqueeze(0))
    
        max_q_index  = torch.argmax(q_val,dim=1)[0] 
        action = max_q_index.detach().item()
        return action
   
   
net = Network().to('cuda')
net.load_state_dict(torch.load('q_network_checkpoint_space.pt'))
net.eval()

obs , _ , _ = space_game.reset()
print(obs)
for step in itertools.count():

        action = net.act(np.array(obs).reshape(-1))
        print(action)
        
        new_obs , reward , done  = space_game.play_step(action)
        obs = new_obs
        
        if done:
      
            obs , _ , _ = space_game.reset()
            
  
        