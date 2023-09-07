import torch.nn as nn
import torch
import numpy as np
import itertools
import random
from custom_env_class import SpaceGame
from collections import deque
import cv2

# setting hyperparameter
GAMMA = 0.999
LR = 3e-05
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.00001
TARGET_UPDATE_FREQUENCY = 2000
MIN_REPLAY_SIZE = 3000
BATCH = 128
BUFFER_SIZE = 100000
FRAME_STACK = 4

space_game = SpaceGame(120)
actions = len(space_game.actions_())
observation , _  , _= space_game.reset()


replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0],maxlen=100)
frames_buffer = deque(maxlen=FRAME_STACK)
episode_reward = 0.0

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        

        self.net = nn.Sequential(nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU())
        
        self.fc = nn.Sequential(nn.Linear(64*16*16,512),
                                nn.ReLU(),
                                nn.Linear(512,actions))
        
    
    def calculate(self):
        image = torch.randn(1,4,160,160)
        val = self.net(image)
        
        output = self.fc(val)
        return output
        
 
    def forward(self,x):
        return self.fc(self.net(x))
    
    def act(self,x):
        obs_t = torch.as_tensor(x,dtype=torch.float32).to('cuda')
  
        q_val = self(obs_t.unsqueeze(0))
    
        max_q_index  = torch.argmax(q_val,dim=1)[0] 
        action = max_q_index.detach().item()
        return action
    
def epsilon_decay(initial_epsilon, final_epsilon, decay_rate, step):
    
    epsilon = final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-decay_rate * step)
    return epsilon

online_net = Network().to('cuda')
target_net = Network().to('cuda')

target_net.load_state_dict(online_net.state_dict())
target_net.eval()


optim = torch.optim.Adam(online_net.parameters(),lr=LR)


def preprocess_im(image):
    images  = cv2.resize(image,(160,160))
    
    
    images = cv2.cvtColor(images , cv2.COLOR_BGR2GRAY)
    images = images / 255.0
    
    return images

rewardsss_ = 0      
observations__ , _ , _ = space_game.reset()

observations__ = preprocess_im(observations__)


for step in itertools.count():
    
  
    
    # if len(frames_buffer) >= 4:
  
        
 
        epsilon = epsilon_decay(EPSILON_START, EPSILON_END, EPSILON_DECAY, step)

        # action will give index
        if np.random.random() >= epsilon:
            
            action = online_net.act(observations__)
        else:
            action = space_game.random_action()
            
            


        new_observation , reward , done  = space_game.play_step(action)
        new_observation = preprocess_im(new_observation)
        transition = (observations__,action,reward,done,new_observation)
        replay_buffer.append(transition)
        
       
      

        episode_reward += reward
    
        
        if done:
            
            observations__ , _ ,_= space_game.reset()
            frames_buffer.clear()
            # print(f'game reset : {obs}')
            reward_buffer.append(episode_reward)
            episode_reward = 0.0
    
   
        
        
        if step > MIN_REPLAY_SIZE:
        # sample random experience 
            transition = random.sample(replay_buffer,BATCH)

            obs_ = np.asarray([t[0] for t in transition])
            action_ = np.asarray([t[1] for t in transition]) 
            reward_ = np.asarray([t[2] for t in transition]) 
            done_ = np.asarray([t[3] for t in transition]) 
            new_obs_ = np.asarray([t[4] for t in transition]) 
        
            obs_t  = torch.as_tensor(obs_ , dtype=torch.float32).to('cuda')
            action_t  = torch.as_tensor(action_ , dtype=torch.int64).unsqueeze(-1).to('cuda')
            reward_t  = torch.as_tensor(reward_ , dtype=torch.float32).unsqueeze(-1).to('cuda')
            done_t  = torch.as_tensor(done_ , dtype=torch.float32).unsqueeze(-1).to('cuda')
            new_obs_t = torch.as_tensor(new_obs_ , dtype=torch.float32).to('cuda')
            
        
            ##target q values 
            target_q = target_net(new_obs_t.view(new_obs_t.size(0),-1))
            max_target_q = target_q.max(dim=1,keepdim=True)[0]
        
            targets = reward_t + (1-done_t) * GAMMA * max_target_q
                
                
            q_values = online_net(obs_t.view(obs_t.size(0),-1))
            action_q_value = torch.gather(q_values,dim=1,index=action_t)
            
            loss = nn.functional.smooth_l1_loss(action_q_value , targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % TARGET_UPDATE_FREQUENCY == 0:
                target_net.load_state_dict(online_net.state_dict())
                
            if step % 1000 == 0:
                re = np.mean(reward_buffer)
                print()
                print(f'Current step : {step}')
                print(f'avg reward : {re}')
                print(f'Epsilon : {epsilon}')
            

                if  re > rewardsss_:
                    
                    print(f'New weights found !')
                    torch.save(online_net.state_dict(), 'q_network_checkpoint_space.pt')
                    rewardsss_ = re
        
        