import torch.nn as nn
import torch
import gym
from collections import deque 
import numpy as np
import itertools
import random


# setting hyperparameter
GAMMA = 0.99
LR = 0.0001
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.0001
TARGET_UPDATE_FREQUENCY = 1000
MIN_REPLAY_SIZE = 1000
BATCH = 32
BUFFER_SIZE = 10000


env = gym.make('CartPole-v1')

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0],maxlen=100)
episode_reward = 0.0


class Network(nn.Module):
    def __init__(self,env):
        super().__init__()

        in_feat = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_feat,65),
            nn.Tanh(),
            nn.Linear(65,env.action_space.n)
        )
    
    def forward(self,x):
        return self.net(x)
    
    def act(self,x):
        obs_t = torch.as_tensor(x,dtype=torch.float32)
        q_val = self(obs_t.unsqueeze(0))
        max_q_index  = torch.argmax(q_val,dim=1)[0] 
        action = max_q_index.detach().item()
        return action




def epsilon_decay(initial_epsilon, final_epsilon, decay_rate, step):
    
    epsilon = final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-decay_rate * step)
    return epsilon



online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optim = torch.optim.Adam(online_net.parameters(),lr=LR)

obs , _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
  
    new_obs , reward , done , *additional_values = env.step(action)
    # print(f'reward')
    transition = (obs,action,reward,done,new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        osb , _ = env.reset()



obs , _ = env.reset()
rewards = 0
for step in itertools.count():

    epsilon = epsilon_decay(EPSILON_START, EPSILON_END, EPSILON_DECAY, step)

  
    # action will give index
    if np.random.random() >= epsilon:
        action = online_net.act(obs)
    else:
        action = env.action_space.sample()


    new_obs , reward , done , *additional_values = env.step(action)
    transition = (obs,action,reward,done,new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += reward

 

    if done:
        osb , _ = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    if len(reward_buffer) >= 100:
        if np.mean(reward_buffer) >= 195:
            env.render()

    # sample random experience 
    transition = random.sample(replay_buffer,BATCH)

    obs_ = np.asarray([t[0] for t in transition])
    action_ = np.asarray([t[1] for t in transition]) 
    reward_ = np.asarray([t[2] for t in transition]) 
    done_ = np.asarray([t[3] for t in transition]) 
    new_obs_ = np.asarray([t[4] for t in transition]) 
   
    obs_t  = torch.as_tensor(obs_ , dtype=torch.float32)
    action_t  = torch.as_tensor(action_ , dtype=torch.int64).unsqueeze(-1)
    reward_t  = torch.as_tensor(reward_ , dtype=torch.float32).unsqueeze(-1)
    done_t  = torch.as_tensor(done_ , dtype=torch.float32).unsqueeze(-1)
    new_obs_t = torch.as_tensor(new_obs_ , dtype=torch.float32)

   
    # target q values 
    target_q = target_net(new_obs_t)
    max_target_q = target_q.max(dim=1,keepdim=True)[0]
    targets = reward_t + (1-done_t) * GAMMA * max_target_q


    
    # # compute loss
    q_values = online_net(obs_t)
    action_q_value = torch.gather(q_values,dim=1,index=action_t)

    #loss 
    loss = nn.functional.mse_loss(action_q_value , targets)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(online_net.state_dict())

    
    

    


    if step % 1000 == 0:
        reward = np.mean(reward_buffer)
        print()
        print(f'Current step : {step}')
        print(f'avg reward : {reward}')

        if reward > rewards:
            print(f'New weights found !')
            torch.save(online_net.state_dict(), 'q_network_checkpoint.pt')
            rewards = reward


env.close()