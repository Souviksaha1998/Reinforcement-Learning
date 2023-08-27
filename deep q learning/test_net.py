import torch
import gym 
import itertools
import torch.nn as nn
import numpy as np

env = gym.make("LunarLander-v2",)



env = gym.make("LunarLander-v2",render_mode='human')

class Network(nn.Module):
    def __init__(self,env):
        super().__init__()

        in_feat = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_feat,128),
            nn.Tanh(),
            nn.Linear(128,256),
            nn.Tanh(),
            nn.Linear(256,512),
            nn.Tanh(),
            nn.Linear(512,env.action_space.n)
        )
    
    def forward(self,x):
        return self.net(x)
    
    def act(self,x):
        obs_t = torch.as_tensor(x,dtype=torch.float32).to('cuda')
        q_val = self(obs_t.unsqueeze(0))
        max_q_index  = torch.argmax(q_val,dim=1)[0] 
        action = max_q_index.detach().item()
        return action
    


online_net= Network(env).to('cuda')
online_net.load_state_dict(torch.load('q_network_checkpoint.pt'))
online_net.eval()



obs , _ = env.reset()
for step in itertools.count():

    action = online_net.act(obs)

    new_obs , reward , done , *additional_values = env.step(action)
    obs = new_obs
    env.render()
    print(done)
    if done:
        osb , _ = env.reset()
        
env.close()