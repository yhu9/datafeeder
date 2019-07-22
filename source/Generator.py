import itertools
import os
import torch
import torch.nn as nn
import resnet
from Memory import ReplayMemory
import random

#generator
class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()

        self.res = resnet.resnet18()
        self.conv = nn.Conv2d(512,1,3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.res(x)
        x = self.conv(x)
        x = self.bn(x)
        img = self.sig(x)
        return img

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()

        self.res = resnet.resnet18()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()

    def forward(self,action):
        x = self.res(action)
        x = self.avgpool(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        x = self.sig(x)
        return x

class Generator():
    def __init__(self,device,data):
        self.data = data
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        #self.ctarget = Critic().to(device)
        self.actor_opt = torch.optim.Adam(itertools.chain(self.actor.parameters()),lr=0.0001,betas=(0.0,0.9))
        self.critic_opt = torch.optim.Adam(itertools.chain(self.critic.parameters()),lr=0.001,betas=(0.0,0.9))
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        #self.ctarget.apply(init_weights)

        self.device = device
        self.memory = ReplayMemory(1000,device=device)
        self.batch_size = 5
        self.GAMMA = 0.99
        self.count = 0

    def select_action(self,imgs):
        with torch.no_grad():
            self.actor.eval()
            action = self.actor(imgs)
            return action

    def step(self,state,action,next_state,reward,done):
        self.memory.push(state,action,next_state,reward,done)

        if len(self.memory) >= self.batch_size:
            self.optimize()

    def optimize(self):
        self.actor.train()
        self.critic.train()
        #self.ctarget.eval()

        s1,a,r,s2,d = self.memory.sample(self.batch_size)

        #train the critic
        for reward,action in zip(r,a):
            qval = self.critic(action)
            avgQ = qval.mean().unsqueeze(0)
            loss = torch.nn.L1Loss()(avgQ,reward)
            self.critic_opt.zero_grad()
            loss.backward()
            self.critic_opt.step()

        #train the actor
        img,target = self.data[random.randint(0,len(self.data)-1)]
        batch = self.actor(img)
        score = self.critic(batch)
        actor_loss = -score.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        #if self.count % 5 == 0:
        #    self.ctarget.load_state_dict(self.critic.state_dict())
        #self.count += 1

    def save(self):
        torch.save(self.actor.state_dict(),os.path.join('model','actor.pth'))
        torch.save(self.critic.state_dict(),os.path.join('model','critic.pth'))

if __name__ == '__main__':
    import numpy as np

    img = torch.Tensor(np.random.normal(0,1,10*28*28).reshape((10,1,28,28)))

    a = Actor()
    c = Critic()
    action = a(img)
    print(action.shape)
    qval = c(action)
    print(qval)

