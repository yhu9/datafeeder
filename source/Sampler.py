
import itertools
import torch
import torch.nn as nn
import resnet
from Memory import ReplayMemory

#DQN Network
class DQN(nn.Module):

    def __init__(self,actionsize):
        super(DQN,self).__init__()

        self.res18 = resnet.resnet18()
        self.m1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1000,actionsize)
                )

    def forward(self,weights):
        x = self.res18(weights)
        x = self.m1(x)
        return x

class Sampler():

    def __init__(self,device,actionsize):
        self.samplenet = DQN(actionsize).to(device)
        self.targetnet = DQN(actionsize).to(device)
        self.opt = torch.optim.Adam(itertools.chain(self.samplenet.parameters()),lr=0.00001,betas=(0.0,0.9))
        self.device = device
        self.memory = ReplayMemory(1000,device=device)
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99
        self.count = 0

    def select_action(self, model):
        self.samplenet.eval()
        action = self.samplenet(model.conv2.weight.data.view(-1,5,5).unsqueeze(0))
        return torch.max(action,1)[1]

    def step(self,state,action,next_state,reward,done):
        self.memory.push(state,action,next_state,reward,done)

        #don't bother if you don't have enough in memory
        if len(self.memory) >= self.BATCH_SIZE:
            self.optimize()

    def optimize(self):
        self.samplenet.train()
        self.targetnet.eval()
        s1,actions,r1,s2,d = self.memory.sample(self.BATCH_SIZE)

        #get old Q values and new Q values for belmont eq
        qvals = self.samplenet(s1)
        state_action_values = qvals.gather(1,actions[:,0].unsqueeze(1))
        with torch.no_grad():
            qvals_t = self.targetnet(s2)
            q1_t = qvals_t.max(1)[0].unsqueeze(1)

        expected_state_action_values = (q1_t * self.GAMMA) * (1-d) + r1

        #LOSS IS l2 loss of belmont equation
        loss = torch.nn.MSELoss()(state_action_values,expected_state_action_values)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.count % 20 == 0:
            self.targetnet.load_state_dict(self.samplenet.state_dict())

        return loss.item()

if __name__ == '__main__':
    import numpy as np

    weights = torch.Tensor(np.zeros((10,50,5,5)))
    a = Actor()
    c = Critic()
    action = a(weights)
    qval = c(weights,action)
    print(qval)


