from collections import namedtuple
import torch
import random

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity,device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device=device

    def push(self,state,action,next_state,reward,done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        reward = torch.Tensor([reward])
        done = torch.Tensor([done])
        self.memory[self.position] = Transition(state,action,next_state,reward,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        s = [None] * batch_size
        a = [None] * batch_size
        r = [None] * batch_size
        s2 = [None] * batch_size
        d = [None] * batch_size
        for i,e in enumerate(experiences):
            s[i] = e.state
            a[i] = e.action
            r[i] = e.reward
            s2[i] = e.next_state
            d[i] = e.done

        states = torch.stack(s).to(self.device)
        actions = torch.stack(a).to(self.device)
        rewards = torch.stack(r).to(self.device)
        next_states = torch.stack(s2).to(self.device)
        dones = torch.stack(d).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)



