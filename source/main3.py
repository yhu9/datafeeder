from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

from Generator import Generator

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def getState(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

import matplotlib.pyplot as plt
def tile_img(f,axarr,imgs):
    for i,img in enumerate(imgs):
        img = img.permute(1,2,0).view(28,28)
        row = i // 8
        col = i % 8
        axarr[row,col].imshow(img,cmap='gray')

#train the network
def train(args, model, generator,device, training_data,test_loader):
    base_correct = test(args,model,device,test_loader) / 10000.0
    f1,axarr1 = plt.subplots(8,8)
    f2,axarr2 = plt.subplots(8,8)
    plt.ion()
    plt.show()
    while True:
        targetmodel = Net().to(device)
        optimizer = optim.SGD(targetmodel.parameters(), lr=args.lr, momentum=args.momentum)
        targetmodel.load_state_dict(model.state_dict())
        actions = []
        next_states = []
        dones = []
        for i in range(10):
            #get a batch
            img,target = training_data[random.randint(0,len(training_data)-1)]
            img = img.to(device)
            target = target.to(device)
            #create a new batch given current batch
            action = generator.select_action(img)

            #normal mnist training
            targetmodel.train()
            optimizer.zero_grad()
            output = targetmodel(action)
            loss = F.nll_loss(output,target)
            loss.backward()
            optimizer.step()

            actions.append(action)
            next_states.append(targetmodel.conv2.weight.data.clone().view(-1,5,5))

        #view action
        tile_img(f1,axarr1,img)
        tile_img(f2,axarr2,action)
        plt.draw()
        plt.pause(0.001)
        plt.show()

        #test the new model
        correct = test(args,targetmodel,device,test_loader) / 10000.0

        #train the generator
        r = correct

        state = model.conv2.weight.data.clone().view(-1,5,5)
        next_state = targetmodel.conv2.weight.data.clone().view(-1,5,5)
        for a,s2 in zip(actions,next_states):
            generator.step(state,a,next_state,r,True)

        if correct > base_correct:
            print(r)
            generator.save()

    plt.ioff()


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--orig', action='store_false', default=True,
                        help='For running the new datafeeder model')
    args = parser.parse_args()

    #####################################################################################################
    #####################################################################################################
    #####################################################################################################
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    #store all data into memory
    data = []
    for batch_idx, state in enumerate(train_loader):
        data.append(state)
    training_data = data[:5]    #we take 0.5% of the total amount of data

    #INITIALIZE THE MODEL
    model = Net().to(device)

    #LOAD Pretrained mnist model from main2
    model.load_state_dict(torch.load('model/mnist_cnn.pt'))

    #GENERATE DATA TO IMPROVE PERFORMANCE
    generator = Generator(device,training_data)
    train(args,model,generator,device,training_data,test_loader)

if __name__ == '__main__':
    main()


