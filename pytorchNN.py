import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def dataGenerator(b):
    # generate input
    i = torch.rand(2,b)
    # calculate output
    o = torch.zeros(2,b)
    o[0] = i[0]**2 + 3*i[1]
    o[1] = i[0]*i[1] + .5*i[0]

    return torch.stack([torch.transpose(i, 0, 1), torch.transpose(o, 0, 1)])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 80)
        self.fc3 = nn.Linear(80, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, optimizer, criterion, steps, batchSize):
    for i in range(steps):
        data = dataGenerator(batchSize)
        inputs = data[0]
        target = data[1]

        optimizer.zero_grad()

        out = net(inputs)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        print(loss.item())

def test(net, criterion, batchSize):
    data = dataGenerator(batchSize)
    inputs = data[0]
    target = data[1]

    out = net(inputs)
    loss = criterion(out, target)

    print("test loss:", loss.item())

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()

#train(net, optimizer, criterion, 100, 1000)
#test(net, criterion, 100000)

import code
code.interact(local=locals())
