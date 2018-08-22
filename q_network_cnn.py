import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, training ):
        super(QNetwork, self).__init__()

        self.c1=nn.Conv2d(in_channels = 3,  out_channels=6, kernel_size=5, stride=1)
        self.r1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.r2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.r3 = nn.ReLU()
        
        self.fc4 = nn.Linear(9*9*120, 84)
        self.r4 = nn.ReLU()
        self.fc5 = nn.Linear(84, action_size)
    
    def forward(self, img):
        output = self.c1(img)
        output = self.r1(output)
        output = self.max1(output)
        output = self.c2(output)
        output = self.r2(output)
        output = self.max2(output)
        output = self.c3(output)
        output = self.r3(output)
        output = output.view(output.size(0), -1)
        output = self.fc4(output)
        output = self.r4(output)
        output = self.fc5(output)
        return output