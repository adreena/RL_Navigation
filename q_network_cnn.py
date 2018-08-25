import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torchvision import transforms
from collections import OrderedDict

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, training ):
        super(QNetwork, self).__init__()

        self.c1=nn.Conv3d(in_channels = 3,  out_channels=10, kernel_size=(1,5,5) , stride=1)
        self.r1 = nn.ReLU()
        self.max1 = nn.MaxPool3d((1,2,2))
        
        # (32-5+ 0)/1 + 1 -> 28x28x10 -> 14x14x10
        self.c2 = nn.Conv3d(in_channels=10, out_channels=32, kernel_size=(1,5,5) , stride=1)
        self.r2 = nn.ReLU()
        self.max2 = nn.MaxPool3d((1,2,2))
            
        # 14-5 +1 -> 5x5x32 
        self.fc4 = nn.Linear(5*5*32*3, action_size)
#         self.r4 = nn.ReLU()
#         self.fc5 = nn.Linear(84, action_size)
    
    def forward(self, img_stack):
#         print('-',img_stack.size())
        output = self.c1(img_stack)
        
        output = self.r1(output)
        output = self.max1(output)
#         print('*',output.size())
        
        output = self.c2(output)
        output = self.r2(output)
        output = self.max2(output)
#         print('**',output.size())
        
        output = output.view(output.size(0), -1)
#         print('***', output.size())
        output = self.fc4(output)
        return output