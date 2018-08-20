import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class QNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, training ):
		super(QNetwork, self).__init__()

		self.seed = torch.manual_seed(seed)
		self.conv1 = nn.Conv2d(1,10, kernel_size=(5,5))
		self.conv2 = nn.Conv2d(10,20, kernel_size=(5,5))
		self.fc1 = nn.Linear(120, 84)
		self.fc2 = nn.Linear(84, action_size)
		self.training = training

	def forward(self, state):
		x = self.conv1(state)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = F.relu(x)
		x = x.view(-1,120)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)