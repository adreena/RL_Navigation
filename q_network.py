import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, training ):
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, 64)
		self.fc2 = nn.Linear(64, 512)
		self.fc3 = nn.Linear(512, 64)
		self.fc4 = nn.Linear(64, action_size)
		self.training = training

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)