import torch.nn as nn
import torch.nn.funnctional as F


class QNetwork(nn.Module):
	def __init__(state_size, action_size):
		super().__init__()
		self.fc1 = nn.Linear(state_size, 178)
		self.fc2 = nn.Linear(178, 64)
		self.fc3 = nn.Linear(64, action_size)

	def forward(state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)