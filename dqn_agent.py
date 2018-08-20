
import numpy as np
from replay_buffer import ReplayBuffer
import torch
from torch import optim
import torch.nn.functional as F
import random
from torchvision import transforms

device = 'cpu'
LR = 5e-3
BATCH_SIZE = 512
BUFFER_SIZE = int(1e5) 
UPDATE_RATE = 4
GAMMA = 0.99
TAU= 1e-3


class Agent():
	def __init__(self, state_size, action_size, seed, training, pixels):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		self.t_step = 0.
		self.pixels = pixels
		if pixels is False:
			from q_network import QNetwork
		else:
			from q_network_cnn import QNetwork
			print('loaded cnn network')
			self.loader = transforms.Compose([
			    transforms.ToTensor(),
			    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		self.QN_local = QNetwork(state_size, action_size, seed, training).to(device)
		self.QN_target = QNetwork(state_size, action_size, seed, training).to(device)
		self.optimizer = optim.Adam(self.QN_local.parameters(), lr=LR )
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device) # TODO

	def act(self, state, eps):
		if self.pixels is True:
			state = self.loader(np.squeeze(state))
		else:
			state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.QN_local.eval()

		if torch.no_grad():
			action_values = self.QN_local(state)
		self.QN_local.train()
		print(action_values)
		if random.random() > eps:
			return int(np.argmax(action_values.cpu().data.numpy()))
		else:
			return int(random.choice(np.arange(self.action_size)))

	def step(self, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)

		self.t_step = (self.t_step + 1) % UPDATE_RATE
		if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
			samples = self.memory.sample()
			self.learn(samples, GAMMA)

	def learn(self, experiences, gamma):
		states, actions, rewards, next_states, dones = experiences
		_target = self.QN_target(next_states).detach().max(1)[0].unsqueeze(1) 
		action_values_target = rewards + gamma * _target * (1-dones)
		action_values_expected = self.QN_local(states).gather(1, actions)

		loss = F.mse_loss(action_values_expected, action_values_target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# update target Qnetwork
		for target_param, local_param in zip(self.QN_target.parameters(), self.QN_local.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

               


