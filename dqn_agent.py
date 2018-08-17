
import numpy as np
from q_network import QNetwork
from replay_buffer import ReplayBuffer
import torch
from torch import optim
import torch.functional as F
import random

device = 'cpu'
LR = 0.01
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5) 
UPDATE_RATE = 4
GAMMA = 0.99
TAU= 1e-3

class DQNAgent:
	def __init__(self, state_size, action_size, seed):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		self.t_step = 0.
		
		self.QN_local = QNetwork(state_size, action_size, seed).to(device)
		self.QN_target = QNetwork(state_size, action_size, seed).to(device)
		self.optimizer = optim.Adam(self.QN_local.parameters(), lr=LR )
		self.memory = ReplayBuffer() # TODO

	def act(self, state, eps):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.QN_local.eval()

		if torch.no_grad():
			action_values = self.QN_local(state)
		self.QN_local.train()

		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))

	def step(self, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)

		self.t_step = (self.t_step + 1) % UPDATE_RATE
		if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
			samples = self.memory.sample()
			self.learn(samples, GAMMA)

	def learn(self, experiences, gamma):
		states, actions, next_states, rewards, dones = experiences
		action_values_target = rewards + gamma * self.QN_target(next_states).detach().max(1)[0].unsqueeze(1) * (1-dones)
		action_values_expected = self.QN_local(states).gather(1, actions)

		self.optimizer.zero_grad()
		loss = F.mse_loss(action_values_expected, action_values_target)
		loss.backward()
		self.optimizer.step()

		# update target Qnetwork
		for target_param, local_param in zip(self.QN_target.parameters(), self.QN_local.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
