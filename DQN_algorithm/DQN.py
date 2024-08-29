import copy
import gym
import math
import numpy as np
from torch import argmax
import torch
from torch import nn
from random import sample
from collections import deque
import matplotlib.pyplot as plt
from neural_network import FeedForwardNetwork, sigmoid, tanh, relu, leaky_relu, linear, ActivationFunction


class ReplayBuffer:
    
    def __init__(self, size):
        self.buffer = deque([], maxlen = size)
        
    def append(self, transition):
        self.buffer.append(transition)
        
    def sample(self, sample_size):
        return sample(self.buffer, sample_size)
    
    def __len__(self):
        return len(self.buffer)



class DQN(FeedForwardNetwork):
    def __init__(self, layer_nodes, hidden_activation, output_activation, lr=0.01):
        super().__init__(layer_nodes, hidden_activation, output_activation)
        self.lr = lr

        
        self.params_tensor = self._params_numpy_to_tensor()
        self.optimizer = torch.optim.SGD(self.params_tensor, lr=self.lr)
        self.loss_function = torch.nn.MSELoss()

    
    def save_torch_params(self):
        self._params_tensor_to_numpy()

    def _params_numpy_to_tensor(self):
        tensors = []
        for l in range(1, len(self.layer_nodes)):
            W_key = f'W{l}'
            b_key = f'b{l}'
          
            W_tensor = torch.tensor(self.params[W_key], dtype=torch.float32, requires_grad=True)
            b_tensor = torch.tensor(self.params[b_key], dtype=torch.float32, requires_grad=True)

            tensors.extend([W_tensor, b_tensor])
        return tensors
    
    def _params_tensor_to_numpy(self):
        for l in range(1, len(self.layer_nodes)):
            W_key = f'W{l}'
            b_key = f'b{l}'

            self.params[W_key] = self.params_tensor[2 * (l - 1)].detach().numpy()
            self.params[b_key] = self.params_tensor[2 * (l - 1) + 1].detach().numpy() 

    

class DQNAgent():

        def __init__(self, 
                     num_actions, 
                     num_states, 
                     model, 
                     ):
        
            # Hyperparameters
            self.sync_network_rate = 200
            self.batch_size = 25
            self.buffer_size = 10_000
            self.discount_value=0.99

            epsilon_start=0.8, 
            epsilon_min=0.01, 
            epsilon_decay=0.995, 

            self.num_states = num_states
            self.num_actions = num_actions
            self.model = model
            self.step_counter = 0

            self.epsilon = epsilon_start

            # Target model initialisation
            self.target_model = copy.deepcopy(self.model)
            # Replay buffer initialisation
            self.replay_buffer = ReplayBuffer(self.buffer_size)


        def choose_action(self, state): 
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, self.num_actions)
            else:
                action = self.choose_best_action(state)
            return action


        def choose_best_action(self, state):
            return argmax(self.model.feed_forward(state))

        def sync_network(self):
            if self.step_counter % self.sync_network_rate == 0:
                self.target_model.params = self.model.params
        
        def decay_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        def learn(self):

            samples = self.replay_buffer.sample(self.batch_size)
            
            for sample in samples:

                state, action, new_state, reward = sample

                predicted_value = self.model.feed_forward(state)[action] # Q(s, a)
                predicted_next_values = self.target_model.feed_forward(new_state) # Q(s', a')
                target_value = reward + self.discount_value * max(predicted_next_values) # r + γQ(s', a')

                target_value = torch.tensor(target_value, dtype=torch.float32, requires_grad=False)
                predicted_value =  torch.tensor(predicted_value, dtype=torch.float32, requires_grad=True)
                # The difference between Q(s, a) and r + γQ(s', a') is used to calculated the loss function:
                loss = self.model.loss_function(predicted_value, target_value)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()      
                self.model.save_torch_params()
                self.step_counter += 1
                self.sync_network()
                self.decay_epsilon