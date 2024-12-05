import copy
from typing import Dict, List, Optional, Union
import numpy as np
from torch import argmax
import torch
from torch import nn
from random import sample
from collections import deque, OrderedDict
from config import Config, fitness_func
from mario import Mario
from neural_network import FeedForwardNetwork, get_activation_by_name, sigmoid, tanh, relu, leaky_relu, linear, ActivationFunction
from utils import SMB
from mario import get_num_inputs



###
# This is a custom implementation of a DQN Agent

class ReplayBuffer:
    
    def __init__(self, size):
        self.buffer = deque([], maxlen = size)
        
    def append(self, transition):
        self.buffer.append(transition)
        
    def sample(self, sample_size):
        return sample(self.buffer, sample_size)
        
    
    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module, FeedForwardNetwork):
    def __init__(self, layer_nodes, hidden_activation, output_activation, lr=0.01):
        #FeedForwardNetwork.__init__(self, layer_nodes, hidden_activation, output_activation)
        nn.Module.__init__(self)
        self.lr = lr
        self.layer_nodes = layer_nodes

        print(f"output_activation: {output_activation}")
        
        self.x_size = layer_nodes[0]
        self.y_size = layer_nodes[-1]
        
        self.torch_model =  self.build_torch_model()

        self.optimizer = torch.optim.SGD(self.torch_model.parameters(), lr=self.lr)
        self.loss_function = torch.nn.MSELoss()
        
        self.sumRewardsEpisode=[]
        
        
  
    def to_torch_activation(self, activation_function):
        if activation_function == sigmoid:
            return nn.Sigmoid()
        elif activation_function == tanh:
            return nn.Tanh()
        elif activation_function == relu:
            return nn.ReLU()
        elif activation_function == leaky_relu:
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation_function == linear:
            return nn.Identity()
        else:
            raise ValueError("unsupported activation function")
        
    def build_torch_model(self):
        L = len(self.layer_nodes) - 1  # len(self.params) // 2
        net = OrderedDict()
        # build hidden layers
        for l in range(1, L):
            net[f'L{l}'] = nn.Linear(self.layer_nodes[l-1], self.layer_nodes[l])
            net[f'Afn{l}'] = self.hidden_activation()
            
        # build output layer
        print(self.output_activation)
        net[f'L{L}'] = nn.Linear(self.layer_nodes[L-1], self.layer_nodes[L])
        net[f'Afn{L}'] =  self.output_activation()
        
        return nn.Sequential(net)
          

    # This one is for FeedForwardNetwork compatability
    def feed_forward(self, X: np.ndarray) -> np.ndarray:

        # Ensures the torch params are synced to the FeedForwardNetwork params
        self.save_torch_params()
        
        A_prev = X
        L = len(self.layer_nodes) - 1  # len(self.params) // 2

        # Feed hidden layers
        for l in range(1, L):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_prev = self.hidden_activation(Z)
            self.params['A' + str(l)] = A_prev

        # Feed output
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A_prev) + b
        out = self.output_activation(Z)
        self.params['A' + str(L)] = out

        self.out = out
        return out
        
    
    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        
        if X.shape[0] == self.x_size and X.shape[1] == 1:  # Case where it's transposed
            X = X.view(1, -1)  # Reshape to [1, 80]

        if len(X.shape) == 3 and X.shape[2] == 1:  # Extra last dimension
            X = X.squeeze(-1)  # Remove the last dimension
        
        out = self.torch_model(X)
        return out

    
    def save_torch_params(self):
        self._params_tensor_to_numpy()
        
    
    def _params_tensor_to_numpy(self):
        for l, (name, param) in enumerate(self.torch_model.named_parameters()):
            if l % 2 == 0:
                W_key = f'W{l//2 + 1}'
                self.params[W_key] = param.detach().numpy()
            else:
                b_key = f'b{l//2 + 1}'
                self.params[b_key] = param.detach().numpy().reshape(-1, 1)

    

class DQNAgent():

        def __init__(self, 
                     num_actions, 
                     num_states, 
                     network,
                     sync_network_rate,
                     batch_size,
                     discount_value,
                     epsilon_start,
                     epsilon_min,
                     epsilon_decay,
                     learning_rate,
                     buffer_size
                     ):
            
            self.sync_network_rate = sync_network_rate
            self.batch_size = batch_size
            self.discount_value = discount_value
            self.epsilon_start = epsilon_start
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.learning_rate = learning_rate
            self.buffer_size = buffer_size

            self.num_states = num_states
            self.num_actions = num_actions
            self.network = network
            self.network.lr = self.learning_rate
            self.step_counter = 0
            self.curr_loss = -1

            self.epsilon = self.epsilon_start


            # Target network initialisation
            self.target_network = copy.deepcopy(self.network)
            # Replay buffer initialisation
            self.replay_buffer = ReplayBuffer(self.buffer_size)

            self.keys_to_output_map = {
                4: 0,  # U
                5: 1,  # D
                6: 2,  # L
                7: 3,  # R
                8: 4,  # A
                0: 5  # B
            }


        def choose_action(self, state): 
            if np.random.random() < self.epsilon:
                #num_indices = np.random.choice([1, 2, 3], p=[0.45, 0.45, 0.1])
                #action = np.random.choice(range(self.num_actions), size=num_indices, replace=False)
                action = np.random.choice(range(self.num_actions), size=1, replace=False)
            else:
                action = self.choose_best_action(state)
            return action


        def choose_best_action(self, state):
            out = self.network.forward(state)
            #scaled = torch.sigmoid(out)
            #threshold = torch.nonzero(scaled > 0.5, as_tuple=True)[0].cpu().numpy()
            highest_rated_action = np.array([torch.argmax(out)])
            return highest_rated_action
        
        def choose_best_action_F(self, state):
            out = self.network.feed_forward(state)
            return np.argmax(out)

        def sync_network(self):
            if self.step_counter % self.sync_network_rate == 0 and self.step_counter != 0:
                self.target_network.torch_model.load_state_dict(self.network.torch_model.state_dict())

        
        def decay_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
        def learn(self):
            
            if self.step_counter % self.batch_size != 0 or self.step_counter == 0:
                return
            

            samples = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, rewards, dones = zip(*samples)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            next_states = torch.FloatTensor(np.array(next_states))
            rewards = torch.FloatTensor(np.array(rewards))
            dones = torch.FloatTensor(np.array(dones))

            # Q-values for all states
            predicted_q_values = self.network.forward(states)  # Shape: [batch_size, num_actions]

            # Converts key action indices to output indices (bound between 0 and output layer size)
            action_indices = actions.argmax(dim=1).squeeze(-1) # extract the index of the one-hot encoded action

            action_indices = torch.tensor([self.keys_to_output_map[item.item()] for item in action_indices]) # map that index to the index of the network's output

            # extract the Q-values we are interested in (chosen actions)
            batch_indices = torch.arange(states.size(0))
            predicted_q_values = predicted_q_values[batch_indices, action_indices]


            # Compute target Q-values for next states
            with torch.no_grad():
                next_max_q_values = self.target_network.forward(next_states).max(dim=1)[0]  # Qmax(s', a')
                target_q_values = rewards + self.discount_value * next_max_q_values * (1 - dones) # r + gamma * max_a' Qmax(s', a')
                         

            # Calculate loss
            loss = self.network.loss_function(predicted_q_values, target_q_values)
 
            self.curr_loss = int(loss.item())

            # backpropagation via SGDs
            self.network.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.torch_model.parameters(), max_norm=1.0)
            self.network.optimizer.step()      



# This is the interface that allows the DQN implementation to be
# compatible with the rest of the code
class DQNMario(DQNAgent, Mario):
    def __init__(self, 
                 config: Config,
                 name: Optional[str] = "DQNAgent",
                 debug: Optional[bool] = False,):
        self.config = config
        self.reward_func = fitness_func
        self.hidden_activation = self.config.NeuralNetworkDQN.hidden_node_activation
        self.output_activation = self.config.NeuralNetworkDQN.output_node_activation
        self.hidden_layer_architecture = self.config.NeuralNetworkDQN.hidden_layer_architecture
      
        Mario.__init__(self, config, None, self.hidden_layer_architecture, self.hidden_activation,
         self.output_activation, np.inf, name, debug)
        
        # overwrite them because Mario's constructor actually just sets it to GA's config's values again fantastic
        self.hidden_activation = self.config.NeuralNetworkDQN.hidden_node_activation
        self.output_activation = self.config.NeuralNetworkDQN.output_node_activation

        if self.config.NeuralNetworkDQN.encode_row:
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height

        self.network_architecture = [num_inputs]                          # Input Nodes
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden Layer Ndoes
        self.network_architecture.append(6)                        # 6 Outputs ['u', 'd', 'l', 'r', 'a', 'b']
        print(f"Network Architecture: {self.network_architecture}")
        ## Parameter initialisation
        self.learning_rate = self.config.DQN.learning_rate
        self.buffer_size = self.config.DQN.buffer_size
        self.sync_network_rate = self.config.DQN.sync_network_rate
        self.batch_size = self.config.DQN.batch_size
        self.discount_value = self.config.DQN.discount_value
        self.epsilon_start = self.config.DQN.epsilon_start
        self.epsilon_min = self.config.DQN.epsilon_min
        self.epsilon_decay = self.config.DQN.epsilon_decay


        model = DQN(layer_nodes=self.network_architecture, hidden_activation=get_activation_by_name(self.hidden_activation), output_activation=get_activation_by_name(self.output_activation))
        DQNAgent.__init__(self, self.network_architecture[-1], get_num_inputs(self.config), model, self.sync_network_rate, self.batch_size, self.discount_value, self.epsilon_start, self.epsilon_min, self.epsilon_decay, self.learning_rate, self.buffer_size)

    def calculate_reward(self, prev_stats, next_stats):

        prev_frames, prev_distance, game_score = prev_stats
        next_frames, next_distance, game_score = next_stats
        
        did_win = self.did_win

        prev_reward = self.reward_func(
            frames=prev_frames,
            distance=prev_distance,
            game_score=game_score,
            did_win=False
        )
        next_reward = self.reward_func(
            frames=next_frames,
            distance=next_distance,
            game_score=game_score,
            did_win=did_win
        )

        return next_reward - prev_reward

    # Mario Override
    def update(self, ram, tiles, buttons, output_to_buttons_map) -> bool:
        """
        The main update call for Mario.
        Takes in inputs of surrounding area and feeds through the Neural Network
        
        Return: True if Mario is alive
                False otherwise
        """
        if self.is_alive:
            self._frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            # Sliding down flag pole
            if ram[0x001D] == 3:
                self.did_win = True
                if not self._printed and self.debug:
                    name = 'Mario '
                    name += f'{self.name}' if self.name else ''
                    print(f'{name} won')
                    self._printed = True
                if not self.allow_additional_time:
                    self.is_alive = False
                    return False
            # If we made it further, reset stats
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += 1

            if self.allow_additional_time and self.did_win:
                self.additional_timesteps += 1
            
            if self.allow_additional_time and self.additional_timesteps > self.max_additional_timesteps:
                self.is_alive = False
                return False
            elif not self.did_win and self._frames_since_progress > 60*3:
                self.is_alive = False
                return False            
        else:
            return False

        # Did you fly into a hole?
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        self.set_input_as_array(ram, tiles)

        # Calculate the output
        output = self.choose_action(self.inputs_as_array)
        action = output[0]

        #threshold = np.where(output > 0.5)[0]

        self.model_output = output
        self.buttons_to_press.fill(0)  # Clear

        # !!! ONLY INCLUDES SINGLE ACTION OUTPUTS FOR NOW
        self.buttons_to_press[output_to_buttons_map[action]] = 1

        # Updates the fitness value as well
        self._fitness = self.reward_func(
            distance=self.x_dist,
            frames=self._frames,
            game_score =self.game_score,
            did_win=self.did_win
        )

        return True