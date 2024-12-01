import copy
from typing import Dict, List, Optional, Union
import numpy as np
from torch import argmax
import torch
from torch import nn
from random import sample
from collections import deque, OrderedDict
from config import Config
from mario import Mario
from neural_network import FeedForwardNetwork, get_activation_by_name, sigmoid, tanh, relu, leaky_relu, linear, ActivationFunction
from utils import SMB
from mario import get_num_inputs



# TO DO:
### Ensure DQNAgent overrides the methods in Individual ie. the equivalent choose_action methods




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
        FeedForwardNetwork.__init__(self, layer_nodes, hidden_activation, output_activation)
        nn.Module.__init__(self)
        self.lr = lr
        
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
            net[f'Afn{l}'] = self.to_torch_activation(self.hidden_activation)
            
        # build output layer
        net[f'L{L}'] = nn.Linear(self.layer_nodes[L-1], self.layer_nodes[L])
        net[f'Afn{L}'] =  self.to_torch_activation(self.output_activation)
        
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
            print(f"Layer {l}: W.shape={W.shape}, A_prev.shape={A_prev.shape}, b.shape={b.shape}")
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
        print(f"Input to forward(): {X.shape}")
        X = torch.as_tensor(X, dtype=torch.float32)
        
        if X.shape[0] == self.x_size and X.shape[1] == 1:  # Case where it's transposed
            X = X.view(1, -1)  # Reshape to [1, 80]

        if len(X.shape) == 3 and X.shape[2] == 1:  # Extra last dimension
            X = X.squeeze(-1)  # Remove the last dimension
        
        print(f"After reshaping: {X.shape}")
        out = self.torch_model(X)
        print(f"Output: {out.shape}")
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
                self.params[b_key] = param.detach().numpy()

    

class DQNAgent():

        def __init__(self, 
                     num_actions, 
                     num_states, 
                     network, 
                     ):
            
            self.sync_network_rate = 500  
            self.batch_size = 128        
            self.buffer_size = 1000      
            self.discount_value = 0.99  

            self.epsilon_start = 0.7
            self.epsilon_min = 0.2   
            self.epsilon_decay = 0.9995

            self.learning_rate = 0.01

            self.num_states = num_states
            self.num_actions = num_actions
            self.network = network
            self.network.lr = self.learning_rate
            self.step_counter = 0

            self.epsilon = self.epsilon_start


            # Target network initialisation
            self.target_network = copy.deepcopy(self.network)
            # Replay buffer initialisation
            self.replay_buffer = ReplayBuffer(self.buffer_size)

            self.output_to_keys_map = {
                0: 4,  # U
                1: 5,  # D
                2: 6,  # L
                3: 7,  # R
                4: 8,  # A
                5: 0   # B
            }


        def choose_action(self, state): 
            if np.random.random() < self.epsilon:
                num_indices = np.random.choice([1, 2, 3], p=[0.45, 0.45, 0.1])
                action = np.random.choice(range(self.num_actions), size=num_indices, replace=False)
            else:
                action = self.choose_best_action(state)
            return action


        def choose_best_action(self, state):
            out = self.network.forward(state)
            scaled = torch.sigmoid(out)
            threshold = torch.nonzero(scaled > 0.5, as_tuple=True)[0].cpu().numpy()
            return threshold
        
        def choose_best_action_F(self, state):
            out = self.network.feed_forward(state)
            return np.argmax(out)

        def sync_network(self):
            if self.step_counter % self.sync_network_rate == 0:
                self.target_network.torch_model.load_state_dict(self.network.torch_model.state_dict())

        
        def decay_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
        def learn(self):
            if len(self.replay_buffer) < self.batch_size:
                return
            
          

            samples = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, rewards, dones = zip(*samples)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor([self.output_to_keys_map[a] for a in actions]).unsqueeze(1)

            next_states = torch.FloatTensor(next_states)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)  
            dones = torch.FloatTensor(dones).unsqueeze(1) 

            print(f"states shape: {states.shape}, actions shape: {actions.shape}, next states shape: {next_states.shape}, rewards shape: {rewards.shape}, dones shape : {dones.shape}") 


            predicted_values = self.network.forward(states).gather(1, actions)  # Q(s, a)

            with torch.no_grad():
                next_q_values = self.target_network.forward(next_states).max(1, keepdim=True)[0]  # max Q(s', a')
                target_values = rewards + (self.discount_value * next_q_values * (1 - dones))  # r + γmax Q(s', a')

            # calculate loss between predicted and target Q-values
            loss = self.network.loss_function(predicted_values, target_values)

            # backpropagation via SGD
            self.network.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.torch_model.parameters(), max_norm=1.0)
            self.network.optimizer.step()      

            # Periodically sync target network with the main network
            if self.step_counter % self.sync_network_rate == 0:
                self.sync_network()

            self.step_counter += 1
            self.decay_epsilon()

# This is the interface that allows the DQN implementation to be
# compatible with the rest of the code
class DQNMario(DQNAgent, Mario):
    def __init__(self, 
                 config: Config,
                 name: Optional[str] = None,
                 debug: Optional[bool] = False,):
        self.config = config
        self.hidden_activation = self.config.NeuralNetworkDQN.hidden_node_activation
        self.output_activation = self.config.NeuralNetworkDQN.output_node_activation
        self.network_architecture = self.config.NeuralNetworkDQN.hidden_layer_architecture
        hidden_layer_architecture = self.network_architecture[1:-1]
      
        Mario.__init__(self, config, None, hidden_layer_architecture, self.hidden_activation,
         self.output_activation, np.inf, name, debug)
        model = DQN(self.network_architecture, get_activation_by_name(self.hidden_activation), get_activation_by_name(self.output_activation))
        DQNAgent.__init__(self, self.network_architecture[-1], get_num_inputs(self.config), model)

    def calculate_reward(self, prev_stats, next_stats):
        # For the DQN, reward for a state transition is calculated by taking the difference between the previous and next reward

        prev_reward = self.config.DQN.reward_func(*prev_stats)
        next_reward = self.config.DQN.reward_func(*next_stats)

        return next_reward - prev_reward




    # Override
    def update(self, ram, tiles, buttons, ouput_to_buttons_map) -> bool:
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

        threshold = np.where(output > 0.5)[0]

        self.model_output = output
        self.buttons_to_press.fill(0)  # Clear

        # Set buttons
        for b in threshold:
            self.buttons_to_press[ouput_to_buttons_map[b]] = 1

        # Updates the fitness value as well
        self._fitness = self.config.DQN.reward_func(self.game_score, self.x_dist, self._frames, self.did_win)

        return True