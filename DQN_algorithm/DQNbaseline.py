import copy
from typing import Dict, List, Optional, Union
import numpy as np
from random import sample
from collections import deque, OrderedDict
from config import Config, performance_func
from mario import Mario
from neural_network import FeedForwardNetwork, get_activation_by_name, sigmoid, tanh, relu, leaky_relu, linear, ActivationFunction
from utils import SMB
from mario import get_num_inputs
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gym.spaces import Box, Discrete
from utils import SMB, StaticTileType, EnemyType
import retro
import torch.nn as nn


from DQN_algorithm.DQN import DQN as CustomDQN

from gym.spaces import Space
import numpy as np
 

import gym
import numpy as np
from gym import spaces

  
###
# This is an implentation of a DQN Agent using stable-baselines


def get_torch_activation_by_name(name: str):
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
        'linear': nn.Identity
    }
    return activations.get(name.lower(), None)


    from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

# force the sb3 network to use defined architecture
class ModifiedDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(ModifiedDQN, self).__init__(*args, **kwargs)

        env = kwargs['env']

        net_arch = [9]
        activation_fn = get_torch_activation_by_name('relu')
        output_a = get_torch_activation_by_name('linear')
        
        input_size = env.observation_space.shape[0] 
        layer_nodes = [input_size] + net_arch + [output_a]
        
        self.q_net = CustomDQN(layer_nodes, activation_fn, output_a, self.learning_rate)



## Restricts the DQN's output to a subset of available actions (ie from 9 to 6)
# This is to match the DQN's output layer to the size of the GA's output layer
# Also reduces the input dimentionality to match that of the GA


class InputSpaceReduction(gym.Env):
    def __init__(self, env, config):
        super().__init__()
        
        self.env = env  
        self.mario = None

        self._start_row = config.NeuralNetworkDQN.input_dims[0]
        self._width = config.NeuralNetworkDQN.input_dims[1]
        self._height = config.NeuralNetworkDQN.input_dims[2]
        self._encode_row = config.NeuralNetworkDQN.encode_row

        self.action_space = spaces.Discrete(5)

        self.episode_steps = 0

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self._height * self._width + (self._height if self._encode_row else 0),), dtype=np.float32
        )

        
    def get_ram(self):
        return self.env.get_ram()
        
    def reset(self):
        obs = self.env.reset()  
        self.episode_frames = 0
        return self._observation(obs)
    
    def step(self, action):

        self.episode_steps += 1

        output_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }

        action = output_to_keys_map[action]

        # size of the action space
        one_hot_v = np.zeros(9)
        one_hot_v[action] = 1

        obs, reward, done, _, info = self.env.step(one_hot_v)  

   
        self.mario.update(self.get_ram(), SMB.get_tiles(self.get_ram()))

        if self.mario.did_win:
            print("DQN WON")

        #override env reward with the fitness func
        reward = self.mario.calculate_fitness()

        if not self.mario.is_alive:
            done = True
            self.mario.reset()


        if done:
            self.episode_steps = 0


        return self._observation(obs), reward, done, info  

    
    def _observation(self, obs):
        ram = self.env.get_ram()  

        mario_row, mario_col = SMB.get_mario_row_col(ram)
        tiles = SMB.get_tiles(ram)
        arr = []
        
        for row in range(self._start_row, self._start_row + self._height):
            for col in range(mario_col, mario_col + self._width):
                try:
                    t = tiles[(row, col)]
                    if isinstance(t, StaticTileType):
                        if t.value == 0:
                            arr.append(0)
                        else:
                            arr.append(1)
                    elif isinstance(t, EnemyType):
                        arr.append(-1)
                    else:
                        raise Exception("This should never happen")
                except:
                    t = StaticTileType(0x00)
                    arr.append(0) # Empty  
        
        input_array = np.array(arr).reshape((-1, 1))
        
        if self._encode_row:
            row = mario_row - self._start_row
            one_hot = np.zeros((self._height, 1))
            if 0 <= row < self._height:
                one_hot[row, 0] = 1
            input_array = np.vstack([input_array, one_hot.reshape((-1, 1))])

        return input_array.flatten()



class DQNCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, data_queue, mario, config, verbose=1):
        super(DQNCallback, self).__init__(verbose)
        self.data_queue = data_queue
        self.mario = mario
        self.config = config
        self.is_training = False

        self.max_distance = 0
        self.max_fitness = 0
        self.episode = 0
        self.episode_rewards = 0
        self.episode_steps = 0


        self.max_episodes = self.config.DQN.total_episodes


    def _on_training_start(self) -> None:
    
        self.is_training = True


    def _on_step(self) -> bool:

        done = False
        collected_rewards = 0
        collected_steps = 0

        self.episode_steps += 1
        self.episode_rewards += self.locals['rewards'].sum()


        if self.locals['dones'].any():
            print("EPISODE: ", self.episode)
            self.episode += 1
            collected_rewards = self.episode_rewards
            collected_steps = self.episode_steps
            self.episode_rewards = 0 
            self.episode_steps = 0 
            done = True

            if self.episode >= self.max_episodes:
                print(f"Stopping training DQN after {self.episode} episodes.")
                return False  # Stops training
            

        if self.mario.farthest_x > self.max_distance:
            self.max_distance = self.mario.farthest_x
        if self.mario.fitness >  self.max_fitness:
            self.max_fitness = self.mario.fitness


        data = {
            'max_fitness':  self.max_fitness,
            'max_distance': self.max_distance,
            'total_steps': self.num_timesteps,
            'episode_num': self.episode,
            'episode_rewards': collected_rewards,
            'episode_steps': collected_steps,
            'episode_distance': self.mario.farthest_x,
            'done': done,
        }

        self.data_queue.put(data)
        return True

 
    def _on_training_end(self) -> None:
        self.is_training = False
        self.model.save(f'{self.config.Statistics.dqn_save_dir}/{self.config.Statistics.dqn_model_name}')

class DQNMario(Mario):
    def __init__(self, 
                 config: Config,
                 env,
                 name: Optional[str] = "DQNAgent",
                 debug: Optional[bool] = False):
        self.config = config
        self.reward_func = performance_func
        nn_params = self.config.NeuralNetworkDQN
        
        Mario.__init__(self, config, None, nn_params.hidden_layer_architecture, nn_params.hidden_node_activation,
         nn_params.output_node_activation, nn_params.encode_row, np.inf, name, debug)

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
        self.train_freq = self.config.DQN.train_freq

        # specifies the model architecture for the DQN
        policy_kwargs = dict(activation_fn=get_torch_activation_by_name(self.hidden_activation), net_arch=self.hidden_layer_architecture)

        self.model = DQN('MlpPolicy', 
                    env=env, 
                    gamma=self.discount_value, 
                    learning_rate=self.learning_rate,  
                    buffer_size=self.buffer_size,
                    exploration_fraction=self.epsilon_decay, 
                    exploration_final_eps=self.epsilon_min, 
                    exploration_initial_eps=self.epsilon_start,
                    train_freq=self.train_freq,
                    batch_size=self.batch_size,
                    target_update_interval= self.sync_network_rate,
                    verbose=1,
                    tensorboard_log= None,
                    policy_kwargs = policy_kwargs,
                    device="cpu"
                    )
        
        
    def reset(self):
        self._frames = 0
        self.is_alive = True
        self.farthest_x = 0
        self.x_dist = None
        self.game_score = None
        self.did_win = False

    # Mario Override
    def update(self, ram, tiles) -> bool:
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

        # Updates the fitness value as well
        self._fitness = self.reward_func(
            distance=self.x_dist,
            frames=self._frames,
            game_score =self.game_score,
            did_win=self.did_win
        )

        return True