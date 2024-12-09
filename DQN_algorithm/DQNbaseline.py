import copy
from typing import Dict, List, Optional, Union
import numpy as np
from random import sample
from collections import deque, OrderedDict
from config import Config, performance_func
from mario_torch import MarioTorch as Mario
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
import torch
import time
import random

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



class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class EpsilonDecayScheduler:
    def __init__(self, initial_epsilon, final_epsilon, decay_fraction, total_episodes):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_episodes = int(total_episodes * decay_fraction)

    def get_epsilon(self, current_episode):
        return max(self.initial_epsilon - current_episode * (self.initial_epsilon - self.final_epsilon) / self.decay_episodes,
                     self.final_epsilon)


## Restricts the DQN's output to a subset of available actions (ie from 9 to 6)
# This is to match the DQN's output layer to the size of the GA's output layer
# Also reduces the input dimentionality to match that of the GA


class InputSpaceReduction(gym.Env):
    def __init__(self, env, input_dims, encode_row):
        super().__init__()
        
        self.env = env  
        self.mario = None

        self._start_row = input_dims[0]
        self._width = input_dims[1]
        self._height = input_dims[2]
        self._encode_row = encode_row

        self.output_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }

        self.episode_steps = 0

        self.input_size = self._height * self._width + (self._height if self._encode_row else 0),
        self.output_size = 6

        self.action_space = spaces.Discrete(self.output_size)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.input_size), dtype=np.float32
        )
        
    def get_ram(self):
        return self.env.get_ram()
        
    def reset(self):

        self.mario.reset()
        self.episode_steps = 0

        obs = self.env.reset()  
        self.mario.update(self.get_ram(), SMB.get_tiles(self.get_ram()))
        
        return self._observation(obs)
    
    def step(self, action):

        self.episode_steps += 1

        action = self.output_to_keys_map[action]

        # size of the action space
        one_hot_v = np.zeros(9)
        one_hot_v[action] = 1

        obs, reward, done, _, info = self.env.step(one_hot_v) 

        self.mario.update(self.get_ram(), SMB.get_tiles(self.get_ram()))
     

        if self.mario.did_win:
            print("WE HAVE A WINNER")

        #override env reward with the fitness func
        reward = self.mario.calculate_fitness()

        if not self.mario.is_alive:
            done = True

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
        self.recent_distance = 0
        self.action_counts = [0] * 6
        self.recent_reward = 0

        self.max_episodes = self.config.DQN.total_episodes

        self.epsilon_scheduler = EpsilonDecayScheduler(config.DQN.epsilon_start, config.DQN.epsilon_min, config.DQN.decay_fraction, self.max_episodes)


    def _on_training_start(self) -> None:
    
        self.is_training = True


    def _on_step(self) -> bool:

        done = True if self.locals['dones'].any() else False

        actions = self.locals['actions']
        for action in actions:
            self.action_counts[action] += 1
        

        self.episode_steps += 1
        self.episode_rewards += self.locals['rewards'].sum()
        self.recent_reward = self.locals['rewards'].sum()


        if self.mario.farthest_x > self.max_distance:
            self.max_distance = self.mario.farthest_x
        if self.mario.fitness >  self.max_fitness:
            self.max_fitness = self.mario.fitness

        if done:
            # manually update epsilon
            self.model.exploration_rate = self.epsilon_scheduler.get_epsilon(self.episode)
            print("epsilon: ", self.model.exploration_rate)

            data = {
                'fitness': self.recent_reward,
                'max_fitness':  self.max_fitness,
                'max_distance': self.max_distance,
                'episode_num': self.episode,
                'episode_rewards': self.episode_rewards,
                'episode_steps': self.episode_steps,
                'episode_distance': self.recent_distance,
                'action_counts': self.action_counts,
                'epsilon': self.model.exploration_rate,
                'done': done,
            }
            self.data_queue.put(data)
        
            self.episode += 1
            self.episode_rewards = 0 
            self.episode_steps = 0 
            self.recent_distance = 0
            self.recent_reward = 0
            print("EPISODE: ", self.episode)

            if self.episode % 10 == 0:
                policy_nn = self.model.policy
                self.save_model("CHECKPOINT")


            if self.episode >= self.max_episodes:
                return False  # Stops training

        self.recent_distance = self.mario.farthest_x

        return True

 
    def _on_training_end(self) -> None:
        print(f"Stopping training DQN after {self.episode} episodes.")
        self.is_training = False
        self.save_model("FINAL")
    
    def save_model(self, title):
        if self.model.env is None:
            return
        
        self.model.save(f'{self.config.Statistics.dqn_save_dir}/{self.config.Statistics.dqn_model_name}_{title}')
        layer_sizes = [self.model.env.observation_space.shape[0]] + [self.config.NeuralNetworkDQN.hidden_layer_architecture] + [self.model.env.action_space.n]
        torch.save({
        'state_dict': self.model.policy.state_dict(),
        'layer_sizes': layer_sizes,
        'hidden_activation': self.config.NeuralNetworkDQN.hidden_node_activation,
        'output_activation': self.config.NeuralNetworkDQN.output_node_activation,
        }, self.config.Statistics.dqn_save_dir + f'/{self.config.Statistics.dqn_model_name}_{title}.pt')


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


        ## Parameter initialisation
        self.learning_rate = self.config.DQN.learning_rate
        self.buffer_size = self.config.DQN.buffer_size
        self.sync_network_rate = self.config.DQN.sync_network_rate
        self.batch_size = self.config.DQN.batch_size
        self.discount_value = self.config.DQN.discount_value
        self.epsilon_start = self.config.DQN.epsilon_start
        self.epsilon_min = self.config.DQN.epsilon_min
        self.decay_fraction = self.config.DQN.decay_fraction
        self.train_freq = self.config.DQN.train_freq

        # specifies the model architecture for the DQN
        policy_kwargs = dict(activation_fn=get_torch_activation_by_name(self.hidden_activation), net_arch=self.hidden_layer_architecture)

        #device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = DQN('MlpPolicy', 
                    env=env, 
                    gamma=self.discount_value, 
                    learning_rate=self.learning_rate,  
                    buffer_size=self.buffer_size,
                    exploration_fraction=self.decay_fraction,
                    exploration_final_eps=self.epsilon_min, 
                    exploration_initial_eps=self.epsilon_start,
                    train_freq=self.train_freq,
                    batch_size=self.batch_size,
                    target_update_interval= self.sync_network_rate,
                    verbose=1,
                    tensorboard_log= None,
                    policy_kwargs = policy_kwargs,
                    device="cuda"
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

        # Updates the fitness value as well
        self._fitness = self.reward_func(
            distance=self.x_dist,
            frames=self._frames,
            game_score =self.game_score,
            did_win=self.did_win
        )

        return True