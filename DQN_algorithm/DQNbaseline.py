import copy
import random
from typing import Dict, List, Optional, Union
import numpy as np
from random import sample
from collections import deque, OrderedDict
from config import Config, performance_func
from mario_torch import MarioTorch as Mario
from mario_torch import output_to_keys_map, SequentialModel, get_torch_activation
from utils import SMB
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gym.spaces import Box, Discrete
from utils import SMB, StaticTileType, EnemyType
import torch.nn as nn
import torch
import os
import shutil

from gym.spaces import Space
import numpy as np
 

import gym
import numpy as np
from gym import spaces

###
# This is an implementation of a DQN Agent using stable-baselines

def clear_dir(target):
    if os.path.exists(target):
        shutil.rmtree(target) 
    os.makedirs(target, exist_ok=True)



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
    def __init__(self, env, input_dims, encode_row, skip=4):
        super().__init__()
        
        self.env = env  
        self.mario = None

        self._start_row = input_dims[0]
        self._width = input_dims[1]
        self._height = input_dims[2]
        self._encode_row = encode_row
        self._skip = skip

        self.output_to_keys_map = output_to_keys_map

        self.episode_steps = 0

        self.input_size = self._height * self._width + (self._height if self._encode_row else 0),
        self.output_size = len(self.output_to_keys_map.keys())

        self.action_space_original = self.env.action_space
        self.action_space = spaces.Discrete(self.output_size)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.input_size), dtype=np.float32
        )
        
    def get_ram(self):
        return self.env.get_ram()
        
    def reset(self):

        if self.mario:
            self.mario.reset()
        self.episode_steps = 0

        obs = self.env.reset()  

        if self.mario:
            self.mario.update(self.get_ram(), SMB.get_tiles(self.get_ram()))
            self.mario.calculate_fitness()
    
        return self._observation(obs)
    
    def step(self, action):

        self.episode_steps += 1
        
        action_indices = self.output_to_keys_map[action]

        # Size of the (original) action space
        multi_hot_action = np.zeros(self.action_space_original.n)

        for action_index in action_indices:
            multi_hot_action[action_index] = 1

        for _ in range(self._skip):
            obs, reward, done, _, info = self.env.step(multi_hot_action) 
            if done:
                break

        if self.mario:
            self.mario.update(self.get_ram(), SMB.get_tiles(self.get_ram()))

        #override env reward with the delta of fitness func
        if self.mario:
            prior_fitness = self.mario.fitness
            reward = self.mario.calculate_fitness() - prior_fitness


<<<<<<< HEAD

=======
>>>>>>> 56b2ed9 (.)
        if self.mario and not self.mario.is_alive:
            done = True

        return self._observation(obs), reward, done, info  

    
    def _observation(self, obs):

        if not self.mario:
            return obs

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
    def __init__(self, data_queue, mario, config, verbose=1, episode_start=0):
        super(DQNCallback, self).__init__(verbose)
        self.data_queue = data_queue
        self.mario = mario
        self.config = config
        self.is_training = False

        self.encode_row = config.NeuralNetworkDQN.encode_row
        self.input_dims = config.NeuralNetworkDQN.input_dims
        
        self.layer_sizes = self.mario.network_architecture

        self.max_distance = 0
        self.max_fitness = 0
        self.episode = episode_start
        self.episode_steps = 0
        self.episode_rewards = 0
        self.recent_distance = 0
        self.action_counts = [0] * len(output_to_keys_map)

        self.best_model_state_dict = None
        self.best_model_distance = 0

        self.max_episodes = self.config.DQN.total_episodes

        self.epsilon_scheduler = EpsilonDecayScheduler(config.DQN.epsilon_start, config.DQN.epsilon_min, config.DQN.decay_fraction, self.max_episodes)


    def _on_training_start(self) -> None:
    
        self.is_training = True

    def _on_step(self) -> bool:

        done = True if self.locals['dones'].any() else False

        actions = self.locals['actions']
        for action in actions:
            self.action_counts[action] += 1

        rewards = self.local['rewards']
        self.episode_rewards += sum(rewards)
        

        self.episode_steps += 1

        if self.mario.farthest_x > self.max_distance:
            self.max_distance = self.mario.farthest_x
        if self.mario.fitness >  self.max_fitness:
            self.max_fitness = self.mario.fitness
            # also save these new weights as new best
            self.best_model_distance = self.mario.farthest_x
            self.best_model_state_dict = copy.deepcopy(self.model.policy.state_dict())

        if done:
            # manually update epsilon
            self.model.exploration_rate = self.epsilon_scheduler.get_epsilon(self.episode)

            if self.episode % self.config.Statistics.dqn_checkpoint_interval == 0 and self.episode > 0:
                self.save_current_model(self.episode, self.recent_distance, postfix="_CHECKPOINT")
                self.save_best_model(self.episode)

            data = {
                'fitness': self.recent_fitness,
                'max_fitness':  self.max_fitness,
                'max_distance': self.max_distance,
                'episode_num': self.episode,
                'episode_steps': self.episode_steps,
                'episode_distance': self.recent_distance,
                'action_counts': self.action_counts,
                'epsilon': self.model.exploration_rate,
                'episode_rewards': self.episode_rewards
            }
            self.data_queue.put(data)


            if self.episode >= self.max_episodes:
                return False  # Stops training
        
            self.episode += 1
            self.episode_steps = 0 
            self.episode_rewards = 0
            self.recent_distance = 0
            self.recent_fitness = 0
            self.action_counts = [0] * len(output_to_keys_map)

            print("EPISODE: ", self.episode)

        self.recent_distance = self.mario.farthest_x
        self.recent_fitness = self.mario.fitness

        return True

 
    def _on_training_end(self) -> None:
        print(f"Stopping training DQN after {self.episode} episodes.")
        self.is_training = False
        self.save_current_model(self.episode, self.recent_distance, postfix="_FINAL")
        self.save_best_model(self.episode)
    
    def save_current_model(self, episode, distance, postfix=""):
        # Saving the model at a given episode
        if self.model.env is None:
            return
        fitness = int(max(0, min(self.recent_fitness, 99999999)))
        save_dir = os.path.join(self.config.Statistics.model_save_dir, f'DQN/EPS{self.episode}{postfix}')
        save_file = f"{self.config.Statistics.dqn_model_name}_fitness{fitness}.pt"
        save_path = os.path.join(save_dir, save_file)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model = SequentialModel(self.layer_sizes, self.config.NeuralNetworkDQN.hidden_node_activation, self.config.NeuralNetworkDQN.output_node_activation)
        
        model.save(save_path, episode, distance, "DQN", self.input_dims, self.encode_row, state_dict=self.model.policy.state_dict())

    def save_best_model(self, episode):
        # Saving the overall best model
        if self.model.env is None:
            return
        
        fitness = int(max(0, min(self.max_fitness, 99999999)))
        save_dir = os.path.join(self.config.Statistics.model_save_dir, f'DQN/BEST_OVERALL')
        save_file = f"{self.config.Statistics.dqn_model_name}_fitness{fitness}.pt"
        save_path = os.path.join(save_dir, save_file)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        clear_dir(save_dir)

        model = SequentialModel(self.layer_sizes, self.config.NeuralNetworkDQN.hidden_node_activation, self.config.NeuralNetworkDQN.output_node_activation)


        model.save(save_path, episode, self.best_model_distance, "DQN", self.input_dims, self.encode_row, state_dict=self.best_model_state_dict)



class DQNMario(Mario):
    def __init__(self, 
                 config: Config,
                 env,
                 name: Optional[str] = "DQNAgent",
                 debug: Optional[bool] = False):
        self.config = config
        self.reward_func = performance_func
        nn_params = self.config.NeuralNetworkDQN
        
        Mario.__init__(self, None, nn_params.hidden_layer_architecture, nn_params.hidden_node_activation,
         nn_params.output_node_activation, nn_params.encode_row, np.inf, name, debug, self.config.Environment.frame_skip, nn_params.input_dims, self.config.Misc.allow_additional_time_for_flagpole)


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

        self.model = self.create_model(self.hidden_layer_architecture, self.hidden_activation, env)

        
    def reset(self):
        self._frames = 0
        self.is_alive = True
        self.farthest_x = 0
        self.x_dist = None
        self.game_score = None
        self.did_win = False

    def create_model(self, hidden_layer_architecture, hidden_activation, env):
        policy_kwargs = dict(
            net_arch=list(hidden_layer_architecture),
            activation_fn=get_torch_activation(hidden_activation)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return DQN(DQNPolicy, 
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
                    device=device
                    )


    def load_saved_model(self, save_path, env):
        '''Loads .pt file from save_path and returns the iterations and distance of the saved model'''

        assert checkpoint['layer_sizes'] == self.network_architecture, "Failed to load model with a differing architecture than what is specified in the .config"

        checkpoint = torch.load(save_path)
        state_dict = checkpoint['state_dict']
        layer_sizes = checkpoint['layer_sizes']
        hidden_activation = checkpoint['hidden_activation']

        self.model = self.create_model(layer_sizes[1:-1], hidden_activation, env)
        self.model.policy.load_state_dict(state_dict)

        return checkpoint['iterations'], checkpoint['distance']
