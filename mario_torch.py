import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Union, Dict, Any, List
import random
import os
import csv

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population

from config import Config, performance_func
from utils import SMB, StaticTileType, EnemyType
from neural_network import get_activation_by_name  

# Simulates multioutput by mapping specific NN outputs to key combinations
output_to_keys_map = {
    0: [],        # No operation
    1: [7],         # Right
    2: [7, 8],      # Right + A
    3: [7, 0],      # Right + B
    4: [7, 8, 0],   # Right + A + B
    5: [8],         # A
    6: [6],         # Left
}

def get_torch_activation(activation_name: str):
    name = activation_name.lower()
    if name == 'relu':
        return nn.ReLU
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'sigmoid':
        return nn.Sigmoid
    elif name == 'leaky_relu':
        return nn.LeakyReLU
    elif name == 'linear':
        return nn.Identity
    else:
        return nn.ReLU

class SequentialModel(nn.Module):
    def __init__(self, layer_sizes: List[int], hidden_activation: str, output_activation: str):
        super(SequentialModel, self).__init__()

        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2: 
                self.layers.append(get_torch_activation(hidden_activation)())
            else: 
                self.layers.append(get_torch_activation(output_activation)())
        
        self.model = nn.Sequential(*self.layers)

        self.layers = [layer for layer in self.layers if hasattr(layer, 'weight')]

    def set_training_mode(self, mode: bool):
        if mode:
            self.train()
        else:
            self.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str, iteration: int, distance: int, algorithm: str, input_dims: Tuple[int, ...], encode_row: bool, state_dict=None):
        if not state_dict:
            state_dict = self.model.state_dict()
        torch.save({
            'iterations': iteration,
            'distance': distance,
            'encode_row': encode_row,
            'input_dims': input_dims,
            'algorithm': algorithm,
            'state_dict': state_dict,
            'layer_sizes': self.layer_sizes,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation
        }, path)

    @classmethod
    def load(cls, path: str):
        details = torch.load(path, weights_only=False)

        model = cls(details['layer_sizes'], details['hidden_activation'], details['output_activation'])
        

        updated_state_dict = {}
        for key, value in details['state_dict'].items():
            if '.' in key:
                suffix = '.'.join(key.split('.')[-2:])  
            else:
                suffix = key
            
            updated_key = f'model.{suffix}'
            updated_state_dict[updated_key] = value
        
        # Load the updated state_dict into the model
        model.load_state_dict(updated_state_dict)
        return model


# Reimplementation of mario using PyTorch instead of FeedForwardNetwork
class MarioTorch(Individual):
    def __init__(self,
                 chromosome: Optional[Dict[str, np.ndarray]] = None,
                 hidden_layer_architecture: List[int] = [12, 9],
                 hidden_activation: Optional[str] = 'relu',
                 output_activation: Optional[str] = 'sigmoid',
                 encode_row: Optional[bool] = True,
                 lifespan: Union[int, float] = np.inf,
                 name: Optional[str] = None,
                 debug: Optional[bool] = False,
                 frame_skip: int = 4,
                 input_dims: Tuple[int, ...] = (4, 7, 10),
                 allow_additional_time_for_flagpole: Optional[bool] = True,
                 ):
        
        self.fitness_func = performance_func
        self.lifespan = lifespan
        self.name = name
        self.debug = debug

        self.frame_skip = frame_skip

        self._fitness = 0
        self._frames_since_progress = 0
        self._frames = 0

        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.encode_row = encode_row
        self.input_dims = input_dims

        self.start_row, self.viz_width, self.viz_height = input_dims

        if self.encode_row:
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height
        
        self.inputs_as_array = np.zeros((num_inputs, 1))

        self.network_architecture = [num_inputs]
        self.network_architecture.extend(self.hidden_layer_architecture)
        self.network_architecture.append(len(output_to_keys_map))

        # Create a PyTorch model
        self.model = SequentialModel(
            layer_sizes=self.network_architecture,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation
        )
        
        # If chromosome is provided, load weights
        if chromosome:
            self._load_chromosome(chromosome)
        
        self.is_alive = True
        self.x_dist = None
        self.game_score = None
        self.did_win = False
        self.allow_additional_time = allow_additional_time_for_flagpole
        self.additional_timesteps = 0
        self.max_additional_timesteps = int(60 * 2.5)
        self._printed = False
        self.farthest_x = 0

  
    @property
    def fitness(self):
        return self._fitness

    @property
    def chromosome(self):
        return self._extract_chromosome()

    def decode_chromosome(self):
        pass

    def encode_chromosome(self):
        pass

    def _extract_chromosome(self) -> Dict[str, np.ndarray]:
        # Extract PyTorch parameters into a dictionary of NumPy arrays
        chromosome = {}
        idx = 0
        for i, layer in enumerate(self.model.layers):
            w_name = f"W{i+1}"
            b_name = f"b{i+1}"
            weights = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy()
            chromosome[w_name] = weights
            chromosome[b_name] = bias
        return chromosome

    def _load_chromosome(self, chromosome: Dict[str, np.ndarray]):
        # Load the chromosome (weights & biases) into the PyTorch model
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers):
                w_name = f"W{i+1}"
                b_name = f"b{i+1}"
                layer.weight.data = torch.from_numpy(chromosome[w_name]).float()
                layer.bias.data = torch.from_numpy(chromosome[b_name]).float()

                #if torch.cuda.is_available():
                    #layer.weight.data = layer.weight.data.to('cuda')
                    #layer.bias.data = layer.bias.data.to('cuda')

    
    def to_cuda(self):
        if torch.cuda.is_available():
            self.model.to('cuda')

    def calculate_fitness(self):
        frames = self._frames
        distance = self.x_dist
        score = self.game_score
        did_win = self.did_win

        self._fitness = self.fitness_func(
            distance=distance,
            frames=frames,
            game_score=score,
            did_win=did_win
        )
        return self._fitness

    def reset(self):
        # For DQN functionality
        pass

    def get_action(self, obs):
        x = torch.from_numpy(obs).float()
        if next(self.model.parameters()).is_cuda:
            x = x.to('cuda')
        output = self.model.forward(x)
        output = output.detach().cpu().numpy().flatten()
        action_index = np.argmax(output)
        return action_index


    def update(self, ram, tiles) -> bool:
        if self.is_alive:
            self._frames += self.frame_skip
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
            
            # Check progress
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += self.frame_skip

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

        # Check if Mario fell into a hole
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        return True


def save_mario(population_folder: str, individual_name: str, mario: MarioTorch, generation, distance) -> None:
    # Make population folder if it doesnt exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    mario.model.save(os.path.join(population_folder, f'{individual_name}.pt'), generation, distance, algorithm='GA', input_dims=mario.input_dims, encode_row=mario.encode_row)
    
def load_mario(population_folder: str, individual_name: str, config: Optional[Config] = None) -> MarioTorch:
    if not os.path.exists(os.path.join(population_folder, individual_name)):
        raise Exception(f'{individual_name} not found inside {population_folder}')

    if not config:
        settings_path = os.path.join(population_folder, 'settings.config')
        if not os.path.exists(settings_path):
            raise Exception(f'settings.config not found under {population_folder}')
        config = Config(settings_path)
    
    chromosome: Dict[str, np.ndarray] = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        if fname.endswith('.npy'):
            param = fname.rsplit('.npy', 1)[0]
            chromosome[param] = np.load(os.path.join(population_folder, individual_name, fname))
        
    mario = MarioTorch(config, chromosome=chromosome)
    return mario

def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))
    return (mean, median, std, _min, _max)

def save_stats(population: Population, fname: str):
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    frames = [individual._frames for individual in population.individuals]
    max_distance = [individual.farthest_x for individual in population.individuals]
    fitness = [individual.fitness for individual in population.individuals]
    wins = [sum([individual.did_win for individual in population.individuals])]

    write_header = True
    if os.path.exists(fname):
        write_header = False

    trackers = [('frames', frames),
                ('distance', max_distance),
                ('fitness', fitness),
                ('wins', wins)]

    stats = ['mean', 'median', 'std', 'min', 'max']
    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(fname, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',')
        if write_header:
            writer.writeheader()

        row = {}
        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat

        writer.writerow(row)

def load_stats(path_to_stats: str, normalize: Optional[bool] = False):
    data = {}

    with open(path_to_stats, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        trackers_stats = [f.split('_') for f in fieldnames]
        trackers = set(ts[0] for ts in trackers_stats)
        stats_names = set(ts[1] for ts in trackers_stats)

        for tracker, stat_name in trackers_stats:
            if tracker not in data:
                data[tracker] = {}
            if stat_name not in data[tracker]:
                data[tracker][stat_name] = []

        for line in reader:
            for tracker in trackers:
                for stat_name in stats_names:
                    value = float(line['{}_{}'.format(tracker, stat_name)])
                    data[tracker][stat_name].append(value)
        
    if normalize:
        factors = {}
        for tracker in trackers:
            factors[tracker] = {}
            for stat_name in stats_names:
                factors[tracker][stat_name] = 1.0

        for tracker in trackers:
            for stat_name in stats_names:
                max_val = max([abs(d) for d in data[tracker][stat_name]])
                if max_val == 0:
                    max_val = 1
                factors[tracker][stat_name] = float(max_val)

        for tracker in trackers:
            for stat_name in stats_names:
                factor = factors[tracker][stat_name]
                d = data[tracker][stat_name]
                data[tracker][stat_name] = [val / factor for val in d]

    return data

def get_num_inputs(config: Config) -> int:
    _, viz_width, viz_height = config.NeuralNetworkGA.input_dims
    if config.NeuralNetworkGA.encode_row:
        num_inputs = viz_width * viz_height + viz_height
    else:
        num_inputs = viz_width * viz_height
    return num_inputs

def get_num_trainable_parameters(config: Config) -> int:
    num_inputs = get_num_inputs(config)
    hidden_layers = config.NeuralNetworkGA.hidden_layer_architecture
    num_outputs = len(output_to_keys_map.keys())

    layers = [num_inputs] + hidden_layers + [num_outputs]
    num_params = 0
    for i in range(len(layers)-1):
        L = layers[i]
        L_next = layers[i+1]
        num_params += L*L_next + L_next

    return num_params

def state_dict_to_chromosome(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    chromosome = {}
    layer_index = 1 
    
    for key, value in state_dict.items():
        if "weight" in key:
            chromosome[f"W{layer_index}"] = value.cpu().numpy()  #
        elif "bias" in key:
            chromosome[f"b{layer_index}"] = value.cpu().numpy() 
            layer_index += 1  

    return chromosome