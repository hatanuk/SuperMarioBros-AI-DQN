import re
import retro
# Removed PyQt5 imports
# from PyQt5 import QtGui, QtWidgets
# from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
# from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
# from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

from typing import Tuple, List, Optional
import random
import sys
import math
import numpy as np

from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config
from nn_viz import NeuralNetworkViz
from mario import Mario, save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation

from DQN_algorithm.DQNbaseline import ActionDiscretizer, DQNCallback, DQNMario

from smb_ai import draw_border, parse_args

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import multiprocessing
import queue
import time  # Added time module for sleep

import atexit

from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, ga_writer, dqn_writer, config):
        self.ga_writer = ga_writer
        self.dqn_writer = dqn_writer
        self.config = config
        self.step_counter = 0

    def log_ga_metrics(self, max_fitness, max_distance, generation, total_steps):
        if total_steps % self.config.Statistics.log_interval == 0:
            self.ga_writer.add_scalar('max_fitness', max_fitness, total_steps)
            self.ga_writer.add_scalar('max_distance', max_distance, total_steps)
            self.ga_writer.add_scalar('total_steps', total_steps, total_steps)

    def log_dqn_metrics(self, max_fitness, max_distance, episode_reward, episode_num, total_steps):
        if total_steps % self.config.Statistics.log_interval == 0:
            self.dqn_writer.add_scalar('max_fitness', max_fitness, total_steps)
            self.dqn_writer.add_scalar('max_distance', max_distance, total_steps)
            self.dqn_writer.add_scalar('total_steps', total_steps, total_steps)
        
        self.dqn_writer.add_scalar('episode_reward', episode_reward, episode_num)

    def increment_step(self):
        self.step_counter += 1


def run_ga_agent(config, data_queue):
    # Initialize environment
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Misc.level}')

    # Initialize population and agent
    individuals = _initialize_population(config)
    population = Population(individuals)
    mario_GA = population.individuals[0]
    best_fitness_GA = 0.0
    max_distance_GA = 0
    current_generation = 0
    _current_individual = 0
    total_steps_GA = 0

    # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
    keys = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)

    # Mapping from output to keys
    ouput_to_keys_map = {
        0: 4,  # U
        1: 5,  # D
        2: 6,  # L
        3: 7,  # R
        4: 8,  # A
        5: 0   # B
    }

    # Determine the size of the next generation
    if config.Selection.selection_type == 'plus':
        _next_gen_size = config.Selection.num_parents + config.Selection.num_offspring
    elif config.Selection.selection_type == 'comma':
        _next_gen_size = config.Selection.num_offspring
    else:
        raise Exception(f'Unknown selection type: {config.Selection.selection_type}')

    # Reset environment
    env.reset()

    while True:
        # Update agent
        ram = env.get_ram()
        tiles = SMB.get_tiles(ram)
        enemies = SMB.get_enemy_locations(ram)

        mario_GA.update(ram, tiles, keys, ouput_to_keys_map)

        # Take a step in the environment
        ret = env.step(mario_GA.buttons_to_press)
        total_steps_GA += 1

        # Get new ram information
        ram = env.get_ram()
        tiles = SMB.get_tiles(ram)
        enemies = SMB.get_enemy_locations(ram)

        # Update the stats
        if mario_GA.is_alive:
            # Check for new farthest distance
            if mario_GA.farthest_x > max_distance_GA:
                max_distance_GA = mario_GA.farthest_x
        else:
            # Once agent dies,
            # Check for new best fitness
            mario_GA.calculate_fitness()
            if mario_GA.fitness > best_fitness_GA:
                best_fitness_GA = mario_GA.fitness

            # Handle end of individual or generation
            _current_individual += 1

            # Checks whether it's time to transition to a new generation
            if _current_individual >= len(population.individuals):
                # Next generation
                current_generation += 1
                _current_individual = 0

                # GA selection
                population.individuals = elitism_selection(population, config.Selection.num_parents)
                random.shuffle(population.individuals)

                next_pop = []

                # Decrement lifespan and carry over individuals
                if config.Selection.selection_type == 'plus':
                    # Decrement lifespan
                    for individual in population.individuals:
                        individual.lifespan -= 1

                    next_pop = individual_to_agent(population.individuals, config)

                while len(next_pop) < _next_gen_size:
                    # Perform crossover and mutation to generate new offspring
                    selection = config.Crossover.crossover_selection
                    if selection == 'tournament':
                        p1, p2 = tournament_selection(population, 2, config.Crossover.tournament_size)
                    elif selection == 'roulette':
                        p1, p2 = roulette_wheel_selection(population, 2)
                    else:
                        raise Exception(f'Unknown crossover selection: {selection}')

                    # Perform SBX crossover and Gaussian mutation
                    c1_params, c2_params = _crossover_and_mutate(p1, p2, config, current_generation)

                    c1 = Mario(config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.encode_row, p1.lifespan)
                    c2 = Mario(config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.encode_row, p2.lifespan)

                    # Adds two children from the crossover to the next population
                    next_pop.extend([c1, c2])

                # Finally, update the individuals to this new population
                population.individuals = next_pop

            # Reset the environment
            env.reset()
            mario_GA = population.individuals[_current_individual]

        # Prepare data to send back
        data = {
            'max_fitness': best_fitness_GA,
            'max_distance': max_distance_GA,
            'total_steps': total_steps_GA,
            'current_generation': current_generation,
            'current_individual': _current_individual,
        }

        # Send data to main process
        data_queue.put(data)

def run_dqn_agent(config, data_queue, dqn_model):
    # Initialize environment
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Misc.level}')
    env = ActionDiscretizer(env)
    env = DummyVecEnv([lambda: env])

    # Initialize DQN agent
    mario_DQN = DQNMario(config, env)

    if dqn_model:
        try:
            model = DQN.load(dqn_model, env=env)
            mario_DQN.model = model
        except Exception as e:
            raise Exception(f'Failed to find model at {dqn_model}') from e
        # Add an inference loop here later
    else:
        callback = DQNCallback(data_queue, mario_DQN, config, verbose=1)
        mario_DQN.model.learn(total_timesteps=config.DQN.total_timesteps, callback=callback)

def _initialize_population(config):
    individuals: List[Individual] = []
    num_parents = config.Selection.num_parents

    hidden_layer_architecture = config.NeuralNetworkGA.hidden_layer_architecture
    hidden_activation = config.NeuralNetworkGA.hidden_node_activation
    output_activation = config.NeuralNetworkGA.output_node_activation
    encode_row = config.NeuralNetworkGA.encode_row
    lifespan = config.Selection.lifespan

    for _ in range(num_parents):
        individual = Mario(config, None, hidden_layer_architecture, hidden_activation, output_activation, encode_row, lifespan)
        individuals.append(individual)
    return individuals

def individual_to_agent(population, config):
    agents = []
    for individual in population:
        chromosome = individual.network.params
        hidden_layer_architecture = individual.hidden_layer_architecture
        hidden_activation = individual.hidden_activation
        output_activation = individual.output_activation
        encode_row = individual.encode_row
        lifespan = individual.lifespan

        if lifespan > 0:
            agent = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, encode_row, lifespan)
            agents.append(agent)
    return agents

def _crossover_and_mutate(p1, p2, config, current_generation):
    L = len(p1.network.layer_nodes)
    c1_params = {}
    c2_params = {}

    # Perform crossover and mutation on each chromosome between parents
    for l in range(1, L):
        p1_W_l = p1.network.params['W' + str(l)]
        p2_W_l = p2.network.params['W' + str(l)]
        p1_b_l = p1.network.params['b' + str(l)]
        p2_b_l = p2.network.params['b' + str(l)]

        # Crossover
        eta = config.Crossover.sbx_eta
        c1_W_l, c2_W_l = SBX(p1_W_l, p2_W_l, eta)
        c1_b_l, c2_b_l = SBX(p1_b_l, p2_b_l, eta)

        # Mutation
        mutation_rate = config.Mutation.mutation_rate
        scale = config.Mutation.gaussian_mutation_scale

        if config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(current_generation + 1)

        # Mutate weights and biases
        gaussian_mutation(c1_W_l, mutation_rate, scale=scale)
        gaussian_mutation(c2_W_l, mutation_rate, scale=scale)
        gaussian_mutation(c1_b_l, mutation_rate, scale=scale)
        gaussian_mutation(c2_b_l, mutation_rate, scale=scale)

        # Assign children from crossover/mutation
        c1_params['W' + str(l)] = c1_W_l
        c2_params['W' + str(l)] = c2_W_l
        c1_params['b' + str(l)] = c1_b_l
        c2_params['b' + str(l)] = c2_b_l

        # Clip to [-1, 1]
        np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
        np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
        np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
        np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

    return c1_params, c2_params

def get_stats(mario):
    # Returns the game's stats for reward calculation
    frames = mario._frames if mario._frames is not None else 0
    distance = mario.x_dist if mario.x_dist is not None else 0
    game_score = mario.game_score if mario.game_score is not None else 0
    return [frames, distance, game_score]

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.stdout = sys.stderr

    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    # Initialize Logger
    ga_writer = SummaryWriter(log_dir=config.Statistics.ga_tensorboard)
    dqn_writer = SummaryWriter(log_dir=config.Statistics.dqn_tensorboard)

    logger = Logger(ga_writer, dqn_writer, config)

    # Queues for data exchange
    ga_data_queue = multiprocessing.Queue()
    dqn_data_queue = multiprocessing.Queue()

    # Start processes
    ga_process = multiprocessing.Process(target=run_ga_agent, args=(config, ga_data_queue))
    ga_process.start()

    dqn_process = multiprocessing.Process(target=run_dqn_agent, args=(config, dqn_data_queue, args.load_dqn_model))
    dqn_process.start()

    # Function to clean up processes
    def cleanup():
        ga_process.terminate()
        dqn_process.terminate()
        ga_process.join()
        dqn_process.join()
        ga_data_queue.close()
        dqn_data_queue.close()
        logger.ga_writer.close()
        logger.dqn_writer.close()

    atexit.register(cleanup)

    try:
        while True:
            # Process data from GA agent
            try:
                while True:
                    ga_data = ga_data_queue.get_nowait()
                    # Log GA metrics
                    logger.log_ga_metrics(
                        ga_data['max_fitness'],
                        ga_data['max_distance'],
                        ga_data['current_generation'],
                        ga_data['total_steps']
                    )
            except queue.Empty:
                pass

            # Process data from DQN agent
            try:
                while True:
                    dqn_data = dqn_data_queue.get_nowait()
                    # Log DQN metrics
                    logger.log_dqn_metrics(
                        dqn_data['max_fitness'],
                        dqn_data['max_distance'],
                        dqn_data['episode_rewards'],
                        dqn_data['episode_num'],
                        dqn_data['total_steps']
                    )
            except queue.Empty:
                pass

            # Sleep briefly to prevent tight loop
            time.sleep(0.1)

    except KeyboardInterrupt:
        cleanup()
