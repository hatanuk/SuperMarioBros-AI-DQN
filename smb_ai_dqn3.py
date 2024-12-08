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
import shutil
import os

from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config
from nn_viz import NeuralNetworkViz
from mario import Mario, save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation

from DQN_algorithm.DQNbaseline import DQNCallback, DQNMario, InputSpaceReduction

from smb_ai import draw_border, parse_args

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import multiprocessing
import queue
import time  # Added time module for sleep

import atexit

from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, writer, config):
        self.writer = writer
        self.config = config

    def log_ga_step(self, max_fitness, max_distance, total_steps):
        if total_steps % self.config.Statistics.log_interval == 0:
            self.writer.add_scalar('GA/max_fitness', max_fitness, total_steps)
            self.writer.add_scalar('GA/max_distance', max_distance, total_steps)
            self.writer.add_scalar('GA/total_steps', total_steps, total_steps)

    def log_dqn_step(self, max_fitness, max_distance, total_steps):
        if total_steps % self.config.Statistics.log_interval == 0:
            self.writer.add_scalar('DQN/max_fitness', max_fitness, total_steps)
            self.writer.add_scalar('DQN/max_distance', max_distance, total_steps)
            self.writer.add_scalar('DQN/total_steps', total_steps, total_steps)
        
        #self.dqn_writer.add_scalar('episode_reward', episode_reward, episode_num)

    def log_ga_generation(self, total_fitness, total_distance, num_individuals, max_fitness, max_distance, generation):
        self.writer.add_scalar('GA/max_fitness/generation', max_fitness, generation)
        self.writer.add_scalar('GA/avg_fitness/generation', round(total_fitness/num_individuals, 2), generation)
        self.writer.add_scalar('GA/max_distance/generation', max_distance, generation)
        self.writer.add_scalar('GA/avg_distance/generation', round(total_distance/num_individuals, 2), generation)

    def log_dqn_episode(self, episode_reward, episode_num):
        self.writer.add_scalar('DQN/reward/episode', episode_reward, episode_num)



def run_ga_agent(config, data_queue):
    # Initialize environment
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Misc.level}', render_mode='human')

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

    while current_generation <= self.config.GA.total_generations:
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
            # Global
            'max_fitness': best_fitness_GA,
            'max_distance': max_distance_GA,
            'total_steps': total_steps_GA,
            'current_generation': current_generation,

            # Per Individual
            'current_individual': _current_individual,
            'current_fitness': mario_GA.fitness,
            'current_distance': mario_GA.farthest_x,
        }

        # Send data to main process
        data_queue.put(data)

    # end while
    print(f"Stopping training GA after {self.config.GA.total_generations} generation.")

 
def run_dqn_agent(config, data_queue, dqn_model):

    # Initialize environment
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Misc.level}', render_mode='rgb_array')
    env = InputSpaceReduction(env, config)

    # Initialize DQN agent
    mario_DQN = DQNMario(config, env)
    env.mario = mario_DQN

    env = DummyVecEnv([lambda: env])


    if dqn_model:
        try:
            model = DQN.load(dqn_model, env=env)
            mario_DQN.model = model
        except Exception as e:
            raise Exception(f'Failed to find model at {dqn_model}') from e
        # Add an inference loop here later
    else:
        callback = DQNCallback(data_queue, mario_DQN, config, verbose=1)
        mario_DQN.model.learn(total_timesteps=int(1e6), callback=callback, log_interval=1)

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

def clear_tensorboard_log_dir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir) 
        print(f"Cleared TensorBoard log directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.stdout = sys.stderr

    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    # clear prior tensorboard logs
    clear_tensorboard_log_dir(config.Statistics.tensorboard_dir)
    clear_tensorboard_log_dir('./monitor_logs/DQNtbFromBaseline')

    # Initialize Logger
    writer = SummaryWriter(log_dir=config.Statistics.tensorboard_dir)

    logger = Logger(writer, config)

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
        logger.writer.close()

    atexit.register(cleanup)

    try:
        ga_counter = 0
        dqn_counter = 0
        processed_steps_ga = set()
        processed_steps_dqn = set()

        # stats specific to a generation
        gen_stats = {
            "current_gen": 0,
            "current_ind": 0,
            "total_fitness": 0,
            "total_distance": 0,
        }

        def reset_generation_stats(stats):
            stats["current_ind"] = 0
            stats["total_fitness"] = 0
            stats["total_distance"] = 0



        while True:
            # Process data from GA agent
            try:
                while True:
                    ga_data = ga_data_queue.get_nowait()
                    ga_counter += 1

                    if ga_counter % 1000 == 0:
                        print("updating GA: ", ga_data)

                    if ga_data['total_steps'] not in processed_steps_ga:
                        processed_steps_ga.add(ga_data['total_steps'])
                        logger.log_ga_step(
                            ga_data['max_fitness'],
                            ga_data['max_distance'],
                            ga_data['total_steps']
                        )

                        if gen_stats['current_ind'] != ga_data['current_individual']:
                            # Individual changed, collect stats
                            gen_stats['current_ind']= ga_data['current_individual']
                            gen_stats['total_fitness'] += ga_data['current_fitness']
                            gen_stats['total_distance'] += ga_data['current_distance']

                    if gen_stats['current_gen'] !=  ga_data['current_generation']:
                        # Generation changed, log the stats
                        gen_stats['current_gen'] = ga_data['current_generation']
                       
                        logger.log_ga_generation(
                            gen_stats['total_fitness'],
                            gen_stats['total_distance'],
                            gen_stats['current_ind'] + 1,
                            ga_data['max_fitness'],
                            ga_data['max_distance'],
                            gen_stats['current_gen']
                        )
                        reset_generation_stats(gen_stats)


            except queue.Empty:
                pass

            # Process data from DQN agent
            try:
                while True:
                    dqn_data = dqn_data_queue.get_nowait()
                    dqn_counter += 1
                    if dqn_counter % 1000 == 0:
                        print("updating DQN: ", dqn_data)
                    # Log DQN metrics
                    if dqn_data['total_steps'] not in processed_steps_dqn:
                        processed_steps_dqn.add(dqn_data['total_steps'])
                        logger.log_dqn_step(
                            dqn_data['max_fitness'],
                            dqn_data['max_distance'],
                            dqn_data['total_steps']
                        )

                    if dqn_data['done'] == True:
                        logger.log_dqn_episode(
                            dqn_data['episode_reward'],
                            dqn_data['episode_num']
                        )

            except queue.Empty:
                pass

            # Sleep briefly to prevent tight loop
            time.sleep(0.1)

    except KeyboardInterrupt:
        cleanup()


