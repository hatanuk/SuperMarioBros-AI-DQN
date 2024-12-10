# multiparallel ga test


# fix the GA logging
# make it log every individual's episode stats
# remove the per-step axis

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
from tqdm import tqdm

from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config
from nn_viz import NeuralNetworkViz
from mario_torch import MarioTorch as Mario
from mario_torch import save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario

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

import logging

logging.basicConfig(level=logging.CRITICAL) 

class Logger:
    def __init__(self, writer, config):
        self.writer = writer
        self.config = config

        self.actions_to_keys_map = {
            0: "U", 
            1: "D", 
            2: "L",  
            3: "R", 
            4: "A", 
            5: "B"   
        }

    def log_ga_generation(self, total_fitness, total_distance, num_individuals, max_fitness, max_distance, generation, action_counts):
        self.writer.add_scalar('GA/max_fitness/generation', max_fitness, generation)
        self.writer.add_scalar('GA/avg_fitness/generation', round(total_fitness/num_individuals, 2), generation)
        self.writer.add_scalar('GA/max_distance/generation', max_distance, generation)
        self.writer.add_scalar('GA/avg_distance/generation', round(total_distance/num_individuals, 2), generation)

        action_dict = {f'{self.actions_to_keys_map[i]}_key': count for i, count in enumerate(action_counts)}
        self.writer.add_scalars('GA/action_counts/generation', action_dict, generation)

        values = [action_id for action_id, count in enumerate(action_counts) for _ in range(count)]
        if generation % 10 == 0 and values:
            self.writer.add_histogram('GA/action_distribution/generation', np.array(values), generation)


    def log_dqn_episode(self, episode_rewards, episode_steps, episode_distance, episode_num, max_fitness, max_distance, action_counts):
        self.writer.add_scalar('DQN/avg_reward/episode', round(episode_rewards / episode_steps, 2), episode_num)
        self.writer.add_scalar('DQN/max_fitness/episode', max_fitness, episode_num)
        self.writer.add_scalar('DQN/max_distance/episode', max_distance, episode_num)
        self.writer.add_scalar('DQN/distance/episode', episode_distance, episode_num)

        action_dict = {f'{self.actions_to_keys_map[i]}_key': count for i, count in enumerate(action_counts)}
        self.writer.add_scalars('DQN/action_counts/episode', action_dict, episode_num)

        values = [action_id for action_id, count in enumerate(action_counts) for _ in range(count)]
        if episode_num % 10 == 0 and values:
            self.writer.add_histogram('DQN/action_distribution/episode', np.array(values), episode_num)




def evaluate_individual_in_separate_process(args):


  
    individual, config = args

    # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
    ouput_to_keys_map = {
        0: 4,  # U
        1: 5,  # D
        2: 6,  # L
        3: 7,  # R
        4: 8,  # A
        5: 0   # B
    }

    print(f"[DEBUG] Starting episode for individual with initial fitness={individual.fitness}")


    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Misc.level}', render_mode='rgb_array')
    env = InputSpaceReduction(env, config)
    env.mario = individual
    obs = env.reset()

    best_fitness = 0
    max_distance = 0

    action_counts = [0] * env.action_space.n

    done = False

    # We run until the individual is no longer alive (done)
    while not done:

        # Take a step in the environment (mario is updated in wrapper)
        action = individual.get_action(obs)
        action_counts[action] += 1
        obs, reward, done, _ = env.step(action)

        if individual.farthest_x > max_distance:
            max_distance = individual.farthest_x

        if done:
            individual.calculate_fitness()
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
            break

    env.close()

    data = {
        'max_fitness': best_fitness,
        'max_distance': max_distance,
        'current_fitness': individual.fitness,
        'current_distance': individual.farthest_x,
        'action_counts': action_counts
    }

    return data


def run_ga_agent(config, data_queue):
    # Initialize population
    individuals = _initialize_population(config)
    population = Population(individuals)

    best_fitness_GA = 0.0
    max_distance_GA = 0
    current_generation = 0
    _current_individual = 0

    # Determine the size of the next generation
    if config.Selection.selection_type == 'plus':
        _next_gen_size = config.Selection.num_parents + config.Selection.num_offspring
    elif config.Selection.selection_type == 'comma':
        _next_gen_size = config.Selection.num_offspring
    else:
        raise Exception(f'Unknown selection type: {config.Selection.selection_type}')

    # Create a pool for parallel evaluation
    with multiprocessing.Pool(processes=config.GA.parallel_processes) as pool:
        while current_generation <= config.GA.total_generations:
            print("GENERATION: ", current_generation)

            # Evaluate all individuals in parallel
            args = [(ind, config) for ind in population.individuals]
            results = []
            for res in tqdm(pool.imap(evaluate_individual_in_separate_process, args), desc=f"GENERATION: {current_generation}", total=len(args)):
                results.append(res)

            # Process results for logging and stats
            total_fitness = 0
            total_distance = 0
            for i, res in enumerate(results):

                population.individuals[i]._fitness = res['current_fitness']
                population.individuals[i].farthest_x = res['max_distance']

                total_fitness += res['current_fitness']
                total_distance += res['current_distance']

                # Update global best fitness and distance
                if res['max_fitness'] > best_fitness_GA:
                    best_fitness_GA = res['max_fitness']
                if res['max_distance'] > max_distance_GA:
                    max_distance_GA = res['max_distance']

            

                # Send data to main process queue for logging steps
                data = {
                    # Global
                    'max_fitness': best_fitness_GA,
                    'max_distance': max_distance_GA,
                    'current_generation': current_generation,

                    # Per Individual
                    'current_individual': i,
                    'current_fitness': res['current_fitness'],
                    'current_distance': res['current_distance'],
                    'action_counts': res['action_counts']
                }
                data_queue.put(data)


            # Selection for next generation
            population.individuals = elitism_selection(population, config.Selection.num_parents)
            random.shuffle(population.individuals)

            next_pop = []

            # Decrement lifespan and carry over individuals if plus-selection
            if config.Selection.selection_type == 'plus':
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

                c1_params, c2_params = _crossover_and_mutate(p1, p2, config, current_generation)

                c1 = Mario(config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.encode_row, p1.lifespan)
                c2 = Mario(config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.encode_row, p2.lifespan)

                next_pop.extend([c1, c2])

            population.individuals = next_pop

            current_generation += 1


    best_individual = max(population.individuals, key=lambda ind: ind.fitness)
    save_mario('GAindividuals', 'best_mario', best_individual)
    print(f"Stopping training GA after {config.GA.total_generations} generation.")


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
        mario_DQN.model.learn(total_timesteps=int(500*config.DQN.total_episodes + 1000), callback=callback, log_interval=int(1e6))


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
        chromosome = individual.chromosome
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
    L = len(p1.model.layers)  # number of layers
    c1_params = {}
    c2_params = {}

    # Extract parameters from PyTorch layers into NumPy arrays
    for l in range(L):
        # Parents
        p1_W_l = p1.model.layers[l].weight.data.cpu().numpy()
        p2_W_l = p2.model.layers[l].weight.data.cpu().numpy()
        p1_b_l = p1.model.layers[l].bias.data.cpu().numpy()
        p2_b_l = p2.model.layers[l].bias.data.cpu().numpy()

        # Crossover
        eta = config.Crossover.sbx_eta
        c1_W_l, c2_W_l = SBX(p1_W_l, p2_W_l, eta)
        c1_b_l, c2_b_l = SBX(p1_b_l, p2_b_l, eta)

        # Mutation rate
        mutation_rate = config.Mutation.mutation_rate
        if config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(current_generation + 1)

        scale = config.Mutation.gaussian_mutation_scale

        # Mutate weights and biases
        gaussian_mutation(c1_W_l, mutation_rate, scale=scale)
        gaussian_mutation(c2_W_l, mutation_rate, scale=scale)
        gaussian_mutation(c1_b_l, mutation_rate, scale=scale)
        gaussian_mutation(c2_b_l, mutation_rate, scale=scale)

        # Clip to [-1, 1]
        np.clip(c1_W_l, -1, 1, out=c1_W_l)
        np.clip(c2_W_l, -1, 1, out=c2_W_l)
        np.clip(c1_b_l, -1, 1, out=c1_b_l)
        np.clip(c2_b_l, -1, 1, out=c2_b_l)

        # Store parameters for child 1 and child 2
        c1_params['W' + str(l+1)] = c1_W_l
        c2_params['W' + str(l+1)] = c2_W_l
        c1_params['b' + str(l+1)] = c1_b_l
        c2_params['b' + str(l+1)] = c2_b_l

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

    print("cores at disposal: ", multiprocessing.cpu_count())

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

    # Start processes
    if not args.no_ga:
        ga_data_queue = multiprocessing.Queue()
        ga_process = multiprocessing.Process(target=run_ga_agent, args=(config, ga_data_queue))
        ga_process.start()

    if not args.no_dqn:
        dqn_data_queue = multiprocessing.Queue()
        dqn_process = multiprocessing.Process(target=run_dqn_agent, args=(config, dqn_data_queue, args.load_dqn_model))
        dqn_process.start()

    # Function to clean up processes
    def cleanup():
        if not args.no_ga:
            ga_process.terminate()
            ga_process.join()
            ga_data_queue.close()
        if not args.no_dqn:
            dqn_process.terminate()
            dqn_process.join()
            dqn_data_queue.close()
        logger.writer.close()

    atexit.register(cleanup)

    try:
        ga_counter = 0
        dqn_counter = 0

        # stats specific to a generation
        gen_stats = {
            "current_gen": 0,
            "total_fitness": 0,
            "total_distance": 0,
            "max_fitness": 0,
            "max_distance": 0,
            "current_ind": 0,
            "action_counts": [0] * 6

        }

        def reset_generation_stats(stats):
            stats["total_fitness"] = 0
            stats["total_distance"] = 0
            stats["max_fitness"] = 0
            stats["max_distance"] = 0
            stats["current_ind"] = 0
            stats["action_counts"] = [0] * 6


        while True:
            # Process data from GA agent
            if args.no_ga:
                pass
            else:
                try:
                    while True:
                        ga_data = ga_data_queue.get_nowait()
                        ga_counter += 1

                        gen_stats['total_fitness'] += ga_data['current_fitness']
                        gen_stats['total_distance'] += ga_data['current_distance']
                        gen_stats['max_fitness'] = ga_data['max_fitness']
                        gen_stats['max_distance'] = ga_data['max_distance']
                        gen_stats['current_ind'] = ga_data['current_individual']

                        for i, action_count in enumerate(ga_data['action_counts']):
                            gen_stats['action_counts'][i] += action_count
                            

                        if gen_stats['current_gen'] != ga_data['current_generation']:
                            # Generation changed, log the old generation's stats
                        
                            logger.log_ga_generation(
                                total_fitness=gen_stats['total_fitness'],
                                total_distance=gen_stats['total_distance'],
                                num_individuals=gen_stats['current_ind'] + 1, # needed to get num. of individuals in generation
                                max_fitness=gen_stats['max_fitness'],
                                max_distance=gen_stats['max_distance'],
                                generation=gen_stats['current_gen'] + 1,
                                action_counts=gen_stats['action_counts']
                            )

                            gen_stats['current_gen'] = ga_data['current_generation']

                            reset_generation_stats(gen_stats)


                except queue.Empty:
                    pass
            if args.no_dqn:
                pass
            else:               
                # Process data from DQN agent
                try:
                    while True:
                        dqn_data = dqn_data_queue.get_nowait()
                        dqn_counter += 1

                        if dqn_data.get('done', False) == True:
                            logger.log_dqn_episode(

                                episode_rewards=dqn_data['episode_rewards'],
                                episode_steps=dqn_data['episode_steps'],
                                episode_distance=dqn_data['episode_distance'],
                                episode_num=dqn_data['episode_num'] + 1,
                                max_fitness=dqn_data['max_fitness'],
                                max_distance=dqn_data['max_distance'],
                                action_counts=dqn_data['action_counts']
                            )  


                except queue.Empty:
                    pass

            # Sleep briefly to prevent tight loop
            time.sleep(0.0001)

    except KeyboardInterrupt:
        cleanup()
