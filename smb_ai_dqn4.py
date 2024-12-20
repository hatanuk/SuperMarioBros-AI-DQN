# multiparallel ga test


# fix the GA logging
# make it log every individual's episode stats
# remove the per-step axis

import argparse
import re

import torch
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
from mario_torch import save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario, state_dict_to_chromosome

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation

from DQN_algorithm.DQNbaseline import DQNCallback, DQNMario, InputSpaceReduction, clear_dir

from smb_ai import draw_border

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import multiprocessing
import queue
import time  # Added time module for sleep

import atexit

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros AI')

    # Config
    parser.add_argument('-c', '--config', dest='config', required=False, help='config file to use')

    parser.add_argument('--load_dqn_model', dest='load_dqn_model', default=None, help='/path/to/model.pt to load DQN model to continue training')
    parser.add_argument('--load_ga_model', dest='load_ga_model', default=None, help='/path/to/model.pt to load a GA model which will be used to populate the a population and continue training')

    parser.add_argument('--no_dqn', dest='no_dqn',  action='store_true', required=False, help='do not train a dqn model')
    parser.add_argument('--no_ga', dest='no_ga',  action='store_true', required=False, help='do not train a ga model')


    args = parser.parse_args()

    return args

def validate_model(model_path):
    try:
        details = torch.load(model_path)
    except FileNotFoundError:
        raise Exception(f'Failed to find model at {model_path}')
    except Exception as e:
        raise Exception(f'Failed to load model at {model_path}: {e}')
    required_keys = ['state_dict', 'layer_sizes', 'hidden_activation', 'output_activation', 'encode_row', 'input_dims', 'algorithm', 'iterations', 'distance']
    for key in required_keys:
        if key not in details:
            raise KeyError(f"Missing key in the model file: {key}")
        
    print(f"Loaded {details['algorithm']} model at {model_path} with {details['iterations']} iterations and {details['distance']} distance")

class Logger:
    def __init__(self, writer, config):
        self.writer = writer
        self.config = config

        self.actions_to_keys_map = {
            0: "None", 
            1: "Right", 
            2: "Right + A",  
            3: "Right + B", 
            4: "Right + A + B", 
            5: "A",
            6: "Left"
        }

    def log_ga_generation(self, total_steps, total_fitness, total_distance, num_individuals, max_fitness, max_distance, generation, action_counts):

        print(f"avg distance GA: {round(total_distance/num_individuals, 2)}")
        self.writer.add_scalar('GA/max_fitness/generation', max_fitness, generation)
        self.writer.add_scalar('GA/avg_fitness/generation', round(total_fitness/num_individuals, 2), generation)
        self.writer.add_scalar('GA/max_distance/generation', max_distance, generation)
        self.writer.add_scalar('GA/avg_distance/generation', round(total_distance/num_individuals, 2), generation)

        self.writer.add_scalar('GA/max_fitness/step', max_fitness, total_steps)

        action_counts = action_counts[:len(self.actions_to_keys_map)]
        action_total = sum(action_counts)
        action_dict = {f'{self.actions_to_keys_map[i]}': round(count/action_total, 2) for i, count in enumerate(action_counts)}
        self.writer.add_scalars('GA/action_counts/generation', action_dict, generation)

        values = [action_id for action_id, count in enumerate(action_counts) for _ in range(count)]
        if generation % 500 == 0 and values:
            self.writer.add_histogram('GA/action_distribution/generation', np.array(values), generation)


    def log_dqn_episode(self, total_steps, fitness, episode_steps, episode_rewards, episode_distance, episode_num, max_fitness, max_distance, action_counts, epsilon):
        print(f"max distance DQN: {max_distance}")
        self.writer.add_scalar('DQN/fitness/episode', fitness, episode_num)
        self.writer.add_scalar('DQN/distance/episode', episode_distance, episode_num)
        self.writer.add_scalar('DQN/max_fitness/episode', max_fitness, episode_num)
        self.writer.add_scalar('DQN/max_distance/episode', max_distance, episode_num)

        self.writer.add_scalar('DQN/avg_delta_reward/episode', round(episode_rewards/episode_steps, 2), episode_num)

        self.writer.add_scalar('DQN/max_fitness/step', max_fitness, total_steps)

        self.writer.add_scalar('DQN/epsilon/episode', round(epsilon, 3), episode_num)

        action_counts = action_counts[:len(self.actions_to_keys_map)] 
        action_total = sum(action_counts)
        action_dict = {f'{self.actions_to_keys_map[i]}_key': round(count/action_total, 2) for i, count in enumerate(action_counts)}
        self.writer.add_scalars('DQN/action_counts/episode', action_dict, episode_num)

        values = [action_id for action_id, count in enumerate(action_counts) for _ in range(count)]
        if episode_num % 500 == 0 and values:
            self.writer.add_histogram('DQN/action_distribution/episode', np.array(values), episode_num)




def evaluate_individual_in_separate_process(args):

    individual, config = args

    start = time.time()

    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Environment.level}', render_mode='rgb_array')
    env = InputSpaceReduction(env, input_dims=config.NeuralNetworkGA.input_dims, encode_row=config.NeuralNetworkGA.encode_row, skip=config.Environment.frame_skip)
    env.mario = individual
  
    obs = env.reset()

    best_fitness = 0
    max_distance = 0

    action_counts = [0] * env.action_space.n

    done = False

    # We run until the individual is no longer alive (done)
    while not done:

        # inferencing
        action = individual.get_action(obs)
        action_counts[action] += 1

        # Take a step in the environment (mario is updated in wrapper)
        obs, rewards, done, _ = env.step(action)
        
        if individual.farthest_x > max_distance:
            max_distance = individual.farthest_x

        if done:
            individual.calculate_fitness()
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
            break

    end = time.time()

    data = {
        'current_fitness': individual.fitness,
        'current_distance': individual.farthest_x,
        'action_counts': action_counts,
        'wall_clock_time': float(f"{(end - start):.5f}"),
        'episode_steps': env.episode_steps,
    }

    return data


def run_ga_agent(config, data_queue, model_save_path):

    if model_save_path:
        details = torch.load(model_save_path, weights_only=False)
    else:
        details = None
        
    individuals = _initialize_population(config, details)
    population = Population(individuals)

    if model_save_path:
        max_distance_GA = details['distance']
        best_fitness_GA = individuals[0].fitness_func(max_distance_GA)
        current_generation = details['iterations']
    else:
        current_generation = 0
        best_fitness_GA = 0.0
        max_distance_GA = 0

    _current_individual = 0

    best_individual = None

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

            # Evaluate all individuals in parallel
            args = [(ind, config) for ind in population.individuals]
            results = []
            for res in tqdm(pool.imap(evaluate_individual_in_separate_process, args), desc=f"GENERATION: {current_generation}", total=len(args)):
                results.append(res)

            # Process results for logging and stats
            total_fitness = 0
            total_distance = 0
            average_time = 0
            for i, res in enumerate(results):

                average_time += res['wall_clock_time']

                population.individuals[i]._fitness = res['current_fitness']
                population.individuals[i].farthest_x = res['current_distance']

                total_fitness += res['current_fitness']
                total_distance += res['current_distance']

                # Update global best fitness and distance (and save the best individual)
                if res['current_fitness'] > best_fitness_GA:
                    best_fitness_GA = res['current_fitness']
                    best_individual = population.individuals[i]

                if res['current_distance'] > max_distance_GA:
                    max_distance_GA = res['current_distance']

            

                # Send data to main process queue for logging steps
                data = {
                    # Global
                    'max_fitness': best_fitness_GA,
                    'max_distance': max_distance_GA,
                    'current_generation': current_generation,

                    # Per Individual
                    'episode_steps': res['episode_steps'],
                    'current_individual': i,
                    'current_fitness': res['current_fitness'],
                    'current_distance': res['current_distance'],
                    'action_counts': res['action_counts']
                }
                data_queue.put(data)
            
            #print(f"average time for a generational episode: {average_time / len(results):.5f}")
            
            if current_generation == config.GA.total_generations:
                save_mario_pop(population, config, config.GA.total_generations, "_FINAL")
                if best_individual:
                    save_overall_best_individual(best_individual, config, config.GA.total_generations)

            elif current_generation % config.Statistics.ga_checkpoint_interval == 0 and current_generation > 0:
                save_mario_pop(population, config, current_generation, "_CHECKPOINT")
                if best_individual:
                    save_overall_best_individual(best_individual, config, current_generation)

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

                
                c1 = Mario(c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.encode_row, p1.lifespan, p1.frame_skip, p1.input_dims, p1.allow_additional_time)
                c2 = Mario(c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.encode_row, p2.lifespan, p2.frame_skip, p2.input_dims, p2.allow_additional_time)

                next_pop.extend([c1, c2])

            population.individuals = next_pop

            current_generation += 1

def save_mario_pop(population, config, generation, postfix=""):
    best_individuals = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)[:config.Statistics.top_x_individuals:]
    for i, ind in enumerate(best_individuals):
        fitness = int(max(0, min(ind.fitness, 99999999)))
        save_mario(f'{config.Statistics.model_save_dir}/GA/GEN{generation}{postfix}', f'{config.Statistics.ga_model_name}_ind{i+1}_fitness{fitness}', ind, generation, ind.farthest_x)
   
def save_overall_best_individual(individual, config, generation):
    clear_dir(f'{config.Statistics.model_save_dir}/GA/OVERALL_BEST')
    fitness = int(max(0, min(individual.fitness, 99999999)))
    save_mario(f'{config.Statistics.model_save_dir}/GA/OVERALL_BEST', f'{config.Statistics.ga_model_name}_best_fitness{fitness}', individual, generation, individual.farthest_x)


def run_dqn_agent(config, data_queue, model_save_path):

    # Initialize environment
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Environment.level}', render_mode='rgb_array')
    env = InputSpaceReduction(env, input_dims=config.NeuralNetworkDQN.input_dims, encode_row=config.NeuralNetworkDQN.encode_row, skip=config.Environment.frame_skip)

    # Initialize DQN agent
    mario_DQN = DQNMario(config, env)
    env.mario = mario_DQN

    env = DummyVecEnv([lambda: env])

    episode_start = 0

    if model_save_path:
        iterations, _ = mario_DQN.load_saved_model(model_save_path, env)
        episode_start = iterations
   
    callback = DQNCallback(data_queue, mario_DQN, config, verbose=1, episode_start=episode_start)

    # total_timesteps and log_interval should be unreachably high - the callback will stop the training
    mario_DQN.model.learn(total_timesteps=int(100_000 * config.DQN.total_episodes), callback=callback, log_interval=int(100_000 * config.DQN.total_episodes))


def _initialize_population(config, details=None):
    individuals: List[Individual] = []
    num_parents = config.Selection.num_parents
    num_offspring = config.Selection.num_offspring

    hidden_layer_architecture = config.NeuralNetworkGA.hidden_layer_architecture
    hidden_activation = config.NeuralNetworkGA.hidden_node_activation
    output_activation = config.NeuralNetworkGA.output_node_activation
    encode_row = config.NeuralNetworkGA.encode_row
    lifespan = config.Selection.lifespan
    frame_skip = config.Environment.frame_skip
    input_dims = config.NeuralNetworkGA.input_dims
    allow_additional_time= config.Misc.allow_additional_time_for_flagpole

    if details:
        # Populate parent population with copies of the saved model
       
        model_chromosome = state_dict_to_chromosome(details['state_dict'])

        # Include the loaded model in the initial population
        individuals.append(Mario(model_chromosome, details['layer_sizes'][1:-1], details["hidden_activation"], details["output_activation"], details["encode_row"], lifespan, frame_skip, details["input_dims"], allow_additional_time))

        while len(individuals) < num_offspring:
            # Populate the initial population with mutated copies of the loaded model
            children = _crossover_and_mutate(model_chromosome, model_chromosome, config, details['iterations'])

            for child in children:
                if len(individuals) < num_offspring:
                    individuals.append(Mario(child, details['layer_sizes'][1:-1], details["hidden_activation"], details["output_activation"], details["encode_row"], lifespan, frame_skip, details["input_dims"], allow_additional_time))
         
    else:
        for _ in range(num_parents):
            individual = Mario(None, hidden_layer_architecture, hidden_activation, output_activation, encode_row, lifespan, frame_skip, input_dims, allow_additional_time)
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
        frame_skip = individual.frame_skip
        input_dims = individual.input_dims
        allow_additional_time = individual.allow_additional_time

        if lifespan > 0:
            agent = Mario(chromosome, hidden_layer_architecture, hidden_activation, output_activation, encode_row, lifespan, frame_skip, input_dims, allow_additional_time)
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


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.stdout = sys.stderr

    print("cores at disposal: ", multiprocessing.cpu_count())

    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    # clear prior tensorboard logs
    clear_dir(config.Statistics.tensorboard_dir)

    # clear prior model saves
    clear_dir(config.Statistics.model_save_dir)

    if not os.path.exists(config.Statistics.model_save_dir):
        os.makedirs(config.Statistics.model_save_dir)
    if not os.path.exists(f'{config.Statistics.model_save_dir}/GA'):
        os.makedirs(f'{config.Statistics.model_save_dir}/GA')
    if not os.path.exists(f'{config.Statistics.model_save_dir}/DQN'):
        os.makedirs(f'{config.Statistics.model_save_dir}/DQN')

    # Add copy of the config file
    shutil.copy(args.config, f'{config.Statistics.model_save_dir}/settings.config')

    # Initialize Logger
    writer = SummaryWriter(log_dir=config.Statistics.tensorboard_dir)

    logger = Logger(writer, config)

    if args.load_ga_model:
        validate_model(args.load_ga_model)
    if args.load_dqn_model:
        validate_model(args.load_dqn_model)

    # Start processes
    if not args.no_ga:
        ga_data_queue = multiprocessing.Queue()
        ga_process = multiprocessing.Process(target=run_ga_agent, args=(config, ga_data_queue. args.load_ga_model))
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
      
        total_steps_dqn = 0 

        # stats specific to a generation
        gen_stats = {
            "current_gen": 0,
            "total_fitness": 0,
            "total_distance": 0,
            "total_steps": 0,
            "max_fitness": 0,
            "max_distance": 0,
            "current_ind": 0,
            "action_counts": [0] * 9

        }

        def reset_generation_stats(stats):
            stats["total_fitness"] = 0
            stats["total_distance"] = 0
            stats['total_steps'] = 0
            stats["max_fitness"] = 0
            stats["max_distance"] = 0
            stats["current_ind"] = 0
            stats["action_counts"] = [0] * 9


        while True:
            # Process data from GA agent
            if args.no_ga:
                pass
            else:
                try:
                    while True:
                        ga_data = ga_data_queue.get_nowait()

                        if gen_stats['current_gen'] != ga_data['current_generation']:
                            # Generation changed, log the old generation's stats
                            gen_stats['current_gen'] = ga_data['current_generation']
                            reset_generation_stats(gen_stats)
                            if ga_data['current_generation'] % config.Statistics.log_interval == 0:
                         
                                logger.log_ga_generation(
                                    total_steps=gen_stats['total_steps'],
                                    total_fitness=gen_stats['total_fitness'],
                                    total_distance=gen_stats['total_distance'],
                                    num_individuals=gen_stats['current_ind'] + 1, # needed to get num. of individuals in generation
                                    max_fitness=gen_stats['max_fitness'],
                                    max_distance=gen_stats['max_distance'],
                                    generation=gen_stats['current_gen'] + 1,
                                    action_counts=gen_stats['action_counts']
                                )

                        gen_stats['total_fitness'] += ga_data['current_fitness']
                        gen_stats['total_steps'] += ga_data['episode_steps']
                        gen_stats['total_distance'] += ga_data['current_distance']
                        gen_stats['max_fitness'] = max(gen_stats['max_fitness'], ga_data['max_fitness'])
                        gen_stats['max_distance'] = max(gen_stats['max_distance'], ga_data['max_distance'])
                        gen_stats['current_ind'] = ga_data['current_individual']

                        for i, action_count in enumerate(ga_data['action_counts']):
                            gen_stats['action_counts'][i] += action_count

                except queue.Empty:
                    pass
            if args.no_dqn:
                pass
            else:               
                # Process data from DQN agent
                try:
         
                    while True:
                        dqn_data = dqn_data_queue.get_nowait()

                        total_steps_dqn += dqn_data['episode_steps']

                        if dqn_data['episode_num'] % config.Statistics.log_interval == 0:
                            logger.log_dqn_episode(
                                total_steps = total_steps_dqn,
                                fitness=dqn_data['fitness'],
                                episode_steps=dqn_data['episode_steps'],
                                episode_rewards=dqn_data['episode_rewards'],
                                episode_distance=dqn_data['episode_distance'],
                                episode_num=dqn_data['episode_num'] + 1,
                                max_fitness=dqn_data['max_fitness'],
                                max_distance=dqn_data['max_distance'],
                                action_counts=dqn_data['action_counts'],
                                epsilon=dqn_data['epsilon']
                            )  


                except queue.Empty:
                    pass

            # Sleep briefly to prevent tight loop
            time.sleep(0.001)

    except KeyboardInterrupt:
        cleanup()
