

import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

from PIL import Image
from typing import Tuple, List, Optional
import random
import sys
import math
import numpy as np
import argparse
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

from DQN_algorithm import DQN, DQNAgent, DQNMario

from smb_ai import next_generation

debug = False



class MainWindow:
    def __init__(self, config: Optional[Config] = None):
        global args
        self.config = config
        self.title = 'Super Mario Bros AI'

        # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
        # index                0  1     2       3      4  5  6  7  8
        self.keys = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }

        self._next_gen_size = None
        if self.config.Selection.selection_type == 'plus':
            self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
        elif self.config.Selection.selection_type == 'comma':
            self._next_gen_size = self.config.Selection.num_offspring

        self.init_ui()


        self.benchmark_stats = {'DQN': [], 'GA': []}

        # GA init
        individuals = self._initialize_population()
        self.population = Population(individuals)
        self.mario_GA = self.population.individuals[0]
        self.best_fitness_GA= 0.0
        self.max_distance_GA = 0
        self._true_zero_gen = 0
        self.current_generation = 0
        self.env_GA = self._make_env()
        self._current_individual = 0

        # DQN init
        self.mario_DQN = DQNMario(self.config)
        self.best_fitness_DQN = 0.0
        self.max_distance_DQN = 0
        self.env_DQN = self._make_env()
        self.step_counter_GA = 0


        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(1000 // 60) # 60 fps

    def init_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.dqn_fitness_label = QLabel('DQN Fitness: 0')
        self.ga_fitness_label = QLabel('GA Fitness: 0')
        self.ga_generation_label = QLabel('Generation: 0')
        self.ga_distance_label = QLabel('Max Distance GA: 0')
        self.dqn_distance_label = QLabel('Max Distance DQN: 0')

        layout.addWidget(self.dqn_fitness_label)
        layout.addWidget(self.dqn_distance_label)
        layout.addWidget(self.ga_fitness_label)
        layout.addWidget(self.ga_generation_label)
        layout.addWidget(self.ga_distance_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


    def _update_gui(self):
        self.dqn_fitness_label.setText(f'DQN Fitness: {self.mario_DQN.fitness}')
        self.dqn_distance_label.setText(f'Max Distance DQN: {self.max_distance_DQN}')
        self.ga_fitness_label.setText(f'GA Fitness: {self.mario_GA.fitness}')
        self.ga_generation_label.setText(f'Generation: {self.current_generation}')
        self.ga_distance_label.setText(f'Max Distance GA: {self.max_distance_GA}')


    def _make_env(self):
        return retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.Misc.level}')

    def _update_gui(self):
        pass

    def _initialize_population(self):
        individuals: List[Individual] = []
        num_parents = self.config.Selection.num_parents
        for _ in range(num_parents):
            individual = Mario(self.config)
            individuals.append(individual)
        return individuals


    def _update(self) -> None:
        self._update_dqn()
        self._update_ga()
        self._update_gui()
    
    def _get_state(self, env, mario):
        ram = env.get_ram()
        tiles = SMB.get_tiles(ram)
        mario.set_input_as_array(ram, tiles)
        return mario.input_as_array
    
    def _get_stats(self, mario):
        frames = mario._frames
        distance = mario.x_dist
        score = mario.game_score

        return {
            "frames" : frames,
            "distance" : distance,
            "score" : score
        }

    def _update_dqn(self):

        # stats are a dict of frames, distance and score
        curr_state = self._get_state(self.env_DQN, self.mario_DQN)
        curr_stats = self._get_stats(self.mario_DQN)
        action = self.mario_DQN.buttons_to_press
        self._take_action(self.mario_DQN, self.env_DQN, action)
        next_stats = self._get_stats(self.mario_DQN)
        next_state = self._get_state(self.env_DQN, self.mario_DQN)
        done = not self.mario_DQN.is_alive

        reward = self.calculate_reward(curr_stats, next_stats)

        self.mario_DQN.replay_buffer.add(curr_state, action, next_state, reward, done)
        self.mario_DQN.learn()

        # Fitness = total reward
        self.mario_DQN.fitness += reward

        if not done:
            if self.mario_DQN.farthest_x > self.max_distance_DQN:
                self.max_distance_DQN = self.mario_DQN.farthest_x
        else:
            self.benchmark_stats['DQN'].append({
                'reward': reward,
                'distance': self.mario_DQN.farthest_x,
                'fitness': self.mario_DQN.fitness,
                'epsilon': self.mario_DQN.epsilon,
                'steps': self.mario_DQN.step_counter
            })
            self.env_DQN.reset()
            self.mario_DQN.fitness = 0

    def _take_action(self, mario, env, action):
        ram = env.get_ram()
        tiles = SMB.get_tiles(ram)
        env.step(action)
        mario.update(ram, tiles, self.keys, self.ouput_to_keys_map)

    def _update_ga(self):

        self.step_counter_GA += 1 

        curr_stats = self._get_stats(self.mario_GA)
        action = self.mario_GA.buttons_to_press
        self._take_action(self.mario_GA, self.env_GA, action)
        next_stats = self._get_stats(self.mario_GA)


        if self.mario_GA.is_alive:
            if self.mario_GA.farthest_x > self.max_distance_GA:
                self.max_distance_GA = self.mario_GA.farthest_x
            self.mario_GA.fitness+= self.calculate_reward(curr_stats, next_stats)
            
        else:
            fitness = self.mario_GA.fitness
            self.benchmark_stats['GA'].append({
                'fitness': fitness,
                'distance': self.mario_GA.farthest_x,
                'generation': self.current_generation,
                'steps': self.step_counter_GA
            })

            self._next_individual_or_generation()

    def _next_individual_or_generation(self):
        self._current_individual += 1
        if self._current_individual >= len(self.population.individuals):
  
            self.next_generation()
        else:
            self.mario_GA = self.population.individuals[self._current_individual]
            self.env_GA.reset()

  

    def _next_generation(self) -> None:
        self._current_individual = 0
        self.current_generation += 1

        if debug == True:
            print(f'----Current Gen: {self.current_generation}, True Zero: {self._true_zero_gen}')
            fittest = self.population.fittest_individual
            print(f'Best fitness of gen: {fittest.fitness}, Max dist of gen: {fittest.farthest_x}')
            num_wins = sum(individual.did_win for individual in self.population.individuals)
            pop_size = len(self.population.individuals)
            print(f'Wins: {num_wins}/{pop_size} (~{(float(num_wins)/pop_size*100):.2f}%)')

        if self.config.Statistics.save_best_individual_from_generation:
            folder = self.config.Statistics.save_best_individual_from_generation
            best_ind_name = 'best_ind_gen{}'.format(self.current_generation - 1)
            best_ind = self.population.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        if self.config.Statistics.save_population_stats:
            fname = self.config.Statistics.save_population_stats
            save_stats(self.population, fname)

        self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)

        random.shuffle(self.population.individuals)
        next_pop = []

        # Parents + offspring
        if self.config.Selection.selection_type == 'plus':
            # Decrement lifespan
            for individual in self.population.individuals:
                individual.lifespan -= 1

            for individual in self.population.individuals:
                config = individual.config
                chromosome = individual.network.params
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan
                name = individual.name

                # If the indivdual would be alve, add it to the next pop
                if lifespan > 0:
                    m = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, lifespan)
                    # Set debug if needed
                    if debug:
                        m.name = f'{name}_life{lifespan}'
                        m.debug = True
                    next_pop.append(m)

        num_loaded = 0

        while len(next_pop) < self._next_gen_size:
            selection = self.config.Crossover.crossover_selection
            if selection == 'tournament':
                p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
            elif selection == 'roulette':
                p1, p2 = roulette_wheel_selection(self.population, 2)
            else:
                raise Exception('crossover_selection "{}" is not supported'.format(selection))

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]  
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                #  Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])


            c1 = Mario(self.config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
            c2 = Mario(self.config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

            # Set debug if needed
            if debug:
                c1_name = f'm{num_loaded}_new'
                c1.name = c1_name
                c1.debug = True
                num_loaded += 1

                c2_name = f'm{num_loaded}_new'
                c2.name = c2_name
                c2.debug = True
                num_loaded += 1

            next_pop.extend([c1, c2])

        # Set next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eta = self.config.Crossover.sbx_eta

        # SBX weights and bias
        child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
        child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, eta)

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        mutation_rate = self.config.Mutation.mutation_rate
        scale = self.config.Mutation.gaussian_mutation_scale

        if self.config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)
        
        # Mutate weights
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

        # Mutate bias
        gaussian_mutation(child1_bias, mutation_rate, scale=scale)
        gaussian_mutation(child2_bias, mutation_rate, scale=scale)


if __name__ == "__main__":
    config = "/settings.config"
    config = Config(config)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())