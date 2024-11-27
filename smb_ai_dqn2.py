import re
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

from smb_ai import next_generation, draw_border, parse_args

normal_font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)


class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config, nn_viz: NeuralNetworkViz):
        super().__init__(parent)
        self.size = size
        self.config = config
        self.nn_viz = nn_viz
        self.ram = None
        self.x_offset = 150
        self.tile_width, self.tile_height = self.config.Graphics.tile_size
        self.tiles = None
        self.enemies = None
        self._should_update = True

    def _draw_region_of_interest(self, painter: QPainter) -> None:
        # Grab mario row/col in our tiles
        mario = SMB.get_mario_location_on_screen(self.ram)
        mario_row, mario_col = SMB.get_mario_row_col(self.ram)
        x = mario_col
       
        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))

        start_row, viz_width, viz_height = self.config.NeuralNetwork.input_dims
        painter.drawRect(x*self.tile_width + 5 + self.x_offset, start_row*self.tile_height + 5, viz_width*self.tile_width, viz_height*self.tile_height)


    def draw_tiles(self, painter: QPainter):
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                painter.setPen(QPen(Qt.black,  1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (self.tile_width * col) + self.x_offset
                y_start = 5 + (self.tile_height * row)

                loc = (row, col)
                tile = self.tiles[loc]

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
                    color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass

                painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        if self._should_update:
            draw_border(painter, self.size)
            if not self.ram is None:
                self.draw_tiles(painter)
                self._draw_region_of_interest(painter)
                self.nn_viz.show_network(painter)
        else:
            # draw_border(painter, self.size)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
            txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
            painter.drawText(event.rect(), Qt.AlignCenter, txt)
            pass

        painter.end()

    def _update(self):
        self.update()


class GameWindow(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config):
        super().__init__(parent)
        self._should_update = True
        self.size = size
        self.config = config
        self.screen = None
        self.img_label = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.img_label)
        self.setLayout(self.layout)
        
 
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        if self._should_update:
            draw_border(painter, self.size)
            if not self.ram is None:
                self.draw_tiles(painter)
                self._draw_region_of_interest(painter)
                self.nn_viz.show_network(painter)
        else:
            # draw_border(painter, self.size)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
            txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
            painter.drawText(event.rect(), Qt.AlignCenter, txt)
            pass

        painter.end()

    def _update(self):
        self.update()

class InformationWidget(QtWidgets.QWidget):
    def __init__(self, parent, size, config):
        super().__init__(parent)
        self.size = size
        self.config = config

        self.grid = QtWidgets.QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        # self.grid.setSpacing(20)
        self.setLayout(self.grid)

        self.init_ga_info()
        self.init_dqn_info()


    def get_info_ga(self):

        lifespan = self.config.Selection.lifespan
        lifespan_txt = 'Infinite' if lifespan == np.inf else str(lifespan)

        selection_type = self.config.Selection.selection_type
        num_parents = self.config.Selection.num_parents
        num_offspring = self.config.Selection.num_offspring
        if selection_type == 'comma':
            selection_txt = '{}, {}'.format(num_parents, num_offspring)
        elif selection_type == 'plus':
            selection_txt = '{} + {}'.format(num_parents, num_offspring)
        else:
            raise Exception('Unkown Selection type "{}"'.format(selection_type))

        num_inputs = get_num_inputs(self.config)
        hidden = self.config.NeuralNetwork.hidden_layer_architecture
        num_outputs = 6
        L = [num_inputs] + hidden + [num_outputs]
        layers_txt = '[' + ', '.join(str(nodes) for nodes in L) + ']'

        mutation_rate = self.config.Mutation.mutation_rate
        mutation_type = self.config.Mutation.mutation_rate_type.capitalize()
        mutation_txt = '{} {}% '.format(mutation_type, str(round(mutation_rate*100, 2)))

        crossover_selection = self.config.Crossover.crossover_selection
        if crossover_selection == 'roulette':
            crossover_txt = 'Roulette'
        elif crossover_selection == 'tournament':
            crossover_txt = 'Tournament({})'.format(self.config.Crossover.tournament_size)
        else:
            raise Exception('Unknown crossover selection "{}"'.format(crossover_selection))

        info_dict = {
            "Generation": '0',
            "Individual": '0',
            "Best Fitness": '0',
            "Max Distance": '0',
            "Total Steps": '0',
            "Num Inputs": str(get_num_inputs(self.config)),
            "Trainable Params": str(get_num_trainable_parameters(self.config)),
            "Offspring": selection_txt,
            "Lifespan": lifespan_txt,
            "Mutation": mutation_txt,
            "Crossover": crossover_txt,
            "SBX Eta": str(self.config.Crossover.sbx_eta),
            "Layers": layers_txt
        }

        return info_dict

    def get_dqn_info(self):

        num_inputs = get_num_inputs(self.config)
        hidden = self.config.NeuralNetworkDQN.hidden_layer_architecture
        num_outputs = 6
        L = [num_inputs] + hidden + [num_outputs]
        layers_txt = '[' + ', '.join(str(nodes) for nodes in L) + ']'

        learning_rate_txt = str(self.config.NeuralNetworkDQN.learning_rate)

        # Prepare info_dict
        info_dict = {
            "Individual": '0',
            "Best Fitness": '0',
            "Max Distance": '0',
            "Total Steps": '0',
            "Learning Rate": learning_rate_txt, 
            "Layers": layers_txt,
        }

        return info_dict

    
    def init_ga_info(self):
        self.ga_vbox = QVBoxLayout()
        self.ga_vbox.setContentsMargins(0, 0, 0, 0)
        ga_info_dict = self.get_info_ga()
        self._create_info_section(ga_info_dict, self.ga_vbox, prefix='ga_')

        self.grid.addLayout(self.ga_vbox, 0, 0) #1st column
     
    
    def init_dqn_info(self):
        dqn_info_dict = self.get_info_dqn()
        self.dqn_vbox = QVBoxLayout()
        self.dqn_vbox.setContentsMargins(0, 0, 0, 0)
        self._create_info_section(dqn_info_dict, self.dqn_vbox, prefix='dqn_')

        self.grid.addLayout(self.dqn_vbox, 0, 1) #2nd column
        

    @staticmethod
    def to_attribute_name(input_string):
        input_string = input_string.replace(" ", "_")
        cleaned_string = re.sub(r'[^a-zA-Z_]', '', input_string)
        return cleaned_string


    def _create_info_section(self, info_dict, vbox, prefix=''):
      
        for key in info_dict.keys():

            label_title = QLabel()
            label_title.setFont(font_bold)
            label_title.setText(key)
            label_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            label_value = QLabel()
            label_value.setFont(normal_font)
            label_value.setText(info_dict[key])
            label_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            # add the value label as an attribute to allow external modification
            attribute_name = f'{prefix}{self.to_attribute_name(key)}'
            setattr(self, attribute_name, label_value)

            hbox = QHBoxLayout()
            hbox.setContentsMargins(5, 0, 0, 0)
            hbox.addWidget(label_title, 1)
            hbox.addWidget(label_value, 1)
            vbox.addLayout(hbox)


    def _create_hbox(self, title: str, title_font: QtGui.QFont,
                     content: str, content_font: QtGui.QFont) -> QHBoxLayout:
        title_label = QLabel()
        title_label.setFont(title_font)
        title_label.setText(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        content_label = QLabel()
        content_label.setFont(content_font)
        content_label.setText(content)
        content_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(5, 0, 0, 0)
        hbox.addWidget(title_label, 1)
        hbox.addWidget(content_label, 1)
        return hbox


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Optional[Config] = None):

        super().__init__()
        self.config = config
        self.top = 150
        self.left = 150
        self.width = 1100
        self.height = 700

        self.title = 'Super Mario Bros AI - GA vs DQN'
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.current_generation = 0
        self._true_zero_gen = 0
        self._should_display = True

        # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
        # index                0  1     2       3      4  5  6  7  8
        self.keys = np.array([0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
        # We need a mapping from the output to the keys above
        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }

        # Initialize agents and environments
        self.init_agents()

        # Then the GUI
        self.init_gui()

        # Start the update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(1000 // 60)  # 60 FPS

        self.show()

    def init_agents(self):
        # GA initialization
        individuals = self._initialize_population()
        self.population = Population(individuals)
        self.mario_GA = self.population.individuals[0]
        self.best_fitness_GA = 0.0
        self.max_distance_GA = 0
        self.current_generation = 0
        self._current_individual = 0
        self._true_zero_gen = 0

        self.env_GA = self._make_env()

        # Determine the size of the next generation
        if self.config.Selection.selection_type == 'plus':
            self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
        elif self.config.Selection.selection_type == 'comma':
            self._next_gen_size = self.config.Selection.num_offspring
        else:
            raise Exception(f'Unknown selection type: {self.config.Selection.selection_type}')

        # DQN initialization
        self.mario_DQN = DQNMario(self.config)
        self.best_fitness_DQN = 0.0
        self.max_distance_DQN = 0
        self.dqn_episodes = 0

        self.env_DQN = self._make_env()

        # Step counters
        self.total_steps_GA = 0
        self.total_steps_DQN = 0

    def init_gui(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        
        # Layouts
        self.main_layout = QtWidgets.QHBoxLayout(self.centralWidget)

        # Info Widget
        self.info_window = InformationWidget(self.centralWidget, (512, 200), self.config)
        self.info_window.setObjectName('info_window')
        
        # GA Widgets
        self.ga_game_window = GameWindow(self.centralWidget, (512, 448), self.config)
        self.ga_game_window.setObjectName('ga_game_window')
        self.ga_viz_window = Visualizer(self.centralWidget, (512, 448), self.config, NeuralNetworkViz(self.centralWidget, self.mario_GA, (512, 448), self.config))
        self.ga_viz_window.setObjectName('ga_viz_window')

       

        
        # DQN Widgets
        self.dqn_game_window = GameWindow(self.centralWidget, (512, 448), self.config)
        self.dqn_game_window.setObjectName('dqn_game_window')
        self.dqn_viz_window = Visualizer(self.centralWidget, (512, 448), self.config, NeuralNetworkViz(self.centralWidget, self.mario_DQN, (512, 448), self.config))
        self.dqn_viz_window.setObjectName('dqn_viz_window')

        
        # Add widgets to layouts
        self.ga_layout = QtWidgets.QVBoxLayout()
        self.ga_layout.addWidget(self.ga_game_window)
        self.ga_layout.addWidget(self.ga_viz_window)
        
        self.dqn_layout = QtWidgets.QVBoxLayout()
        self.dqn_layout.addWidget(self.dqn_game_window)
        self.dqn_layout.addWidget(self.dqn_viz_window)
        
        self.main_layout.addLayout(self.ga_layout)
        self.main_layout.addLayout(self.dqn_layout)
        self.main_layout.addWidget(self.info_window) 

    def _make_env(self):
        return retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.Misc.level}')

    def _update(self):
        # Update GA agent
        self._update_ga()
        # Update DQN agent
        self._update_dqn()
        # Refresh GUI components
        self.ga_game_window._update()
        self.ga_viz_window._update()
        self.dqn_game_window._update()
        self.dqn_viz_window._update()


    def _update_ga(self):
        
        ram = self.env_GA.get_ram()
        tiles = SMB.get_tiles(ram)

         # Update the GA agent to get the output
        self.mario_GA.update(ram, tiles, self.keys, self.ouput_to_keys_map)

        # Take a step in the environment
        ret = self.env_GA.step(self.mario_GA.buttons_to_press)
        self.total_steps_GA += 1

        # Get new ram information
        ram = self.env_GA.get_ram()
        tiles = SMB.get_tiles(ram)
        enemies = SMB.get_enemy_locations(ram)

        # Update the GUI
        self._update_GUI(self.ga_game_window, self.ga_viz_window, ret, ram, tiles, enemies)

        # Update the stats - 
        if self.mario_GA.is_alive:

            # Check for new farthest distance
            if self.mario_GA.farthest_x > self.max_distance_GA:
                self.max_distance_GA = self.mario_GA.farthest_x

        else:
            # Once agent dies,
            # Check for new best fitness
            self.mario_GA.calculate_fitness()
            if self.mario_GA.fitness > self.best_fitness_GA:
                self.best_fitness_GA = self.mario_GA.fitness

            # Handle end of individual or generation
            self._current_individual += 1

            # !!! Implement replay functionality

            # Checks whether it's time to transition to a new generation
            if self._current_individual >= len(self.population.individuals):
                self._next_generation()
                
            if args.no_display:
                self.env_GA.reset()
            else:
                self.ga_game_window.screen = self.env_GA.reset()
            
            self.mario_GA = self.population.individuals[self._current_individual]

            # Perform comprehensive label refresh
            self._update_labels_GA()


    def _update_dqn(self):

        ram = self.env_GA.get_ram()
        tiles = SMB.get_tiles(ram)

        # Update the DQN agent to get the output
        self.mario_DQN.update(ram, tiles, self.keys, self.ouput_to_keys_map)

        # Take a step in the environment
        ret = self.env_DQN.step(self.mario_DQN.buttons_to_press)
        self.total_steps_DQN += 1

        # Get new ram information
        ram = self.env_DQN.get_ram()
        tiles = SMB.get_tiles(ram)
        enemies = SMB.get_enemy_locations(ram)

        # Get current state
        curr_state = self.mario_DQN.get_state(ram, tiles)
        action = self.mario_DQN.buttons_to_press

        # Update the GUI 
        self._update_GUI(self.dqn_game_window, self.dqn_viz_window, ret, ram, tiles, enemies)

        # Get next state
        next_ram = self.env_DQN.get_ram()
        next_tiles = SMB.get_tiles(next_ram)
        next_state = self.mario_DQN.get_state(next_ram, next_tiles)

        # Calculate reward
        reward = self.mario_DQN.calculate_reward(curr_state, next_state)
        done = not self.mario_DQN.is_alive

        # Experience replay buffer
        self.mario_DQN.replay_buffer.add(curr_state, action, next_state, reward, done)

        # Perform learning step
        self.mario_DQN.learn()

        # Update stats
        self.mario_DQN.fitness += reward
        if self.mario_DQN.is_alive:
            if self.mario_DQN.farthest_x > self.max_distance_DQN:
                self.max_distance_DQN = self.mario_DQN.farthest_x
            if self.mario_DQN.fitness > self.best_fitness_DQN:
                self.best_fitness_DQN = self.mario_DQN.fitness
        else:
            # Episode ended
            self.dqn_episodes += 1
            self.env_DQN.reset()
            self.mario_DQN.reset()
            self.dqn_viz_window.nn_viz.mario = self.mario_DQN
        
        # Refresh labels with new information
        self._update_labels_DQN()

    def _next_generation(self):
        self.current_generation += 1
        self._current_individual = 0

        self._save_ga_info()

        # GA selection
        self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)
        random.shuffle(self.population.individuals)

        next_pop = []

        # Decrement lifespan and carry over individuals
        if self.config.Selection.selection_type == 'plus':
            # Decrement lifespan
            for individual in self.population.individuals:
                individual.lifespan -= 1

            next_pop = self.individual_to_agent(self.population.individuals)

        while len(next_pop) < self._next_gen_size:
            # Perform crossover and mutation to generate new offspring
            selection = self.config.Crossover.crossover_selection
            if selection == 'tournament':
                p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
            elif selection == 'roulette':
                p1, p2 = roulette_wheel_selection(self.population, 2)
            else:
                raise Exception(f'Unknown crossover selection: {selection}')

            # Perform SBX crossover and Gaussian mutation
            c1_params, c2_params = self._crossover_and_mutate(p1, p2)

            c1 = Mario(self.config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
            c2 = Mario(self.config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

            # Adds two children from the crossover to the next population
            next_pop.extend([c1, c2])

        # Finally, update the individuals to this new population
        self.population.individuals = next_pop

    def _crossover_and_mutate(self, p1, p2):
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

        return c1_params, c2_params

    def keyPressEvent(self, event):
        k = event.key()
        modifier = int(event.modifiers())
        if modifier == Qt.CTRL:
            if k == Qt.Key_P:
                if self._timer.isActive():
                    self._timer.stop()
                else:
                    self._timer.start(1000 // 60)

    def _initialize_population(self):
        individuals: List[Individual] = []
        num_parents = self.config.Selection.num_parents
        for _ in range(num_parents):
            individual = Mario(self.config)
            individuals.append(individual)
        return individuals
    

    def _save_ga_info(self):
        if self.config.Statistics.save_best_individual_from_generation:
            folder = self.config.Statistics.save_best_individual_from_generation
            best_ind_name = 'best_ind_gen{}'.format(self.current_generation - 1)
            best_ind = self.population.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        if self.config.Statistics.save_population_stats:
            fname = self.config.Statistics.save_population_stats
            save_stats(self.population, fname)

    def individual_to_agent(self, population):
        agents = []
        for individual in population:
            config = individual.config
            chromosome = individual.network.params
            hidden_layer_architecture = individual.hidden_layer_architecture
            hidden_activation = individual.hidden_activation
            output_activation = individual.output_activation
            lifespan = individual.lifespan
            name = individual.name

            if lifespan > 0:
                agent = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, lifespan)
                agents.append(agent)
        return agents
    
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

    def _update_GUI(self, game_window, viz_window, ret, ram, tiles, enemies):

        # Update Game Window
        if not args.no_display:
            if self._should_display:
                # Update GUI components
                game_window.screen = ret[0]
                game_window._should_update = True    
            else:
                # Unless it is hidden
                game_window._should_update = False
            game_window._update()
        
        # Update Info Window
        if not args.no_display:
            self.info_window.show()
        else:
            self.info_window.hide()

        # Update Viz Window
        if not args.no_display:
            if self._should_display:
                viz_window.ram = ram
                viz_window.tiles = tiles
                viz_window.enemies = enemies
                viz_window._should_update = True
                viz_window.ram = ram
            else:
                viz_window._should_update = False
            viz_window._update()

        # Update NN Viz
        if not args.no_display:
            viz_window.nn_viz.mario = self.mario_GA

    def _update_labels_GA(self):
        # Dynamic 
        if not args.no_display:
            self.info_window.ga_generation.setText(str(self.current_generation))

            current_pop = self._next_gen_size if self.current_generation == self._true_zero_gen else self.config.Selection.num_offspring
            self.info_window.ga_individual.setText('{}/{}'.format(self._current_individual + 1, current_pop))

            self.info_window.ga_best_fitness.setText('{:.2f}'.format(self.best_fitness_GA))
            self.info_window.ga_max_distance.setText(str(self.max_distance_GA))
            self.info_window.ga_total_steps.setText(str(self.total_steps_GA))
        
        ### Static 
        # ga_info = self.info_window.get_info_ga()
        # self.info_window.ga_num_inputs.setText(ga_info["Num Inputs"])
        # self.info_window.ga_trainable_params.setText(ga_info{"Trainable Params"})
        # self.info_window.ga_offspring.setText(ga_info["Offspring"])
        # self.info_window.ga_lifespan.setText(ga_info["Lifespan"])
        # self.info_window.ga_mutation.setText(ga_info["Mutation"])
        # self.info_window.ga_crossover.setText(ga_info["Crossover"])
        # self.info_window.ga_sbx_eta.setText(ga_info["SBX Eta"])
        # self.info_window.ga_layers.setText(ga_info["Layers"])

    def _update_labels_DQN(self):
        # Dynamic
        if not args.no_display:
            self.info_window.dqn_individual.setText(str(self.dqn_episodes))
            self.info_window.dqn_best_fitness.setText('{:.2f}'.format(self.best_fitness_DQN))
            self.info_window.dqn_max_distance.setText(str(self.max_distance_DQN))
            self.info_window.dqn_total_steps.setText(str(self.total_steps_DQN))
        ### Static 
        # dqn_info = self.info_window.get_dqn_info()
        # self.info_window.dqn_learning_rate.setText(dqn_info["Learning Rate"])
        # self.info_window.dqn_layers.setText(dqn_info["Layers"])
        

    

if __name__ == "__main__":
    global args
    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())





