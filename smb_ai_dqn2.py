import re
import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

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

from DQN_algorithm.DQN import DQNMario

from smb_ai import draw_border, parse_args

import multiprocessing
import queue

import atexit


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
        _, mario_col = SMB.get_mario_row_col(self.ram)
        x = mario_col

        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)

        start_row, viz_width, viz_height = self.config.NeuralNetworkGA.input_dims

        rect_x = int(x * self.tile_width + 5 + self.x_offset)
        rect_y = int(start_row * self.tile_height + 5)
        rect_width = int(viz_width * self.tile_width)
        rect_height = int(viz_height * self.tile_height)

        painter.drawRect(rect_x, rect_y, rect_width, rect_height)



    def draw_tiles(self, painter: QPainter):
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = int(5 + (self.tile_width * col) + self.x_offset)
                y_start = int(5 + (self.tile_height * row))

                loc = (row, col)
                tile = self.tiles[loc]

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
                    color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass

                painter.drawRect(x_start, y_start, int(self.tile_width), int(self.tile_height))


    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            if self._should_update:
                draw_border(painter, self.size)
                if self.ram is not None:
                    self.draw_tiles(painter)
                    self._draw_region_of_interest(painter)
                    self.nn_viz.show_network(painter)
            else:
                painter.setPen(QColor(0, 0, 0))
                painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
                txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
                painter.drawText(event.rect(), Qt.AlignCenter, txt)
        finally:
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
        # Hopefully this should stop complaining now
        painter = QPainter(self)
        try:
            if self._should_update:
                draw_border(painter, self.size)
                if self.screen is not None:
                    height, width, channel = self.screen.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(self.screen.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    self.img_label.setPixmap(QPixmap.fromImage(qImg))
            else:
                painter.setPen(QColor(0, 0, 0))
                painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
                txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
                painter.drawText(event.rect(), Qt.AlignCenter, txt)
        finally:
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
        hidden = self.config.NeuralNetworkGA.hidden_layer_architecture
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

        learning_rate_txt = str(self.config.DQN.learning_rate)

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
        dqn_info_dict = self.get_dqn_info()
        self.dqn_vbox = QVBoxLayout()
        self.dqn_vbox.setContentsMargins(0, 0, 0, 0)
        self._create_info_section(dqn_info_dict, self.dqn_vbox, prefix='dqn_')

        self.grid.addLayout(self.dqn_vbox, 0, 1) #2nd column
        

    @staticmethod
    def to_attribute_name(input_string):
        input_string = input_string.replace(" ", "_")
        cleaned_string = re.sub(r'[^a-zA-Z_]', '', input_string)
        return cleaned_string.lower()


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

    #QWidget override
    def closeEvent(self, event):
        self.ga_process.terminate()
        self.dqn_process.terminate()
        self.ga_process.join()
        self.dqn_process.join()
        event.accept()

    def init_agents(self):
        # Queues for inter-process communication
        self.ga_data_queue = multiprocessing.Queue()
        self.dqn_data_queue = multiprocessing.Queue()

        # Start the GA process
        self.ga_process = multiprocessing.Process(target=run_ga_agent, args=(self.config, self.ga_data_queue))
        self.ga_process.start()

        # Start the DQN process
        self.dqn_process = multiprocessing.Process(target=run_dqn_agent, args=(self.config, self.dqn_data_queue))
        self.dqn_process.start()

        # multiprocessing cleanup
        def cleanup():
            self.ga_process.terminate()
            self.dqn_process.terminate()
            self.ga_data_queue.close()
            self.dqn_data_queue.close()

        atexit.register(cleanup)

    def init_gui(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        
        # Layouts
        self.main_layout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.algo_container = QtWidgets.QHBoxLayout(self.centralWidget)

        # Info Widget
        self.info_window = InformationWidget(self.centralWidget, (512, 200), self.config)
        self.info_window.setObjectName('info_window')
        self.info_window.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        
        # GA Widgets
        self.ga_game_window = GameWindow(self.centralWidget, (512, 448), self.config)
        self.ga_game_window.setObjectName('ga_game_window')
        self.ga_viz_window = Visualizer(self.centralWidget, (512, 448), self.config, NeuralNetworkViz(self.centralWidget, None, (512, 448), self.config, nn_params=self.config.NeuralNetworkGA))
        self.ga_viz_window.setObjectName('ga_viz_window')

        
        # DQN Widgets
        self.dqn_game_window = GameWindow(self.centralWidget, (512, 448), self.config)
        self.dqn_game_window.setObjectName('dqn_game_window')
        self.dqn_viz_window = Visualizer(self.centralWidget, (512, 448), self.config, NeuralNetworkViz(self.centralWidget, None, (512, 448), self.config, nn_params=self.config.NeuralNetworkDQN))
        self.dqn_viz_window.setObjectName('dqn_viz_window')

        
        # Add widgets to layouts
        self.ga_layout = QtWidgets.QHBoxLayout()
        self.ga_layout.addWidget(self.ga_viz_window)
        self.ga_layout.addWidget(self.ga_game_window)

        self.dqn_layout = QtWidgets.QHBoxLayout()
        self.dqn_layout.addWidget(self.dqn_viz_window)
        self.dqn_layout.addWidget(self.dqn_game_window)

        self.algo_container.addLayout(self.ga_layout)
        self.algo_container.addLayout(self.dqn_layout)

        self.main_layout.addLayout(self.algo_container)
        self.main_layout.addWidget(self.info_window) 


    def _update(self):
        # Get data from GA agent
        try:
            while True:
                ga_data = self.ga_data_queue.get_nowait()
                if not args.no_display:
                    if self._should_display:
                        self.ga_game_window.screen = ga_data['screen']
                        self.ga_game_window._should_update = True    
                    else:
                        self.ga_game_window._should_update = False
                    self.ga_game_window._update()

                if not args.no_display:
                    self.info_window.show()
                else:
                    self.info_window.hide()

                if not args.no_display:
                    if self._should_display:
                        self.ga_viz_window.ram = ga_data['ram']
                        self.ga_viz_window.tiles = ga_data['tiles']
                        self.ga_viz_window.enemies = ga_data['enemies']
                        self.ga_viz_window.nn_viz.mario = ga_data['mario'] 
                        self.ga_viz_window._should_update = True
                        self.ga_viz_window.ram = ga_data['ram']
                    else:
                        self.ga_viz_window._should_update = False
                    self.ga_viz_window._update()

                self.info_window.ga_best_fitness.setText('{:.2f}'.format(ga_data['best_fitness']))
                self.info_window.ga_max_distance.setText(str(ga_data['max_distance']))
                self.info_window.ga_total_steps.setText(str(ga_data['total_steps']))
                self.info_window.ga_generation.setText(str(ga_data['current_generation']))
                self.info_window.ga_individual.setText(str(ga_data['current_individual']))

        except queue.Empty:
            pass

        # Get data from DQN agent
        try:
            while True:
                dqn_data = self.dqn_data_queue.get_nowait()
                if not args.no_display:
                    if self._should_display:
                        self.dqn_game_window.screen = dqn_data['screen']
                        self.dqn_game_window._should_update = True    
                    else:
                        self.dqn_game_window._should_update = False
                    self.dqn_game_window._update()

                if not args.no_display:
                    if self._should_display:
                        self.dqn_viz_window.ram = dqn_data['ram']
                        self.dqn_viz_window.tiles = dqn_data['tiles']
                        self.dqn_viz_window.enemies = dqn_data['enemies']
                        self.dqn_viz_window.nn_viz.mario = dqn_data['mario'] 
                        self.dqn_viz_window._should_update = True
                        self.dqn_viz_window.ram = dqn_data['ram']
                    else:
                        self.dqn_viz_window._should_update = False
                    self.dqn_viz_window._update()

                self.info_window.dqn_best_fitness.setText('{:.2f}'.format(dqn_data['best_fitness']))
                self.info_window.dqn_max_distance.setText(str(dqn_data['max_distance']))
                self.info_window.dqn_total_steps.setText(str(dqn_data['total_steps']))
                self.info_window.dqn_individual.setText(str(dqn_data['dqn_episodes']))

        except queue.Empty:
            pass

    def keyPressEvent(self, event):
        k = event.key()
        modifier = int(event.modifiers())
        if modifier == Qt.CTRL:
            if k == Qt.Key_P:
                if self._timer.isActive():
                    self._timer.stop()
                else:
                    self._timer.start(1000 // 60)


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
    _true_zero_gen = 0
    total_steps_GA = 0

    # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
    # index                0  1     2       3      4  5  6  7  8
    keys = np.array([0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

    # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
    # We need a mapping from the output to the keys above
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

                    c1 = Mario(config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
                    c2 = Mario(config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

                    # Adds two children from the crossover to the next population
                    next_pop.extend([c1, c2])

                # Finally, update the individuals to this new population
                population.individuals = next_pop

            # Reset the environment
            env.reset()
            mario_GA = population.individuals[_current_individual]

        # Prepare data to send back
        data = {
            'screen': ret[0],
            'ram': ram,
            'tiles': tiles,
            'enemies': enemies,
            'best_fitness': best_fitness_GA,
            'max_distance': max_distance_GA,
            'total_steps': total_steps_GA,
            'current_generation': current_generation,
            'current_individual': _current_individual + 1,
            'mario': mario_GA
        }

        # Send data to main process
        data_queue.put(data)

def run_dqn_agent(config, data_queue):
    # Initialize environment
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{config.Misc.level}')

    # Initialize DQN agent
    mario_DQN = DQNMario(config)
    best_fitness_DQN = 0.0
    max_distance_DQN = 0
    dqn_episodes = 0
    total_steps_DQN = 0

    # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
    # index                0  1     2       3      4  5  6  7  8
    keys = np.array([0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

    # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
    # We need a mapping from the output to the keys above
    ouput_to_keys_map = {
        0: 4,  # U
        1: 5,  # D
        2: 6,  # L
        3: 7,  # R
        4: 8,  # A
        5: 0   # B
    }

    # Reset environment
    env.reset()

    while True:
        ram = env.get_ram()
        tiles = SMB.get_tiles(ram)
        enemies = SMB.get_enemy_locations(ram)

        curr_stats = get_stats(mario_DQN) # gets the x distance, frames, score, etc. for reward calculation
        curr_state = mario_DQN.inputs_as_array

        # Update the DQN agent to get the output
        mario_DQN.update(ram, tiles, keys, ouput_to_keys_map)

        # Take a step in the environment
        ret = env.step(mario_DQN.buttons_to_press)
        total_steps_DQN += 1

        next_stats = get_stats(mario_DQN) # do not confuse this with state
        next_state = mario_DQN.inputs_as_array

        # Calculate reward

        reward = mario_DQN.calculate_reward(curr_stats, next_stats)
        done = not mario_DQN.is_alive


        # Experience replay buffer
        mario_DQN.replay_buffer.append((curr_state, mario_DQN.buttons_to_press, next_state, reward, done))

        # Perform learning step
        mario_DQN.learn()

        if mario_DQN.is_alive:
            if mario_DQN.farthest_x > max_distance_DQN:
                max_distance_DQN = mario_DQN.farthest_x
            if mario_DQN.fitness > best_fitness_DQN:
                best_fitness_DQN = mario_DQN.fitness
        else:
            # Episode ended
            dqn_episodes += 1
            env.reset()

        # Prepare data to send back
        data = {
            'screen': ret[0],
            'ram': ram,
            'tiles': tiles,
            'enemies': enemies,
            'best_fitness': best_fitness_DQN,
            'max_distance': max_distance_DQN,
            'total_steps': total_steps_DQN,
            'dqn_episodes': dqn_episodes,
            'mario': mario_DQN
        }

        # Send data to main process
        data_queue.put(data)

def _initialize_population(config):
    individuals: List[Individual] = []
    num_parents = config.Selection.num_parents
    for _ in range(num_parents):
        individual = Mario(config)
        individuals.append(individual)
    return individuals

def individual_to_agent(population, config):
    agents = []
    for individual in population:
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

def _crossover_and_mutate(p1, p2, config, current_generation):
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
        eta = config.Crossover.sbx_eta
        c1_W_l, c2_W_l = SBX(p1_W_l, p2_W_l, eta)
        c1_b_l, c2_b_l =  SBX(p1_b_l, p2_b_l, eta)

        # Mutation
        # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
        mutation_rate = config.Mutation.mutation_rate
        scale = config.Mutation.gaussian_mutation_scale

        if config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(current_generation + 1)

        # Mutate weights
        gaussian_mutation(c1_W_l, mutation_rate, scale=scale)
        gaussian_mutation(c2_W_l, mutation_rate, scale=scale)

        # Mutate bias
        gaussian_mutation(c1_b_l, mutation_rate, scale=scale)
        gaussian_mutation(c2_b_l, mutation_rate, scale=scale)

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

def get_stats(mario):
        # returns the game's stats for reward calculation
        frames = mario._frames if mario._frames is not None else 0
        distance = mario.x_dist if mario.x_dist is not None else 0
        game_score = mario.game_score if mario.game_score is not None else 0
        return [frames, distance, game_score]

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)
    sys.stdout = sys.stderr
    print("test")

    global args
    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())
