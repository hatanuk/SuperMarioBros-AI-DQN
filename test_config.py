import unittest
import dill
import os
from config import Config, SerializableFunction

class TestConfigPickle(unittest.TestCase):

    def setUp(self):
        # Create a sample config file
        self.config_filename = 'test_config.ini'
        with open(self.config_filename, 'w') as f:
            f.write("""
            [Graphics]
            tile_size = (32, 32)
            neuron_radius = 5.0

            [Statistics]
            save_best_individual_from_generation = True
            save_population_stats = True

            [NeuralNetwork]
            input_dims = (10, 10)
            hidden_layer_architecture = (64, 64)
            hidden_node_activation = relu
            output_node_activation = softmax
            encode_row = True

            [NeuralNetworkDQN]
            input_dims = (10, 10)
            hidden_layer_architecture = (64, 64)
            hidden_node_activation = relu
            output_node_activation = softmax
            encode_row = True
            learning_rate = 0.001

            [DQN]
            reward_func = lambda x: x + 1

            [GeneticAlgorithm]
            fitness_func = lambda x: x * 2

            [Crossover]
            probability_sbx = 0.9
            sbx_eta = 1.0
            crossover_selection = tournament
            tournament_size = 3

            [Mutation]
            mutation_rate = 0.01
            mutation_rate_type = gaussian
            gaussian_mutation_scale = 0.1

            [Selection]
            num_parents = 2
            num_offspring = 4
            selection_type = tournament
            lifespan = 5.0

            [Misc]
            level = 1-1
            allow_additional_time_for_flagpole = True
            """)

    def test_pickle_config(self):
        config = Config(self.config_filename)
        pickled_config = dill.dumps(config)
        unpickled_config = dill.loads(pickled_config)
        self.assertEqual(config._config_dict, unpickled_config._config_dict)

    def test_pickle_serializable_function(self):
        func = lambda x: x + 1
        serializable_func = SerializableFunction(func)
        pickled_func = dill.dumps(serializable_func)
        unpickled_func = dill.loads(pickled_func)
        self.assertEqual(serializable_func(2), unpickled_func(2))

    def tearDown(self):
        os.remove(self.config_filename)

if __name__ == '__main__':
    unittest.main()
