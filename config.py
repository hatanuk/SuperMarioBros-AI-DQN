import configparser
import os
from typing import Any, Dict

import ast
import operator as op

# Supported operators for safe evaluation
allowed_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def safe_eval(expr, variables):
    # Parse an expression and evaluate it safely
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Num):  # For Python versions < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # For Python 3.8 and above
            return node.value
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            if type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
            else:
                raise TypeError(f"Unsupported binary operator: {ast.dump(node.op)}")
        elif isinstance(node, ast.UnaryOp):  # -<operand> or +<operand>
            if type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(f"Unsupported unary operator: {ast.dump(node.op)}")
        elif isinstance(node, ast.Call):  # Function calls like max(), min(), int()
            func = _eval(node.func)
            args = [_eval(arg) for arg in node.args]
            if func in variables.values():
                return func(*args)
            else:
                raise NameError(f"Use of unsupported function '{func.__name__}'")
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            else:
                raise NameError(f"Use of undefined variable '{node.id}'")
        else:
            raise TypeError(f"Unsupported expression: {ast.dump(node)}")
    node = ast.parse(expr, mode='eval')
    return _eval(node.body)

# A mapping from parameters name -> final type
_params = {
    # Graphics Params
    'Graphics': {
        'tile_size': (tuple, float),
        'neuron_radius': float,
    },

    # Statistics Params
    'Statistics': {
        'save_best_individual_from_generation': str,
        'save_population_stats': str,
    },

    # NeuralNetwork Params
    'NeuralNetwork': {
        'input_dims': (tuple, int),
        'hidden_layer_architecture': (tuple, int),
        'hidden_node_activation': str,
        'output_node_activation': str,
        'encode_row': bool,
    },

    'NeuralNetworkDQN': {
        'input_dims': (tuple, int),
        'hidden_layer_architecture': (tuple, int),
        'hidden_node_activation': str,
        'output_node_activation': str,
        'encode_row': bool,
        'learning_rate': float
    },

    # Deep Q Network
    'DQN': {
    },

    # Genetic Algorithm
    'GeneticAlgorithm': {
    },

    # Crossover Params
    'Crossover': {
        'probability_sbx': float,
        'sbx_eta': float,
        'crossover_selection': str,
        'tournament_size': int,
    },

    # Mutation Params
    'Mutation': {
        'mutation_rate': float,
        'mutation_rate_type': str,
        'gaussian_mutation_scale': float,
    },

    # Selection Params
    'Selection': {
        'num_parents': int,
        'num_offspring': int,
        'selection_type': str,
        'lifespan': float
    },

    # Misc Params
    'Misc': {
        'level': str,
        'allow_additional_time_for_flagpole': bool
    }
}

class DotNotation(object):
    def __init__(self, d: Dict[Any, Any]):
        for k in d:
            # If the key is another dictionary, keep going
            if isinstance(d[k], dict):
                self.__dict__[k] = DotNotation(d[k])
            # If it's a list or tuple then check to see if any element is a dictionary
            elif isinstance(d[k], (list, tuple)):
                l = []
                for v in d[k]:
                    if isinstance(v, dict):
                        l.append(DotNotation(v))
                    else:
                        l.append(v)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = d[k]
    
    def __getitem__(self, name) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)


class Config(object):
    def __init__(self,
                 filename: str
                 ):
        self.filename = filename
        
        if not os.path.isfile(self.filename):
            raise Exception('No file found named "{}"'.format(self.filename))

        with open(self.filename) as f:
            self._config_text_file = f.read()

        self._config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._config.read(self.filename)

        self._verify_sections()
        self._create_dict_from_config()
        self._set_dict_types()
        dot_notation = DotNotation(self._config_dict)
        self.__dict__.update(dot_notation.__dict__)


    def _create_dict_from_config(self) -> None:
        d = {}
        for section in self._config.sections():
            d[section] = {}
            for k, v in self._config[section].items():
                d[section][k] = v

        self._config_dict = d

    def _set_dict_types(self) -> None:
        for section in self._config_dict:
            for k, v in self._config_dict[section].items():
                if k in ('reward_func', 'fitness_func'):
                    # Store the function expression as a string
                    self._config_dict[section][k] = v
                else:
                    # Existing type handling
                    try:
                        _type = _params[section][k]
                    except:
                        raise Exception('No value "{}" found for section "{}". Please set this in _params'.format(k, section))
                    
                    if isinstance(_type, tuple):
                        if len(_type) == 2:
                            cast = _type[1]
                            v = v.replace('(', '').replace(')', '')
                            self._config_dict[section][k] = tuple(cast(val) for val in v.split(','))
                        else:
                            raise Exception('Expected a 2-tuple value describing parsing logic')
                    elif _type == bool:
                        self._config_dict[section][k] = _type(eval(v))
                    else:
                        self._config_dict[section][k] = _type(v)


    def _verify_sections(self) -> None:
        # Validate sections
        for section in self._config.sections():
            # Make sure the section is allowed
            if section not in _params:
                raise Exception('Section "{}" has no parameters allowed. Please remove this section and run again.'.format(section))

    def _get_reference_from_dict(self, reference: str) -> Any:
        path = reference.split('.')
        d = self._config_dict
        for p in path:
            d = d[p]
        
        assert type(d) in (tuple, int, float, bool, str)
        return d

    def _is_number(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    def get_reward_func(self):
        expr = self._config_dict['DQN']['reward_func']
        def reward_func(**variables):
            # Include built-in functions like max(), min(), int() in variables
            safe_vars = {
                'max': max,
                'min': min,
                'int': int,
                **variables
            }
            return safe_eval(expr, safe_vars)
        return reward_func

    def get_fitness_func(self):
        expr = self._config_dict['GeneticAlgorithm']['fitness_func']
        def fitness_func(**variables):
            # Include built-in functions like max(), min(), int() in variables
            safe_vars = {
                'max': max,
                'min': min,
                'int': int,
                **variables
            }
            return safe_eval(expr, safe_vars)
        return fitness_func
