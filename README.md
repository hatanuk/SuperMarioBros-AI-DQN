
## Introduction
This is a heavily modified fork from Chrispresso's original visualisation of a genetic algorithm (GA) learning to beat Super Mario Brothers. Check out his original video [here](https://www.youtube.com/watch?v=CI3FRsSAa_U&t=60s)!

This project implements an additional reinforcement learning (RL) agent to train alongside the GA agent. Specifically, a Deep Q-Network (DQN) is implemented, which was used back in [2016 by Deepmind to complete Atari games.](https://arxiv.org/abs/1312.5602)

My aim for this project was to compare the learning dynamics of a gradient-based approach such as RL versus an evolutionary-based approach in finding solutions for a game environment - I thought it would be neat to see whether a simple process like genetic algorithms can stay competitive against more modern ML techniques.

In practice, the GA agents seem to converge to a solution much more reliably than the DQN agents, but do take longer to train as each individual requires their own environment instance.

Demo video can be seen [here](https://youtu.be/80GpsGpEq70)!

## Data Visualization
The project includes Tensorboard logs for both agents which provide more details on the learning dynamics of the agents such as fitness, distance, loss, etc. The log rate and directory can be specified in the configs under `[Statistics]`.

![graph 1](https://media.discordapp.net/attachments/831589316820074557/1335032755376885760/Screenshot202025-02-0120at2012.png?ex=679eb1a4&is=679d6024&hm=6e066a6decd79000fee22f5d83c280b4095a624df60778267f8f7ff8405f9fd2&=&format=webp&quality=lossless&width=2210&height=786)
![graph 2](https://media.discordapp.net/attachments/831589316820074557/1335032522853056534/Screenshot202025-02-0120at2012.png?ex=679eb16d&is=679d5fed&hm=82a44975261734bc8805f1509e9f3dec095e806ee8b19b1174e6abbe1a70999d&=&format=webp&quality=lossless&width=1340&height=1228)


## Installation Instructions

[](https://github.com/hatanuk/SuperMarioBros-AI-DQN?tab=readme-ov-file#installation-instructions)

**Disclaimer:** This has only been tested to work on a Windows Linux Subsystem virtual machine running on Windows 11. Gym-retro is considered an outdated library, and compatability can be finicky. This repo uses retro-stable which is slightly more compatible.

You will need Python 3.6 or newer.

1.  `cd /path/to/SuperMarioBros-AI`
2.  Run  `pip install -r requirements.txt`
3.  Install the ROM
    -   Go read the  [disclaimer](https://wowroms.com/en/disclaimer)
    -   Head on over to the ROM for  [Super Mario Bros.](https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html)  and click  `download`.
4.  Unzip  `Super Mario Bros. (World).zip`  to some location
5.  Run  `python -m retro.import "/path/to/unzipped/super mario bros. (world)"`
    -   Make sure you run this on the folder, i.e.  `python -m retro.import "c:\Users\chris\Downloads\Super Mario Bros. (World)"`
    -   You should see output text:
        
        ```
        Importing SuperMarioBros-Nces
        Imported 1 games
        ```

## Training Agents
Specify experiment configurations in the config file (default is `settings.config`)
A custom fitness function used by both algorithms can be specified in `config.py`
Run `python train_agents.py` to obtain model weights for both GA and DQN agents (`.pt` files)

## Running Trained Agents

[](https://github.com/hatanuk/SuperMarioBros-AI-DQN?tab=readme-ov-file#running-examples)

Once an agent has been trained, you can see them play out a level of SMB using the following command:

-   `python run_model.py /path/to/model.pt --level="1-1" --frame_skip=4`

You can set level with `--level`  (eg. "1-1") and the amount of frames an input should be held for with  `--frameskip`

## Configurations

Configurations to the experiment set-up, including the architecture of the neural networks for both agents, can be altered in the `settings.config` file.

## Neural Network (GA)

Specified by `[NeuralNetworkGA]`.

-   **input_dims :Tuple.** This defines the "pink box." The parameters are `(start_row, width, height)` where `start_row` begins at 0 at the top of the screen and increases going toward the bottom. `width` defines how wide the box is, beginning at `start_row`. `height` defines how tall the box is. Currently, start_col is not supported and is set to wherever Mario is at.
-   **hidden_layer_architecture :Tuple.** Describes how many hidden nodes are in each hidden layer. `(12, 9)` would create two hidden layers, the first with 12 nodes and the second with 9. This can be any length of 1 or more.
-   **hidden_node_activation :str.** Options are `(relu, sigmoid, linear, leaky_relu, tanh)`. Defines what activation to use on hidden layers.
-   **output_node_activation :str.** Options are `(relu, sigmoid, linear, leaky_relu, tanh)`. Defines what activation to use on the output layer.
-   **encode_row :bool.** Whether or not to have one-hot encoding to describe Mario's row location.

## Neural Network (DQN)

Specified by `[NeuralNetworkDQN]`.

The same parameters apply here as above (`input_dims`, `hidden_layer_architecture`, `hidden_node_activation`, `encode_row`), with the difference that:

-   **output_node_activation :str.** For DQN, this is set to `linear` to preserve Q-values during training.

## Graphics

Specified by `[Graphics]`.

-   **tile_size :Tuple.** The size in pixels in the `(X, Y)` direction to draw the tiles on the screen.
-   **neuron_radius :float.** Radius to draw nodes on the screen.

## Environment

Specified by `[Environment]`.

-   **level :str.** The current options are `(1-1, 2-1, 3-1, 4-1, 5-1, 6-1, 7-1, 8-1)`. More can be supported by adding state information for the gym environment.
-   **frame_skip :int.** Each action is repeated for `frame_skip` frames.

## Statistics

Specified by `[Statistics]`.

-   **save_best_individual_from_generation :str.** A folder location `/path/to/save/generation` to save best individuals.
-   **save_population_stats :str.** A file location `/file/location/of/stats.csv` where you wish to save statistics.
-   **model_save_dir :str.** Directory (`./models` by default) for saving `.pt` model files.
-   **ga_model_name :str.** Base name for GA model files (e.g., `GAbest`).
-   **top_x_individuals :int.** How many of the highest-fitness individuals to save from a generation.
-   **ga_checkpoint_interval :int.** How often (in generations) to save a GA checkpoint.
-   **dqn_model_name :str.** Base name for DQN model files (e.g., `DQNbest`).
-   **dqn_checkpoint_interval :int.** How often (in episodes) to save a DQN checkpoint.
-   **tensorboard_dir :str.** Where to log TensorBoard info.
-   **log_interval :int.** Every how many steps to log progress in TensorBoard.


A model `.pt` file contains: 
- **'iterations'**: Number of episodes/generations  
- **'distance'**: Level distance covered by the agent 
- **'encode_row'**: Taken from the config 
- **'input_dims'**: Taken from the config 
- **'state_dict'**: PyTorch weights and biases 
-  **'layer_sizes'**: Tuple of input, hidden, and output layer node sizes  
- **'hidden_activation'**: String of the chosen activation function **'output_activation'**: String of the chosen activation function

Files will be saved at `model_save_dir/DQN` or `model_save_dir/GA` respectively. **WARNING**: This will clear all existing files in those directories.

## Genetic Algorithm

Specified by `[GA]`.

    
-   **total_generations :int.** Total number of generations to run.
    
-   **parallel_processes :int.** How many processes to run in parallel.
    

## DQN

Specified by `[DQN]`.

-   **learning_rate :float.** The learning rate for the DQN network.
-   **discount_value :float.** The discount factor (gamma) for Q-learning.
-   **train_freq :int.** How often (in steps) the DQN is trained.
-   **total_episodes :int.** Total number of episodes to run for training.
-   **sync_network_rate :int.** How often (in steps) to sync the target network.
-   **batch_size :int.** Mini-batch size used for experience replay.
-   **buffer_size :int.** Maximum size of the replay buffer.
-   **epsilon_start :float.** Starting epsilon value for exploration.
-   **epsilon_min :float.** Minimum epsilon value for exploration.
-   **decay_fraction :float.** Fraction of total episodes over which epsilon is decayed linearly.

## Mutation

Specified by `[Mutation]`.

-   **mutation_rate :float.** Must be between `[0.00, 1.00)`. Specifies the probability that each gene (trainable parameter) will mutate.
-   **mutation_rate_type :str.** Options are `(static, dynamic)`.
    -   **static** mutation will remain the same throughout.
    -   **dynamic** will decrease mutation_rate as the number of generations increases.
-   **gaussian_mutation_scale :float.** When a mutation occurs it is a normal Gaussian mutation `N(0, 1)`. Because the parameters are capped between `[0.00, 1.00)`, a scale is provided to narrow this. The mutation would then be `N(0, 1) * scale`.

## Crossover

Specified by `[Crossover]`.

-   **probability_sbx :float.** The probability to perform Simulated Binary Crossover (SBX). Currently always `1.0`.
-   **sbx_eta :int.** The smaller the value, the more variance when creating offspring. As the parameter increases, variance decreases and offspring are more centered around parent values. `100` still has variance but centers more around the parents, helping genes change gradually rather than abruptly.
-   **crossover_selection :str.** Options are `(roulette, tournament)`.
    -   **roulette** sums all individual fitnesses and assigns each individual a probability proportional to its fitness relative to total fitness.
    -   **tournament** will randomly pick `n` individuals from the population and then select the highest fitness individual from that subset.
-   **tournament_size :int.** Used only if `crossover_selection = tournament`. Controls how many individuals form the subset from which a winner is chosen.

## Selection

Specified by `[Selection]`.

-   **num_parents :int.** Number of individuals in the initial population.
    
-   **num_offspring :int.** Number of offspring produced at the end of each generation.
    
-   **selection_type :str.** Options are `(comma, plus)`.
    
    Let’s say we define `<num_parents> <selection_type> <num_offspring>` and compare `(50, 100)` and `(50 + 100)`:
    
    -   `(50, 100)` will begin generation 0 with 50 parents and then produce 100 offspring from those parents at the end of the generation. At generation 1, there will be 100 parents. No best individuals are carried over.
    -   `(50 + 100)` will begin generation 0 with 50 parents, produce 100 offspring, then carry over the best 50 individuals to the next generation. At generation 1, you have 150 parents. In this case, 50 individuals get carried over.
-   **lifespan :float.** Essentially an int but can be `inf`. This dictates how long a certain individual can remain in the population before dying off. For `selection_type = plus`, this means an individual only reproduces for a given number of generations before it dies. For `comma`, it doesn’t matter since no best individuals get carried over.
    

## Misc

Specified by `[Misc]`.

-   **allow_additional_time_for_flagpole :bool.** Generally as soon as Mario touches the flag, he “dies” because he wins. This allows some extra time just to see the ending animation.
