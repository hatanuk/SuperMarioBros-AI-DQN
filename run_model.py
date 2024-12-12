import retro
import torch
import numpy as np
import argparse
import time
from mario_torch import MarioTorch as Mario
from config import Config
from smb_ai import parse_args
from mario_torch import SequentialModel
import os
from DQN_algorithm.DQNbaseline import InputSpaceReduction


class Agent:
    def __init__(self, model):
        self.model = model
    
    def get_action(self, obs):
        x = torch.from_numpy(obs).float()
        if next(self.model.parameters()).is_cuda:
            x = x.to('cuda')
        output = self.model.forward(x)
        output = output.detach().cpu().numpy().flatten()
        action_index = np.argmax(output)
        return action_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved Mario .pt model')
    parser.add_argument('--level', type=str, default="1-1", help='Which level to load')
    parser.add_argument('--config', type=str, default='settings.config', help='Path to config')
    parser.add_argument('--arch', type=str, default='ga', help='Whether to use the DQN or GA NN architecture defined in config. Only matter if they differ.')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model at {args.model_path}. Provide the generated .pt file.")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Could not find config file at {args.config}.")
    else:
        config = Config(args.config)

    try:
        model = SequentialModel.load(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)


    if args.arch.lower() == 'ga':
        input_dims = config.NeuralNetworkGA.input_dims
        encode_row = config.NeuralNetworkGA.encode_row
    else:
        input_dims = config.NeuralNetworkDQN.input_dims
        encode_row = config.NeuralNetworkDQN.encode_row

    agent = Agent(model)
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{args.level}', render_mode='human')
    env = InputSpaceReduction(env, input_dims, encode_row, skip=args.Misc.skip_frames)
    obs = env.reset()

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, rewards, done, infos = env.step(action)
        time.sleep(0.03) 

    env.close()

