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
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model at {args.model_path}. Provide the generated .pt file.")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Could not find config file at {args.config}.")

    try:
        model = SequentialModel.load(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    agent = Agent(model)
    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{args.level}', render_mode='human')
    env = InputSpaceReduction(env, config.Graphics.input_dims, config.Graphics.encode_row)
    env = FrameSkipWrapper(env, skip=4)
    obs = env.reset()

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, rewards, done, infos = env.step(action)
        time.sleep(0.03) 

    env.close()

