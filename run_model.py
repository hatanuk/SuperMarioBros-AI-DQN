import retro
import torch
import numpy as np
import argparse
import time
from mario_torch import MarioTorch as Mario
from config import Config
from mario_torch import SequentialModel
from mario_torch import MarioTorch as Mario
import os
from DQN_algorithm.DQNbaseline import InputSpaceReduction



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to saved Mario .pt model')
    parser.add_argument('--level', type=str, default="1-1", help='Which level to load')
    parser.add_argument('--frame_skip', type=int, default=4, help='Specify a custom amount of frames to hold an action for')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model at {args.model_path}. Provide the generated .pt file.")
    try:
        model = SequentialModel.load(args.model_path)
        details = torch.load(args.model_path, weights_only=False)

    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    
    input_dims = details['input_dims']
    encode_row = details['encode_row']
    frame_skip = args.frame_skip

    agent = Mario(frame_skip=frame_skip, input_dims=input_dims, encode_row=encode_row)
    agent.model = model

    env = retro.make(game='SuperMarioBros-Nes', state=f'Level{args.level}', render_mode='human')
    env = InputSpaceReduction(env, input_dims, encode_row, skip=frame_skip)
    env.mario = agent
    obs = env.reset()

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, rewards, done, infos = env.step(action)
        time.sleep(1/60) 

    print(f'distance: {agent.farthest_x}')

    env.close()

