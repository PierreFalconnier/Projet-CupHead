#---- IMPORTATIONS
import pyautogui as pg
import time, datetime
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
import os, sys
from pprint import pprint 
from tqdm import tqdm
from glob import glob
from collections import deque
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from mario import MarioNet
from mario import MetricLogger
from mario import Mario
from mario import SkipFrame,GrayScaleObservation, ResizeObservation, FrameStack

#---- BOOLEANS 

LOGGING = True
RESUME_CHECKPOINT = True
LOAD_MEMORY = False
LEARN_DURING_EPISODE = False                # Si False, learn entre les épisodes -> pas de ralentissements de cuphead 

#---- PATHS

# save_dir = Path("generic") / "checkpoints" / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

save_dir = Path("generic") / "checkpoints" / "test_3"
CHECKPOINT_PATH = save_dir
if LOGGING:
    writer = SummaryWriter(save_dir/"log_dir")

# l = glob('checkpoints/*')
# l.sort()
# CHECKPOINT_PATH = Path(l[-1])
# print('CHECKPOINT_PATH : ',CHECKPOINT_PATH) ; time.sleep(2)


#---- HYPER PARAMETERS 

# logging
save_dir.mkdir(parents=True, exist_ok=True)
    # logger = MetricLogger(save_dir)

# env
RESIZE_H = 84
RESIZE_W = 84
DIM_STACK = 3
# agent
EXPLORATION_RATE_INIT = 1
EXPLORATION_RATE_DECAY = 0.99999
EXPLORATION_RATE_MIN = 0.1
BATCH_SIZE = 32
GAMMA = 0.9
BURNIN = 300  # min. eps or steps (if learn while playing) before training
LEARN_EVERY = 3  # if learn while playing, no. of steps between updates to Q_online. 1 = every episodes
SYNC_EVERY = 1000  # no. of eps or steps (if learn while playing) between Q_target & Q_online sync
LEARNING_RATE = 0.00025
DEVICE = "cpu"       # attention, pour les tensors, par pour le model ni entrainement


#---- ENVIRONMENT AND TRANSFORMS

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

# Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)
    # env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

# Limit the action-space to : walk right, jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
# print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# Apply Wrappers to environment (transforms)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=RESIZE_H)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=DIM_STACK, new_step_api=True)
else:
    env = FrameStack(env, num_stack=DIM_STACK)


#---- AGENT

mario = Mario(
    state_dim=(DIM_STACK, RESIZE_H, RESIZE_W), 
    action_dim=env.action_space.n,
    save_dir=save_dir,
    exploration_rate_init=EXPLORATION_RATE_INIT,
    exploration_rate_decay= EXPLORATION_RATE_DECAY,
    exploration_rate_min=EXPLORATION_RATE_MIN,
    burnin=BURNIN,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
    gamma=GAMMA,
    batch_size=BATCH_SIZE,
    sync_every=SYNC_EVERY,
    learn_during_episode = LEARN_DURING_EPISODE,
    )

# print('Agent : ')
# pprint(vars(mario)) 


#---- LOADINGS 

# Load model
if RESUME_CHECKPOINT:
    STAT_DICT_MODEL_PATH = CHECKPOINT_PATH / 'model_stat_dict.pt' 
    if Path.exists(STAT_DICT_MODEL_PATH):
        # print('Loading model')
        mario.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))

# Load memory
if LOAD_MEMORY and Path.exists(CHECKPOINT_PATH / 'memory.pt' ):
    # print('Loading memory')
    mario.memory = torch.load(CHECKPOINT_PATH / 'memory.pt' )

# Load episode
if not(LEARN_DURING_EPISODE) and Path.exists(CHECKPOINT_PATH / 'episode.pt' ):
    # print('Loading episode')
    episode = torch.load(CHECKPOINT_PATH / 'episode.pt' )

if not(LEARN_DURING_EPISODE) and not(Path.exists(CHECKPOINT_PATH / 'episode.pt' )):
    episode = 0

# Load epsilon
if not(LEARN_DURING_EPISODE) and Path.exists(CHECKPOINT_PATH / 'epsilon.pt' ):
    # print('Loading epsilon')
    epsilon = torch.load(CHECKPOINT_PATH / 'epsilon.pt' )
    mario.exploration_rate = epsilon



#---- START

if LEARN_DURING_EPISODE == False:
    episodes_in_one_run = 1
else:
    episodes_in_one_run = 50000

# for e in tqdm(range(episodes_in_one_run)):
for e in tqdm(range(episodes_in_one_run)):

    reward_list = []
    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)


        if LEARN_DURING_EPISODE:
            # Learn
            q, loss = mario.learn()

            # Logging
            if LOGGING :
                writer.add_scalar('reward', reward, e)
                if loss: writer.add_scalar('loss', loss, e)
                if q   : writer.add_scalar('q', q, e)
                writer.add_scalar('epsilon', mario.exploration_rate, e)
        else:
            reward_list.append(reward)


        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:

            if LEARN_DURING_EPISODE:
                break
            else:

                loss,q = None, None
                loss_list = []
                q_list = []

                if os.path.exists(save_dir / 'memory.pt' ):
                        print("Loading old memory...")
                        old_memory = torch.load(save_dir / 'memory.pt' ) 
                        old_memory.extend(mario.memory)
                        mario.memory = old_memory

                if episode >= BURNIN:

                    # Switch to GPU
                    mario.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device used for training
                    mario.net.to(device=mario.device)
                    mario.net.train()

                    print('Learning...')
                    for _ in tqdm(range(min(len(mario.memory),200))):
                        q, loss = mario.learn()
                        loss_list.append(loss)
                        q_list.append(q)

                    mario.device = "cpu"
                    mario.net.eval()
                    mario.net.to(device=mario.device)
                    
                if LOGGING : 
                    if len(loss_list) and len(q_list): 
                        loss = sum(loss_list)/len(loss_list)
                        q = sum(q_list)/len(q_list)
                        writer.add_scalar('loss', loss, episode)
                        writer.add_scalar('q', q, episode)
                    reward_mean = sum(reward_list)/len(reward_list)
                    writer.add_scalar('mean_reward', reward_mean, episode)
                    writer.add_scalar('epsilon', mario.exploration_rate, episode)


                print('Saving memory...')
                torch.save(mario.memory, save_dir / 'memory.pt' )
                mario.memory = deque(maxlen=5000)

                print(f"Episode {episode} - Mean reward {reward_mean} - Mean Loss {loss} - Mean q {q} - Epsilon {mario.exploration_rate}")

                

            break



if not(LEARN_DURING_EPISODE):

    if episode % SYNC_EVERY == 0:
        mario.sync_Q_target()

    # Saving useful variables : model, episode, epsilon
    episode = episode +1 
    torch.save(mario.net.state_dict(), save_dir /'model_stat_dict.pt')
    torch.save(mario.exploration_rate, save_dir / 'epsilon.pt')
    torch.save(episode, save_dir / 'episode.pt')
    
