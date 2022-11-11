# Importations
import pyautogui as pg
import time, datetime
import keyboard  # utile pour taper certain caractères comme le "#" qui ne sont pas supportés par pyautogui
import matplotlib.pyplot as plt
from collections import namedtuple,  deque
import random
import torch
from pathlib import Path
from agent import CupHead
from metric_logger import MetricLogger
from environment import CupHeadEnvironment
import os


CUR_DIR_PATH = Path(__file__).resolve()
save_dir = os.path.join(CUR_DIR_PATH, "save")


use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

# Hyperparameters



# transforms and environment

env = CupHeadEnvironment()


# 




save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

cuphead = CupHead(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = cuphead.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        cuphead.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = cuphead.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step)
