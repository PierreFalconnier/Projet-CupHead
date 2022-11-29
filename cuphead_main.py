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



# CUR_DIR_PATH = Path(__file__).resolve()
# save_dir = os.path.join(CUR_DIR_PATH, "save")

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

pg.PAUSE = 0

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
logger = MetricLogger(save_dir)

# Hyperparameters

# SCREEN_SHOT_WIDTH = 960
# SCREEN_SHOT_HEIGHT =  540
SCREEN_SHOT_WIDTH = 1920
SCREEN_SHOT_HEIGHT = 1080
RESIZE_H = 128
RESIZE_W = 128
DIM_STATE = 2
CONTROLS_ENABLED = True  # Mettre False pour des tests sans utiliser le jeu
EPISODE_TIME_LIMITE = 180

REWARD_DICT = {
    'Health_point_lost':-10,
    'GameWin' : 100,
    'GameOver' : -20,
    'Forward': 0.1
    }


env = CupHeadEnvironment(
    screen_shot_width=SCREEN_SHOT_WIDTH,
    screen_shot_height=SCREEN_SHOT_HEIGHT,
    resize_h=RESIZE_H,
    resize_w=RESIZE_W,
    dim_state=DIM_STATE,
    controls_enabled=CONTROLS_ENABLED,
    episode_time_limite=EPISODE_TIME_LIMITE,
    reward_dict=REWARD_DICT,
    )

EXPLORATION_RATE_DECAY = 0.999965   # 0.5 proba reached after 20 000 steps
EXPLORATION_RATE_MIN = 0.1
BATCH_SIZE = 32
GAMMA = 0.7
BURNIN = 100  # min. steps before training
LEARN_EVERY = 3  # no. of steps between updates to Q_online
SYNC_EVERY = 1e2  # no. of steps between Q_target & Q_online sync

cuphead = CupHead(
    state_dim=(env.dim_state, env.resize_h, 
    env.resize_w), 
    action_dim=env.actions_dim, 
    save_dir=save_dir,
    exploration_rate_decay= EXPLORATION_RATE_DECAY,
    burnin=BURNIN,
    )

# Load previous model

STAT_DICT_MODEL_PATH = 'checkpoints/2022-11-29T13-38-03/model_cuphead_stat_dict.pt' 
cuphead.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))

# Start
episodes = 1000
previous_loss = None

print("Training time !")

for e in range(episodes):

    if CONTROLS_ENABLED :
        pg.keyUp('x')
    start_time = time.time()
    prev_frame_time = time.time()
    temps = 0
    current_hp = 3
    state = env.reset_episode()
    if CONTROLS_ENABLED :
        pg.keyDown('x')

    # Play the game!
    while True:  

        # Run agent on the state
        action_idx = cuphead.act(state)

        # Agent performs action
        next_state, reward, done, current_hp = env.step(action_idx, current_hp=current_hp, temps=temps)

        # Remember
        cuphead.cache(state, next_state, action_idx, reward, done)

        # Learn
        q, loss = cuphead.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        plt.figure()
        plt.imshow(state[0].numpy())
        plt.show()
        exit()

        # # Vérifications en dirxect
        # print("------------")
        # print("Step ",cuphead.curr_step,"q ",q,"Loss ",loss)
        # print("Action ",env.actions_list[action_idx],"Reward ", reward,"Current HP ", current_hp)

        # Check if end of game
        if done:
            # print("Step ",cuphead.curr_step,"q ",q,"Loss ",loss)
            # print("Action ",env.actions_list[action_idx],"Reward ", reward,"Current HP ", current_hp)
            break

        temps = time.time()-start_time

    logger.log_episode()

    if e % 5 == 0:
        logger.record(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step)
        if loss and previous_loss and loss<previous_loss :                                      # sauve que si amélioration
            torch.save(cuphead.net.state_dict(), os.path.join(save_dir,'model_cuphead_stat_dict.pt'))
    
    if loss:
        previous_loss = loss



