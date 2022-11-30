# Importations
import pyautogui as pg
import time
import matplotlib.pyplot as plt
import random
import torch
from agent import CupHead
from environment import CupHeadEnvironment
import os
from pathlib import Path


# CUR_DIR_PATH = Path(__file__).resolve()
# save_dir = os.path.join(CUR_DIR_PATH, "save")

# Load config dictionary

CHECKPOINT_PATH = Path("checkpoints") / "2022-11-30T01-29-39"
# CHECKPOINT_PATH = Path("checkpoints") / "2022-11-29T13-38-03"
dict_config = torch.load(os.path.join(CHECKPOINT_PATH /'dict_config.pt'))

ACTION_LIST= dict_config['ACTION_LIST']
HOLD_TIMINGS= dict_config['HOLD_TIMINGS']
SCREEN_SHOT_WIDTH= dict_config['SCREEN_SHOT_WIDTH']
SCREEN_SHOT_HEIGHT= dict_config['SCREEN_SHOT_HEIGHT']
RESIZE_H= dict_config['RESIZE_H']
RESIZE_W= dict_config['RESIZE_W']
DIM_STATE = dict_config['DIM_STATE']
CONTROLS_ENABLED = True  
EPISODE_TIME_LIMITE = dict_config['EPISODE_TIME_LIMITE']
REWARD_DICT = dict_config['REWARD_DICT']
EXPLORATION_RATE_DECAY = 0   
EXPLORATION_RATE_MIN = 0
# DEVICE = dict_config['DEVICE']
DEVICE = 'cpu'
pg.PAUSE = 0

# Environment and Agent

env = CupHeadEnvironment(
    screen_shot_width=SCREEN_SHOT_WIDTH,
    screen_shot_height=SCREEN_SHOT_HEIGHT,
    resize_h=RESIZE_H,
    resize_w=RESIZE_W,
    dim_state=DIM_STATE,
    controls_enabled=CONTROLS_ENABLED,
    episode_time_limite=EPISODE_TIME_LIMITE,
    reward_dict=REWARD_DICT,
    actions_list=ACTION_LIST,
    hold_timings=HOLD_TIMINGS,
    )

cuphead = CupHead(
    state_dim=(env.dim_state, env.resize_h, 
    env.resize_w), 
    action_dim=env.actions_dim, 
    exploration_rate_decay= EXPLORATION_RATE_DECAY,
    exploration_rate_min=EXPLORATION_RATE_MIN,
    device=DEVICE,
    )

# Load previous model

STAT_DICT_MODEL_PATH = CHECKPOINT_PATH / 'model_cuphead_stat_dict.pt' 
cuphead.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))

# Start
episodes = 1000
previous_loss = None

if CONTROLS_ENABLED:
    if os.name == 'nt':
        import pygetwindow as gw
        window = gw.getWindowsWithTitle('Cuphead')[-1]
        window.restore()
    else:
        print("LINUX")
        print("Go on the game...")
        time.sleep(5)

print("Playing time !")

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
        next_state, _, done, current_hp = env.step(action_idx, current_hp=current_hp, temps=temps)

        # Remember
        # cuphead.cache(state, next_state, action_idx, reward, done)

        # Learn
        # q, loss = cuphead.learn()

        # Logging
        # logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # plt.figure()
        # plt.imshow(state[0].numpy())
        # plt.show()
        # exit()

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





