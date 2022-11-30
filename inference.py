# Importations
import pyautogui as pg
import time
import matplotlib.pyplot as plt
import random
import torch
from agent import CupHead
from environment import CupHeadEnvironment
import os


# CUR_DIR_PATH = Path(__file__).resolve()
# save_dir = os.path.join(CUR_DIR_PATH, "save")

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

pg.PAUSE = 0

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
    'Health_point_lost': -10,
    'GameWin' :          100,
    'GameOver' :         -20,
    'Forward':           0.1,
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

cuphead = CupHead(
    state_dim=(env.dim_state, env.resize_h, 
    env.resize_w), 
    action_dim=env.actions_dim, 
    exploration_rate_decay= 0,
    )

# Load previous model

STAT_DICT_MODEL_PATH = 'checkpoints/2022-11-29T13-38-03/model_cuphead_stat_dict.pt' 
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





