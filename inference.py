# Importations
import pyautogui as pg
import time, datetime
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
from agent import CupHead
from environment import CupHeadEnvironment
import os, sys
from pprint import pprint 

pg.PAUSE = 0

CONTROLS_ENABLED     = True                    # Mettre False pour des tests sans utiliser le jeu
CONSTANT_SHOOTING    = True

# save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# CHECKPOINT_PATH = Path("checkpoints/2022-12-26T23-04-01")

save_dir = Path("checkpoints") / "double_1"
CHECKPOINT_PATH = save_dir

# l = glob('checkpoints/*')
# l.sort()
# CHECKPOINT_PATH = Path(l[-1])
# print('CHECKPOINT_PATH : ',CHECKPOINT_PATH) ; time.sleep(2)

# Load config dictionary
dict_config = torch.load(os.path.join(CHECKPOINT_PATH /'dict_config.pt'))
print("Config loaded :")
pprint(dict_config) 
# env
ACTION_LIST = dict_config['ACTION_LIST']    
HOLD_TIMINGS = dict_config['HOLD_TIMINGS'] 
FORWARD_ACTION_INDEX_LIST = dict_config['FORWARD_ACTION_INDEX_LIST']
BACKWARD_ACTION_INDEX_LIST = dict_config['BACKWARD_ACTION_INDEX_LIST']
SCREEN_WIDTH = dict_config['SCREEN_WIDTH'] 
SCREEN_HEIGHT = dict_config['SCREEN_HEIGHT'] 
RESIZE_H = dict_config['RESIZE_H']  
RESIZE_W = dict_config['RESIZE_W']  
DIM_STATE = dict_config['DIM_STATE'] 
CONTROLS_ENABLED = dict_config['CONTROLS_ENABLED'] 
EPISODE_TIME_LIMITE = dict_config['EPISODE_TIME_LIMITE'] 
REWARD_DICT = dict_config['REWARD_DICT']  
USE_MOBILENET = dict_config['USE_MOBILENET'] ,
# agent
DEVICE = dict_config['DEVICE']
# DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# AGENT
cuphead = CupHead(
    state_dim=(DIM_STATE, RESIZE_H, RESIZE_W), 
    action_dim=len(ACTION_LIST), 
    logging=False,
    save_dir=save_dir,
    device=DEVICE,
    use_mobilenet=USE_MOBILENET,
    exploration_rate_init=0,
    )

# ENVIRONMENT 
env = CupHeadEnvironment(
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT,
    resize_h=RESIZE_H,
    resize_w=RESIZE_W,
    use_mobilenet = USE_MOBILENET,
    dim_state=DIM_STATE,
    controls_enabled=CONTROLS_ENABLED,
    episode_time_limite=EPISODE_TIME_LIMITE,
    reward_dict=REWARD_DICT,
    actions_list=ACTION_LIST,
    hold_timings=HOLD_TIMINGS,
    forward_action_index_list=FORWARD_ACTION_INDEX_LIST,
    backward_action_index_list=BACKWARD_ACTION_INDEX_LIST,
    )


# Load model
STAT_DICT_MODEL_PATH = CHECKPOINT_PATH / 'model_cuphead_stat_dict.pt' 
if Path.exists(STAT_DICT_MODEL_PATH):
    print('Loading model')
    cuphead.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))


# Start
episodes = 10

if CONTROLS_ENABLED:
    if os.name == 'nt':
        print("WINDOWS")
        # import pygetwindow as gw
        # window = gw.getWindowsWithTitle('Cuphead')[-1]
        # window.restore()
        print("Go on the game...")
        time.sleep(5)
    else:
        print("LINUX")
      
print("Playing time !")

if not CONTROLS_ENABLED : print("CONTROLS DISABLED")

for e in range(episodes):

    if CONTROLS_ENABLED and CONSTANT_SHOOTING :
        pg.keyUp('x')
    start_time = time.time()
    prev_frame_time = time.time()
    temps = 0
    reward_list = []
    step = cuphead.curr_step
    state = env.reset_episode()
    if CONTROLS_ENABLED and CONSTANT_SHOOTING:
        pg.keyDown('x')


    print('New episode')
    pg.keyDown('enter')
    time.sleep(0.1)
    pg.keyUp('enter')

    while True:  

        
        # start = time.time()

        # Run agent on the state
        action_idx = cuphead.act(state)

        # Agent performs action
        next_state, _, done = env.step(action_idx, temps=temps)
        
        # Update state
        state = next_state

        # Check if end of episode
        if done:
            break
        
        # print(time.time()-start)

        temps = time.time()-start_time
        




