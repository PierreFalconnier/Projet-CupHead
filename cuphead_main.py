# Importations
import pyautogui as pg
import time, datetime
import keyboard  # utile pour taper certain caractères comme le "#" qui ne sont pas supportés par pyautogui
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
from agent import CupHead
from metric_logger import MetricLogger
from environment import CupHeadEnvironment
import os, sys
from pprint import pprint 

pg.PAUSE = 0

CONTROLS_ENABLED  = True                    # Mettre False pour des tests sans utiliser le jeu
TRAINING          = True                    # Entrainer ou inference
LOGGING           = TRAINING and False       # Logger pendant entrainement 
RESUME_CHECKPOINT = TRAINING and False      # Reprendre un entrainement

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CHECKPOINT_PATH = Path("checkpoints") / "2022-12-19T17-22-29"


if TRAINING:
    # logging
    if LOGGING:
        save_dir.mkdir(parents=True)
        logger = MetricLogger(save_dir)
    # env

    # ACTION_LIST  = [["right"],["right"],["left"],["left"],["z"],["z","right"],["shiftleft"],["z","left"],["s"]]   # s correspond à 'still', cuphead ne fait rien
    # HOLD_TIMINGS = [0.1,       0.6,     0.6,        0.1,  0.1,      0.4  ,         0.1,        0.4    , 0.2]
    
    ACTION_LIST  = [["right"],["z","right"],["shiftleft"]]   # s correspond à 'still', cuphead ne fait rien
    HOLD_TIMINGS = [0.6,           0.4,      0.6]
    
    FORWARD_ACTION_INDEX_LIST = [0,1,5,6]
    SCREEN_WIDTH, SCREEN_HEIGHT = pg.size() 
    RESIZE_H = 256
    RESIZE_W = 256
    DIM_STATE = 3
    EPISODE_TIME_LIMITE = 180
    REWARD_DICT = {
        'Health_point_lost': -20,
        'GameWin' :          100,
        'GameOver' :         -20,
        'Forward':           0.1,
        }
    # agent
    EXPLORATION_RATE_DECAY = 0.999986137152479   # 0.5 proba reached after 50 000 steps
    EXPLORATION_RATE_MIN = 0.1
    BATCH_SIZE = 48
    GAMMA = 0.92
    BURNIN = 1e3  # min. steps before training
    LEARN_EVERY = 2  # no. of steps between updates to Q_online
    SYNC_EVERY = 1e3  # no. of steps between Q_target & Q_online sync
    LEARNING_RATE = 0.015
    DEVICE = "cpu"
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Save config
    dict_config = {
        'ACTION_LIST' : ACTION_LIST,   
        'HOLD_TIMINGS'  : HOLD_TIMINGS,
        'FORWARD_ACTION_INDEX_LIST' : FORWARD_ACTION_INDEX_LIST,
        'SCREEN_WIDTH': SCREEN_WIDTH,
        'SCREEN_HEIGHT' : SCREEN_HEIGHT,
        'RESIZE_H' : RESIZE_H,
        'RESIZE_W' : RESIZE_W,
        'DIM_STATE' : DIM_STATE,
        'CONTROLS_ENABLED' : CONTROLS_ENABLED,  
        'EPISODE_TIME_LIMITE' : EPISODE_TIME_LIMITE,
        'REWARD_DICT' : REWARD_DICT,
        'EXPLORATION_RATE_DECAY' : EXPLORATION_RATE_DECAY,   
        'EXPLORATION_RATE_MIN' :EXPLORATION_RATE_MIN,
        'BATCH_SIZE' : BATCH_SIZE,
        'GAMMA' : GAMMA,
        'BURNIN' : BURNIN ,
        'LEARN_EVERY' : LEARN_EVERY ,
        'SYNC_EVERY' : SYNC_EVERY  ,
        'LEARNING_RATE' : LEARNING_RATE,
        'DEVICE' : DEVICE,
    }
    if LOGGING : torch.save(dict_config, save_dir / 'dict_config.pt')


else:   # INFERENCE
    # Load config dictionary
    dict_config = torch.load(os.path.join(CHECKPOINT_PATH /'dict_config.pt'))
    print("Config loaded :")
    pprint(dict_config)
    # env
    ACTION_LIST = dict_config['ACTION_LIST']    
    HOLD_TIMINGS = dict_config['HOLD_TIMINGS'] 
    FORWARD_ACTION_INDEX_LIST = dict_config['FORWARD_ACTION_INDEX_LIST']
    SCREEN_WIDTH = dict_config['SCREEN_WIDTH'] 
    SCREEN_HEIGHT = dict_config['SCREEN_HEIGHT'] 
    RESIZE_H = dict_config['RESIZE_H']  
    RESIZE_W = dict_config['RESIZE_W']  
    DIM_STATE = dict_config['DIM_STATE'] 
    CONTROLS_ENABLED = dict_config['CONTROLS_ENABLED'] 
    EPISODE_TIME_LIMITE = dict_config['EPISODE_TIME_LIMITE'] 
    REWARD_DICT = dict_config['REWARD_DICT']  
    # agent
    EXPLORATION_RATE_DECAY = 0
    EXPLORATION_RATE_MIN = 0
    BATCH_SIZE = dict_config['BATCH_SIZE'] 
    GAMMA = dict_config['GAMMA']  
    BURNIN = dict_config['BURNIN']   
    LEARN_EVERY = dict_config['LEARN_EVERY']
    SYNC_EVERY = dict_config['SYNC_EVERY']
    LEARNING_RATE = dict_config['LEARNING_RATE'] 
    DEVICE = dict_config['DEVICE']
    # DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ENVIRONMENT 
env = CupHeadEnvironment(
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT,
    resize_h=RESIZE_H,
    resize_w=RESIZE_W,
    dim_state=DIM_STATE,
    controls_enabled=CONTROLS_ENABLED,
    episode_time_limite=EPISODE_TIME_LIMITE,
    reward_dict=REWARD_DICT,
    actions_list=ACTION_LIST,
    hold_timings=HOLD_TIMINGS,
    forward_action_index_list=FORWARD_ACTION_INDEX_LIST,
    )

# AGENT
cuphead = CupHead(
    state_dim=(env.dim_state, env.resize_h, 
    env.resize_w), 
    action_dim=env.actions_dim, 
    logging=LOGGING,
    save_dir=save_dir,
    exploration_rate_decay= EXPLORATION_RATE_DECAY,
    burnin=BURNIN,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
    )

# Load model
if not(TRAINING) or RESUME_CHECKPOINT:
    STAT_DICT_MODEL_PATH = CHECKPOINT_PATH / 'model_cuphead_stat_dict.pt' 
    cuphead.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))

# Start
episodes = 10000
previous_loss = None

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
        # print("Go on the game...")
        # time.sleep(5)

if TRAINING : 
    print("Training time !")
else:
    print("Playing time !")

if not CONTROLS_ENABLED : print("CONTROLS DISABLED")

for e in range(episodes):

    if CONTROLS_ENABLED :
        pg.keyUp('x')
    start_time = time.time()
    prev_frame_time = time.time()
    temps = 0
    state = env.reset_episode()
    if CONTROLS_ENABLED :
        pg.keyDown('x')

    # Play the game!
    while True:  
        # s = time.time()

        # Run agent on the state
        action_idx = cuphead.act(state)

        # Agent performs action
        next_state, reward, done = env.step(action_idx, temps=temps)
        
        # Remember
        if TRAINING:
            cuphead.cache(state, next_state, action_idx, reward, done)   # 80 bytes

        # Learn
            q, loss = cuphead.learn()

        # Logging
        if LOGGING:
            logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # print(cuphead.curr_step)
        # print(s-time.time())

        # # # Vérifications en direct
        # # os.system('clear')
        # print("------------")
        # dir  = {'Step':cuphead.curr_step,'q':q,'Loss':loss,'Action':env.actions_list[action_idx],'Reward':reward,}
        # pprint.pprint(dir,width=1)
        # # print(f'Step {cuphead.curr_step} - q {q} - Loss {loss}')
        # # print(f'Action {env.actions_list[action_idx]} - Reward {reward}')

        # Check if end of game
        if done:
            # print(f'Step {cuphead.curr_step} - q {q} - Loss {loss}')
            # print(f'Action {env.actions_list[action_idx]} - Reward {reward}')
            break

        temps = time.time()-start_time

    if TRAINING: 
        if LOGGING:logger.log_episode(env.last_progress)

        if e % 5 == 0 and LOGGING:
            print(f"Logging episode {e}")
            logger.record(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step, progress= env.last_progress)
            if loss and previous_loss and loss<previous_loss :                                      # sauve que si amélioration
                torch.save(cuphead.net.state_dict(), os.path.join(save_dir,'model_cuphead_stat_dict.pt'))
        
        if loss:
            previous_loss = loss



