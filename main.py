# Importations
import pyautogui as pg
import time, datetime
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
from agent import CupHead
from metric_logger import MetricLogger
from environment import CupHeadEnvironment
import os, sys
from pprint import pprint 
from tqdm import tqdm
from glob import glob
from collections import deque
import threading
import psutil
import gc
from pympler import asizeof
from torch.utils.tensorboard import SummaryWriter


pg.PAUSE = 0

#---- BOOLEANS 

CONTROLS_ENABLED     = True                    # Mettre False pour des tests sans utiliser le jeu
CONSTANT_SHOOTING    = True
LEARN_DURING_EPISODE = False                # Si False, learn entre les épisodes -> pas de ralentissements de cuphead 
RESUME_CHECKPOINT    = True      # Reprendre un entrainement
LOAD_MEMORY          = False
LOGGING              = True       # Logger pendant entrainement 


#---- PATHS

# save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# CHECKPOINT_PATH = Path("checkpoints/2022-12-26T23-04-01")

save_dir = Path("checkpoints") / "test_pierre"
CHECKPOINT_PATH = save_dir

# l = glob('checkpoints/*')
# l.sort()
# CHECKPOINT_PATH = Path(l[-1])
# print('CHECKPOINT_PATH : ',CHECKPOINT_PATH) ; time.sleep(2)

#---- HYPER PARAMETERS 

# logging
if LOGGING:
    save_dir.mkdir(parents=True, exist_ok=True)
    # logger = MetricLogger(save_dir)
    writer = SummaryWriter(save_dir/"log_dir")

# env
ACTION_LIST  = [["r"],["r"],["l"],["l"],["a"],["a","r"],["r","d"],["a","l"],["l","d"]]   # s correspond à 'still', cuphead ne fait rien
HOLD_TIMINGS = [0.1,   0.6,  0.6,  0.1,  0.1,    0.4  ,  0.1,        0.4   , 0.1,    ]
FORWARD_ACTION_INDEX_LIST = [0,1,5,6]
BACKWARD_ACTION_INDEX_LIST = [2,3,7,8]
SCREEN_WIDTH, SCREEN_HEIGHT = pg.size() 
RESIZE_H = 256
RESIZE_W = 256
DIM_STATE = 3                   # correspond en fait à la dim du stack
EPISODE_TIME_LIMITE = 180
REWARD_DICT = {
    'Health_point_lost': -50,
    'GameWin' :          100,
    'GameOver' :         -50,
    'Forward':           1,
    'Backward':         -1,
    }
# agent
if Path.exists(CHECKPOINT_PATH / 'epsilon.pt' ):
    print('Loading epsilon')
    EXPLORATION_RATE_INIT = torch.load(CHECKPOINT_PATH / 'epsilon.pt')
else:
    EXPLORATION_RATE_INIT = 0.99
EXPLORATION_RATE_DECAY = 0.99999
EXPLORATION_RATE_MIN = 0.1
BATCH_SIZE = 32
GAMMA = 0.9
USE_MOBILENET = True
BURNIN = 0  # min. eps or steps before training
LEARN_EVERY = 1  # no. of eps or steps between updates to Q_online. 1 = every episodes
SYNC_EVERY = 2  # no. of eps or steps between Q_target & Q_online sync
LEARNING_RATE = 0.00025
DEVICE = "cpu"       # attention, pour les tensors, par pour le model ni entrainement
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device used for training


#---- AGENT
cuphead = CupHead(
    state_dim=(DIM_STATE, RESIZE_H, RESIZE_W), 
    action_dim=len(ACTION_LIST), 
    logging=LOGGING,
    save_dir=save_dir,
    exploration_rate_init=EXPLORATION_RATE_INIT,
    exploration_rate_decay= EXPLORATION_RATE_DECAY,
    exploration_rate_min=EXPLORATION_RATE_MIN,
    burnin=BURNIN,
    learning_rate=LEARNING_RATE,
    device=DEVICE,
    learn_during_episode=LEARN_DURING_EPISODE,
    use_mobilenet=USE_MOBILENET,
    gamma=GAMMA,
    batch_size=BATCH_SIZE,
    sync_every=SYNC_EVERY
    )

#---- ENVIRONMENT 
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


#--- LOAD VARIABLES

# Load model
if RESUME_CHECKPOINT:
    STAT_DICT_MODEL_PATH = CHECKPOINT_PATH / 'model_stat_dict.pt' 
    if Path.exists(STAT_DICT_MODEL_PATH):
        # print('Loading model')
        cuphead.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))

# Load memory
if LOAD_MEMORY and Path.exists(CHECKPOINT_PATH / 'memory.pt' ):
    # print('Loading memory')
    cuphead.memory = torch.load(CHECKPOINT_PATH / 'memory.pt' )

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
    cuphead.exploration_rate = epsilon


#---- START
episodes = 1
previous_loss = None

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

    # print('New episode')
    pg.keyDown('enter')
    time.sleep(0.1)
    pg.keyUp('enter')

    while True:  
        # start = time.time()

        # Run agent on the state
        action_idx = cuphead.act(state)

        # Agent performs action
        next_state, reward, done = env.step(action_idx, temps=temps)
        
        # Remember
        cuphead.cache(state, next_state, action_idx, reward, done)
        # Update state
        state = next_state

        # Learn during episode
        if LEARN_DURING_EPISODE:
            q, loss = cuphead.learn()

            if LOGGING :
                    pass  # à faire 

        # Liste rewards
        if LOGGING and LEARN_DURING_EPISODE == False   : 
            reward_list.append(reward)

        # Check if end of episode
        if done:
        
            if LEARN_DURING_EPISODE:
                cuphead.curr_ep = e
                break
            else:
                loss,q = None, None
                loss_list = []
                q_list = []

                cuphead.curr_ep = episode

                if os.path.exists(save_dir / 'memory.pt' ):
                        print("Loading old memories...")
                        old_memory = torch.load(save_dir / 'memory.pt' ) 
                        old_memory.extend(cuphead.memory)
                        cuphead.memory = old_memory

                if episode >= BURNIN:

                    cuphead.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device used for training
                    cuphead.net.to(device=cuphead.device)
                    cuphead.net.train()

                    print('Learning...')
                    for _ in tqdm(range(min(len(cuphead.memory),200))):
                        q, loss = cuphead.learn()
                        loss_list.append(loss)
                        q_list.append(q)

                cuphead.device = "cpu"
                cuphead.net.eval()
                cuphead.net.to(device=cuphead.device)
                
                if LOGGING : 
                    if len(loss_list) and len(q_list): 
                        loss = sum(loss_list)/len(loss_list)
                        q = sum(q_list)/len(q_list)
                        writer.add_scalar('loss', loss, episode)
                        writer.add_scalar('q', q, episode)
                    reward_mean = sum(reward_list)/len(reward_list)
                    writer.add_scalar('mean_reward', reward_mean, episode)
                    writer.add_scalar('epsilon', cuphead.exploration_rate, episode)
                    writer.add_scalar('progress', env.last_progress, episode)

                print('Saving Cuphead memory...')
                torch.save(cuphead.memory, save_dir / 'memory.pt' )
                
                cuphead.memory = deque(maxlen=5000)

                print(f"Episode {episode} - Mean reward {reward_mean} - Mean Loss {loss} - Mean q {q} - Epsilon {cuphead.exploration_rate}")



            break
        
        # print(time.time()-start)

        temps = time.time()-start_time
            

        # Loggings
         
        if LEARN_DURING_EPISODE and LOGGING : 
            pass  # a faire
        else:
            # logger.record_2(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step,  progress = env.last_progress, loss = loss, reward_mean=reward_mean)
            # logger.init_episode()

            # Saving useful variables : model, episode, epsilon
            episode = episode +1 
            torch.save(cuphead.net.state_dict(), save_dir /'model_stat_dict.pt')
            torch.save(cuphead.exploration_rate, save_dir / 'epsilon.pt')
            torch.save(episode, save_dir / 'episode.pt')


        # if loss and previous_loss and loss<previous_loss :                                      # sauve que si amélioration
        #     torch.save(cuphead.net.state_dict(), os.path.join(save_dir,'model_stat_dict.pt'))
        # if loss:
        #     previous_loss = loss
      





