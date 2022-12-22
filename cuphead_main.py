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

pg.PAUSE = 0

CONTROLS_ENABLED     = True                    # Mettre False pour des tests sans utiliser le jeu
CONSTANT_SHOOTING    = True
TRAINING             = True                    # Entrainer ou inference
TRAINING_AND_PLAYING = TRAINING and True
TRAINING_ON_MEMORY   = TRAINING and not(TRAINING_AND_PLAYING)
RESUME_CHECKPOINT    = TRAINING and False      # Reprendre un entrainement
LOGGING              = TRAINING and True       # Logger pendant entrainement 
dict_bool = {'TRAINING':TRAINING ,'TRAINING_AND_PLAYING':TRAINING_AND_PLAYING,'TRAINING_ON_MEMORY':TRAINING_ON_MEMORY,'RESUME_CHECKPOINT':RESUME_CHECKPOINT,'LOGGING':LOGGING,'CONTROLS_ENABLED':CONTROLS_ENABLED, 'CONSTANT_SHOOTING':CONSTANT_SHOOTING}
pprint(dict_bool) 

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CHECKPOINT_PATH = Path("checkpoints") / "2022-12-22T15-46-06"


if TRAINING:
    # logging
    if LOGGING:
        save_dir.mkdir(parents=True)
        logger = MetricLogger(save_dir)
    # env
    # MODIFIER les controls du jeu si conflie avec les touches pour naviguer dans le menu, sinon bugs
    ACTION_LIST  = [["r"],["r"],["l"],["l"],["a"],["a","r"],["d"],["a","l"],["s"]]   # s correspond à 'still', cuphead ne fait rien
    HOLD_TIMINGS = [0.1,   0.6,  0.6,  0.1,  0.1,    0.4  ,  0.1,    0.4   , 0.2]
    # ACTION_LIST  = [["r"],["a","r"],["d"]]   # "a" sauter, "r" droite, "l" gauche, "d" dash, "s" ne rien faire
    # HOLD_TIMINGS = [0.6,           0.4,      0.6]
    FORWARD_ACTION_INDEX_LIST = [0,1,5,6]
    BACKWARD_ACTION_INDEX_LIST = [2,3,7]
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
        'Backward':         -0.1,
        }
    # agent
    EXPLORATION_RATE_DECAY = 0.999986137152479   # 0.5 proba reached after 50 000 steps
    EXPLORATION_RATE_MIN = 0.1
    BATCH_SIZE = 32
    GAMMA = 0.9
    LEARN_DURING_EPISODE = False                # Si False, learn entre les épisodes -> pas de ralentissements de cuphead 
    BURNIN = 0  # min. eps or steps before training
    LEARN_EVERY = 1  # no. of eps or steps between updates to Q_online. 1 = every episodes
    SYNC_EVERY = 10  # no. of eps or steps between Q_target & Q_online sync
    LEARNING_RATE = 0.001
    DEVICE = "cpu"       
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device used for training
    # Save config
    dict_config = {
        'ACTION_LIST' : ACTION_LIST,   
        'HOLD_TIMINGS'  : HOLD_TIMINGS,
        'FORWARD_ACTION_INDEX_LIST' : FORWARD_ACTION_INDEX_LIST,
        'BACKWARD_ACTION_INDEX_LIST': BACKWARD_ACTION_INDEX_LIST,
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
        'LEARN_DURING_EPISODE' : LEARN_DURING_EPISODE,
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
    BACKWARD_ACTION_INDEX_LIST = dict_config['BACKWARD_ACTION_INDEX_LIST']
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
    LEARN_DURING_EPISODE = dict_config['LEARN_DURING_EPISODE']
    # DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ENVIRONMENT 
if not(TRAINING_ON_MEMORY):
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
        backward_action_index_list=BACKWARD_ACTION_INDEX_LIST,
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
    learn_during_episode=LEARN_DURING_EPISODE,
    )

# Load model
if not(TRAINING) or RESUME_CHECKPOINT:
    STAT_DICT_MODEL_PATH = CHECKPOINT_PATH / 'model_cuphead_stat_dict.pt' 
    cuphead.net.load_state_dict(torch.load(STAT_DICT_MODEL_PATH))

# # Load memory
# cuphead.memory = torch.load(CHECKPOINT_PATH / 'cuphead_memory.pt' )

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
      
if TRAINING : 
    print("Training time !")
else:
    print("Playing time !")

if not CONTROLS_ENABLED : print("CONTROLS DISABLED")

if TRAINING_AND_PLAYING or TRAINING==False:     # Si apprentissage avec jeu ou inférence
    for e in range(episodes):

        if CONTROLS_ENABLED and CONSTANT_SHOOTING :
            pg.keyUp('x')
        start_time = time.time()
        prev_frame_time = time.time()
        temps = 0
        step = cuphead.curr_step
        state = env.reset_episode()
        if CONTROLS_ENABLED and CONSTANT_SHOOTING:
            pg.keyDown('x')

        # Play the game!
        pg.press('enter')
        while True:  
            # Run agent on the state
            action_idx = cuphead.act(state)

            # Agent performs action
            next_state, reward, done = env.step(action_idx, temps=temps)
            
            # Remember
            if TRAINING:
                cuphead.cache(state, next_state, action_idx, reward, done)   # 80 bytes

            # Update state
            state = next_state

            # Learn during episode
            if LEARN_DURING_EPISODE:
                q, loss = cuphead.learn()

                if LOGGING :
                        logger.log_step(reward, q, reward)


            # # Vérifications en direct
            # print("------------")
            # # print(f'Step {cuphead.curr_step} - q {q} - Loss {loss}')
            # # print(f'Action {env.actions_list[action_idx]} - Reward {reward}')

            # Check if end of episode
            if done:
            
                if LEARN_DURING_EPISODE == False   : # learn between episodes
                    loss = None
                    cuphead.curr_ep = e
                    cuphead.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device used for training
                    cuphead.net.to(device=cuphead.device)
                    if e >= BURNIN:
                        print('Learning...')
                        for _ in tqdm(range(min(len(cuphead.memory),5000))):
                            q, loss = cuphead.learn()
                    cuphead.device = "cpu"
                    cuphead.net.to(device=cuphead.device)
            
                break

            temps = time.time()-start_time
            

        # Loggings
        if TRAINING: 
            if LEARN_DURING_EPISODE and LOGGING : 
                logger.log_episode(env.last_progress)
                if e % 1 == 0 and LOGGING:
                    print(f"Logging episode {e}")
                    logger.record(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step, progress= env.last_progress)
                logger.init_episode()
            else:
                print(f"Logging episode {e}")
                logger.record_2(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step,  progress = env.last_progress, loss = loss,)
                logger.init_episode()

            if loss and previous_loss and loss<previous_loss :                                      # sauve que si amélioration
                torch.save(cuphead.net.state_dict(), os.path.join(save_dir,'model_cuphead_stat_dict.pt'))
            if loss:
                previous_loss = loss
            
            if e % 1 == 0 :
                print('Saving Cuphead memory...')
                torch.save(cuphead.memory,os.path.join(save_dir,'cuphead_memory.pt'))
                print('Done')

elif TRAINING_ON_MEMORY:
    # CHECKPOINT_PATH = 
    print('Training on memory')
    EPOCH = 5
    cuphead.learning_rate = 0.001
    cuphead.batch_size = 256
    cuphead.sync_every = 5
    cuphead.memory = torch.load(CHECKPOINT_PATH / 'cuphead_memory.pt' ) 
    cuphead.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device used for training
    cuphead.net.to(device=cuphead.device)
    print('Learning...')
    for epoch in range(EPOCH):
        loss_list = []
        cuphead.curr_ep = epoch
        print(f'Epoch : {epoch}')
        for _ in tqdm(range(min(len(cuphead.memory),5000))):
            q, loss = cuphead.learn()
            loss_list.append(loss)
        print(sum(loss_list)/len(loss_list))
    cuphead.device = "cpu"
    cuphead.net.to(device=cuphead.device)




