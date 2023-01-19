# Importations
import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3 import PPO

from tqdm import tqdm
import time
import pyautogui as pg
from pathlib import Path
from environment import CupHeadEnvironment
import datetime

pg.PAUSE = 0

#---- Algorithms to test
# DQN with CnnPolicy

#---- BOOLEANS 

CONSTANT_SHOOTING    = True
LEARN                = True
RESUME_CHECKPOINT    = False      

#---- PATHS
date = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints")  / f"A2C-{date}"
log_path = save_dir / "logdir"
model_save_path = save_dir/ f"model-{date}"
best_model_save_path = save_dir/ f"best_model-{date}"

#---- HYPER PARAMETERS 

# env

# ACTION_LIST  = [["r"],["r"],["l"],["l"],["a"],["a","r"],["r","d"],["a","l"],["l","d"]]   
# HOLD_TIMINGS = [0.1,   0.6,  0.6,  0.1,  0.1,    0.4  ,  0.1,        0.4   , 0.1  ]
# FORWARD_ACTION_INDEX_LIST = [0,1,5,6]
# BACKWARD_ACTION_INDEX_LIST = [2,3,7,8]

ACTION_LIST  = [["r"],["r"],["l"],["l"],["a"],["a","r"],["r","d"],["a","l"],["l","d"],["s"], ["l","d"]]   # s correspond à 'still', cuphead ne fait rien
HOLD_TIMINGS = [0.1,   0.6,  0.6,  0.1,  0.1,    0.4  ,    0.1,       0.4   , 0.1,      0.1 ,  0.1,         ]
FORWARD_ACTION_INDEX_LIST = [0,1,5,6]
BACKWARD_ACTION_INDEX_LIST = [2,3,7,8, 10]

# # Pour multidiscret
# ACTION_LIST  = [["r"],["r"],["l"],["l"],["a"],["a"],["d"]]   
# HOLD_TIMINGS = [0.1,   0.6,  0.6,  0.1,  0.1,    0.4  ,  0.1    ]
# FORWARD_ACTION_INDEX_LIST = [0,1]
# BACKWARD_ACTION_INDEX_LIST = [2,3]

SCREEN_WIDTH, SCREEN_HEIGHT = pg.size() 
RESIZE_H = 512
RESIZE_W = 512
DIM_STATE = 3                   # correspond en fait à la dim du stack
EPISODE_TIME_LIMITE = 180
REWARD_DICT = {
    'Health_point_lost': -0.1,
    'GameWin' :          10,
    'GameOver' :         -0.1,
    'Forward':           0.005,
    'Backward':         -0.03*0,
    'Progress_factor':      1,      # facteur qui multiplie la progression (entre 0 et 1)
    }

env = CupHeadEnvironment(
      screen_width=SCREEN_WIDTH,
      screen_height=SCREEN_HEIGHT,
      resize_h=RESIZE_H,
      resize_w=RESIZE_W,
      use_mobilenet = False,
      dim_state=DIM_STATE,
      controls_enabled=True,
      episode_time_limite=EPISODE_TIME_LIMITE,
      reward_dict=REWARD_DICT,
      actions_list=ACTION_LIST,
      hold_timings=HOLD_TIMINGS,
      forward_action_index_list=FORWARD_ACTION_INDEX_LIST,
      backward_action_index_list=BACKWARD_ACTION_INDEX_LIST,
      stable_baseline= True,
      )

# check_env(env)

env = Monitor(env)

#---- Callback

checkpoint = CheckpointCallback(save_freq=int(5e2), save_path= model_save_path)
# eval       = EvalCallback(env, best_model_save_path= best_model_save_path,log_path= log_path, eval_freq=500, deterministic=True, render=False)  # peu faire bugger le programme si se lance au mauvais moment (ex: juste après une mort)
MyCallback = [checkpoint]

#---- Wapper TimeLimit

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
  
env = TimeLimit(env, max_episode_steps= 350)

# check_env(env)

#---- Entrainement

if LEARN:
    model_path = Path("checkpoints/test_A2C/best_model/best_model.zip")
    if RESUME_CHECKPOINT and Path.exists(model_path):
        print(f"Chargement du model : {model_path}")
        model = A2C.load(model_path, env=env)
    else:
        model = A2C("CnnPolicy",env, tensorboard_log= log_path)
        # print(vars(model))
        print(vars(model.policy)) ; exit()


  # model = DQN("CnnPolicy",
  #               env,
  #               verbose=1, 
  #               buffer_size=50000, 
  #               tensorboard_log=save_dir / "logdir", 
  #               learning_starts=25000,
  #               learning_rate=0.0004,
  #               batch_size=32,
  #               gamma=0.99,
  #               train_freq=4,
  #               target_update_interval=10000,
  #               exploration_fraction=0.5,
  #               exploration_initial_eps=1,
  #               exploration_final_eps=0.05)


  # model = PPO("CnnPolicy", env, tensorboard_log=save_dir / "logdir",)

    TIME_STEP = int(1e6)

    if CONSTANT_SHOOTING:
            pg.keyDown('x')

    model.learn(total_timesteps=TIME_STEP, log_interval=1, progress_bar=True, reset_num_timesteps=False, tb_log_name="A2C", callback= MyCallback)

    if CONSTANT_SHOOTING :
            pg.keyUp('x')
    

if not(LEARN):
  model_path = Path("checkpoints/test_A2C/best_model/best_model.zip")
  model = A2C.load(model_path)
  if CONSTANT_SHOOTING:
          pg.keyDown('x')
  obs = env.reset()
  for _ in range(10):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      # env.render()
      if done:
        obs = env.reset()