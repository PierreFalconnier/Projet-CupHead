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


pg.PAUSE = 0

env = CupHeadEnvironment()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

cuphead = CupHead(state_dim=(env.dim_state, env.resize_h, env.resize_w), action_dim=env.actions_dim, save_dir=save_dir)
logger = MetricLogger(save_dir)

episodes = 1000
print("Go on the game !...")
time.sleep(5)
print("Training time !")


for e in range(episodes):

    pg.keyUp('x')
    start_time = time.time()
    prev_frame_time = time.time()
    temps = 0
    current_hp = 3
    state = env.reset_episode()
    pg.keyDown('x')

    c=0
    # Play the game!
    while True:  

        # Run agent on the state
        action_idx = cuphead.act(state)

        # Agent performs action
        next_state, reward, done, current_hp = env.step(action_idx, current_hp=current_hp)

        # Remember
        cuphead.cache(state, next_state, action_idx, reward, done)

        # Learn
        q, loss = cuphead.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state


        # Vérifications en direct
        print("------------")
        print("Steap ",cuphead.curr_step,"q ",q,"Loss ",loss)
        print(env.actions_list[action_idx],reward, current_hp)
        # if c%5==0:
        #     for k in range(state.shape[0]):
        #         plt.figure()
        #         plt.imshow(state[k])
        #         plt.show()

        # Check if end of game
        if done:
            break

        # # Calcul FPS

        # new_frame_time = time.time()
        # fps = 1/(new_frame_time-prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = str(int(fps))
        # os.system('clear')
        # print("FPS : ",fps)
        # print("Current HP : ",current_hp)

        # temps = time.time()-start_time

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=cuphead.exploration_rate, step=cuphead.curr_step)
