# Imports et fonctions

import pyautogui as pg
# from PIL import ImageGrab  # plus rapide pour les screenshots
# import pyscreenshot as ImageGrab
import time
import keyboard  # utile pour taper certain caractères comme le "#" qui ne sont pas supportés par pyautogui
import matplotlib.pyplot as plt
import cv2
import PIL
import numpy as np
import os
import mss
from agent import CupHead

from torchvision.transforms import ToTensor, Resize, Grayscale
from torchvision.transforms import Compose
import torch


# Maintenir une touche un certain temps
def holdKey(key,seconds=0.1):
    pg.keyDown(key)
    time.sleep(seconds)
    pg.keyUp(key)

def hold2Keys(key1,key2,seconds=0.1):
    pg.keyDown(key1)
    pg.keyDown(key2)
    time.sleep(seconds)
    pg.keyUp(key1)
    pg.keyUp(key2)

def print_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


# Actions
# actions_binds_list = ["x"    ,"z"   ,"v"      ,"shiftleft","c"   ,"right","left","up","down","tab"          ]
# actions_names_list = ["shoot","jump","exshoot","dash"     ,"lock","right","left","up","down","switch_weapon"]
# flying_actions_list =       ["x"    ,"v"      ,"z"    ,"shift"]
# flying_actions_names_list = ["shoot","special","parry","shrink"]
# action_dim = len(actions_binds_list)

# Environment class

class CupHeadEnvironment(object):

    def __init__(
        self,
        screen_shot_width = 1920,
        screen_shot_height = 1080,
        resize_w=128,
        resize_h=128,
        dim_state=1,
        controls_enabled = True,
        episode_time_limite = 180,
        ) -> None:

        # Actions
        # self.actions_list = [["right"],["left"],["up"],["down"],["s"],["z"],["shiftleft"],["z","right"],["z","left"]]   # s correspond à 'still', cuphead ne fait rien
        # self.hold_timings = [0.75,   0.75,  0.3, 0.3,   0.75,  0.75,   0.1,      0.65,          0.65]
        # self.actions_list = [["right"],["left"],["left"],["z"],["z","right"],["shiftleft"]]   # s correspond à 'still', cuphead ne fait rien
        # self.hold_timings = [0.75,   0.75,        0.1,    0.75,      0.65,        0.1]
       
        self.actions_list = [["right"],["left"],["left"],["z"],["z","right"]]   # s correspond à 'still', cuphead ne fait rien
        self.hold_timings = [0.75,   0.75,        0.1,    0.75,      0.65   ]
       
        self.actions_dim = len(self.actions_list)
        self.controls_enabled = controls_enabled   # si True, le programme utilie PyAutoGUI et controle le clavier

        # Crop variables and screen shot
        # self.mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080} 
        # self.mon = {'top': 0, 'left': 0, 'width': 1600, 'height': 900} 
        # self.mon = {'top': 0, 'left': 0, 'width': 960, 'height': 540} 
        self.mon = {'top': 0, 'left': 0, 'width': screen_shot_width, 'height': screen_shot_height} 

        img = self.take_screenshot()

        # Crop variables
        self.w_min_bar = int(img.shape[1]*812/2560)
        self.w_max_bar = int(img.shape[1]*1758/2560)
        self.h_min_bar = int(img.shape[0]*691/1440)
        self.h_max_bar = int(img.shape[0]*915/1440)

        self.w_min_shift = int(img.shape[1]*1759/1920)
        self.w_max_shift = int(img.shape[1]*1854/1920)
        self.h_min_shift = int(img.shape[0]*37/1080)
        self.h_max_shift = int(img.shape[0]*73/1080)

        self.h_min_win = int(img.shape[1]*14/1920)
        self.h_max_win = int(img.shape[1]*220/1920)
        self.w_min_win = int(img.shape[0]*340/1080)
        self.w_max_win = int(img.shape[0]*743/1080)

        self.h_min_hp =  int(img.shape[1]*1008/1920)
        self.h_max_hp = int(img.shape[1]*1041/1920)
        self.w_min_hp = int(img.shape[0]*38/1080)
        self.w_max_hp = int(img.shape[0]*137/1080)

        # Images for correlations
        self.img_shift = cv2.imread('images/shift.png')
        self.img_win = cv2.imread('images/the_from_win_screen.png')
        self.img_1hp_1 = cv2.imread('images/1hp_1.png')
        self.img_1hp_2 = cv2.imread('images/1hp_2.png')
        self.img_2hp = cv2.imread('images/2hp.png')

        # Progression bar threshold
        self.lower_red = np.array([0, 197, 116])
        self.upper_red = np.array([71, 255, 198])

        # Transforms pour les states
        self.resize_w = resize_w
        self.resize_h = resize_h 
        TR_ARRAY2TENSOR = ToTensor()
        TR_COLOR2GRAY = Grayscale()
        TR_RESIZE = Resize((self.resize_h,self.resize_w))
        TR_LIST = [TR_ARRAY2TENSOR,TR_COLOR2GRAY,TR_RESIZE]
        self.transform = Compose(TR_LIST)

        # Variables des timing FPS et skip frame
        self.dim_state = dim_state
        self.episode_time_limite = episode_time_limite

    def take_screenshot(self):
        with mss.mss() as sct:
            img = sct.grab(self.mon)         
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
    
    def reset_episode(self):
        # pg.press('esc')  # vérifier si pas plutôt 'escape'
        # time.sleep(0.5)
        # pg.press('down') # temps du sleep est ok ?
        # time.sleep(0.5)
        # pg.press('enter') # temps du sleep est ok ?
        time.sleep(2)
        # img = self.take_screenshot()                                # A MODIFIER EN FONCTION DE L ETAT A ENVOYNER CHOISI ET DES TRANSFORMES
        # img = self.transform(img).shape
        img_tensor = torch.zeros((self.dim_state,self.resize_h, self.resize_w))
        return img_tensor
    
    def holdKeys(*keys,seconds=0.1):
        for key in keys:
            print(key) 
            pg.keyDown(key)
        time.sleep(seconds)
        for key in keys:
            pg.keyUp(key)

    def is_GameOver(self,img):
        res = cv2.matchTemplate(img[self.h_min_shift:self.h_max_shift,self.w_min_shift:self.w_max_shift] \
                                , self.img_shift, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
        return (res >= 0.8).any()
    
    def compute_progression(self,img):
        img_progress_bar = img[self.h_min_bar:self.h_max_bar,self.w_min_bar:self.w_max_bar]
        hsv_progress_bar = cv2.cvtColor(img_progress_bar, cv2.COLOR_BGR2HSV)
        hsv_progress_bar = cv2.GaussianBlur(src=hsv_progress_bar,ksize=(3,3),sigmaX=1,sigmaY=1) 
        mask = cv2.inRange(hsv_progress_bar, self.lower_red, self.upper_red) # seuillage
        _, mass_x = np.where(mask >= 255)
        try:
            cX = np.average(mass_x)
        except:
            ValueError('CupHead non-detecté dans la barre de progression')
        progression = cX/mask.shape[1]
        return progression

    def is_GameWin(self,img):
        res = cv2.matchTemplate(img[self.h_min_win:self.h_max_win,self.w_min_win:self.w_max_win] \
                                , self.img_win, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
        return (res >= 0.8).any()
    
    def is_health_point_lost(self,current_hp,img):
        if current_hp == 3:
            res = cv2.matchTemplate(img[self.h_min_hp:self.h_max_hp,self.w_min_hp:self.w_max_hp] \
                                , self.img_2hp, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
            return (res >= 0.8).any()
            
        if current_hp == 2:
            res1 = cv2.matchTemplate(img[self.h_min_hp:self.h_max_hp,self.w_min_hp:self.w_max_hp] \
                                , self.img_1hp_1, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
            res2 = cv2.matchTemplate(img[self.h_min_hp:self.h_max_hp,self.w_min_hp:self.w_max_hp] \
                                , self.img_1hp_2, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
            return (res2 >= 0.8).any() or (res1 >= 0.8).any()
        return False
    
    def act_in_environment(self, action_idx, seconds=0.1):
        keys = self.actions_list[action_idx]
        timing = self.hold_timings[action_idx]
        if self.controls_enabled == True:
            for key in keys:
                pg.keyDown(key)
            time.sleep(timing)
            for key in keys:
                pg.keyUp(key)

    def step(self,action_idx, current_hp, temps):    # correspond à un épisode, renvoie : next_state, reward, done, trunc, info
        done = False
        reward = 0

        # Mise a jour de l'image via capture d'écran 

        img = self.take_screenshot()

        # Game Over

        if self.is_GameOver(img) : 
            done = True
            print("You Died !")
            time.sleep(3)
            img = self.take_screenshot()
            progression = self.compute_progression(img)             
            # reward += int(40*progression)                           # reward en fonction dela progression
            reward += -10
            print(f"PROGRESSION : {progression:.4f}")
            pg.press('enter') # recommencer une partie en pressant entrer sur Retry
        
        # Game Win

        if self.is_GameWin(img):
            done = True
            reward += 20
            print("GG ! You won !")
            exit()      # dans un premier temps, cuphead n'arrivera pas jusqu'ici...
        
        # Perte d'HP
        
        if self.is_health_point_lost(current_hp=current_hp, img=img):
            reward += -5
            current_hp += -1

        # Action de l'agent

        self.act_in_environment(action_idx)
        if action_idx in [0,4] :
            reward += 0.4                       # récompense pour avancer
        
        # Génération de l'état suivant

        next_state = torch.zeros(self.dim_state,self.resize_h,self.resize_w)
        for k in range(self.dim_state):
            next_state[k] = self.transform(self.take_screenshot()) 
        

        # Limite de temps pour un épisode atteinte

        if temps > self.episode_time_limite:
            done = True
            reward += -10 
            print("Time limite reached, reseting...")

        return next_state, reward, done, current_hp
    

if __name__ == '__main__':


    # Environnement
    env = CupHeadEnvironment()
    print("Go on the game !...")
    time.sleep(5)

    # # Test des actions
    # for k in range(env.actions_dim):
    #     print(k)
    #     time.sleep(2)
    #     env.act_in_environment(k)

    env.reset_episode()
    exit()
    
    # Agent
    # cuphead = CupHead()
    current_hp = 3

    # Test
    start_time = time.time()
    prev_frame_time = time.time()
    temps = 0
    time_limite = 180
    print("Go on the game !")
    # time.sleep(5)

    state = env.reset_episode()

    while temps<time_limite :  # seuil temps maximal en secondes

        # action_idx = cuphead.act(state)


        next_state, reward, done, current_hp = env.step(action_idx=0, current_hp = current_hp)
        # print(next_state.shape) ; exit()
        


        # Calcul FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        os.system('clear')
        print("FPS : ",fps)
        print("Current HP : ",current_hp)

        temps = time.time()-start_time
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Time limite atteinte")



# TODO : ajouter une option de skip image
# TODO : créer la varible state formée de plusieurs images (concaténation de tenseurs)
# TODO : prendre en compte le temps dans les rewards ?
