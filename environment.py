# Imports et fonctions
import pyautogui as pg
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from torchvision.transforms import ToTensor, Resize, Grayscale, Compose
import torch

if os.name == 'nt':
    from mss.windows import MSS as mss
else:
    from wmctrl import Window
    from mss.linux import MSS as mss


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

def iprint(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

# Classe pour utiliser le clavier sur un nouveau thread

import threading

class InteractWithKeyboard(threading.Thread):

    def __init__(self,actions_list = [], 
    hold_timings = [],
    controls_enabled=True,
    mon_for_correlation=None,):
        super().__init__()
        self.actions_list = actions_list  
        self.hold_timings = hold_timings
        self.controls_enabled = controls_enabled
        self.mon_for_correlation = mon_for_correlation

    def act_in_environment(self, action_idx):
        keys = self.actions_list[action_idx]
        timing = self.hold_timings[action_idx]
        if self.controls_enabled == True:
            for key in keys:
                pg.keyDown(key)
            time.sleep(timing)
            for key in keys:
                pg.keyUp(key)

# Actions
# actions_binds_list = ["x"    ,"z"   ,"v"      ,"shiftleft","c"   ,"right","left","up","down","tab"          ]
# actions_names_list = ["shoot","jump","exshoot","dash"     ,"lock","right","left","up","down","switch_weapon"]
# flying_actions_list =       ["x"    ,"v"      ,"z"    ,"shift"]
# flying_actions_names_list = ["shoot","special","parry","shrink"]
# action_dim = len(actions_binds_list)

# Environment class, ATTENTION : on manipule des images BGR, pas RGB, c'est important pour les corrélations

class CupHeadEnvironment(object):

    def __init__(
        self,
        screen_width = 1920,
        screen_height = 1080,
        resize_w=128,
        resize_h=128,
        dim_state=2,
        controls_enabled = True,
        episode_time_limite = 180,
        reward_dict={},
        actions_list = [],
        hold_timings = [],
        forward_action_index_list = [],
        ) -> None:


        # Check if game running
    
        is_cuphead_launched = False
        for w in Window.list():
            if w.wm_name == 'Cuphead':
                is_cuphead_launched = True
                w.activate()
                break
        if is_cuphead_launched == False : 
            print('Game not running, exiting.') ; exit()
        w_active = Window.get_active()
        if w_active.wm_name != 'Cuphead':
            print('Cuphead window not on screen ! Activating cuphead window.')
            w.activate()
        
        x,y,w,h = w.x-1,w.y-38,w.w,w.h   # prise en compte de l'offset de la bar de titre, jai utilsé xwininfo dans un terminal
        # pg.moveTo(x,y)
        # pg.moveTo(x+w,y+h)

        # Get window position and size needed for screenshots
        print('Cuphead window box(x,y,w,h) : ',x,y,w,h)
        self.mon = {'top': max(0,y), 'left': max(0,x), 'width': min(screen_width-x,w), 'height': min(screen_height-y,h)} 
        self.mon_for_correlation = {'top': max(0,y)+ 7*h//8, 'left': max(0,x)+ 3*w//8, 'width': 2*w//8, 'height': h//8}   # /!\ A MODIFIER AVEC LA POSITION DE LA FENETRE à mettre comme arg du constructeur
        
        if any([y<0,x<0,x+w>screen_width, y+h>screen_height]):
            print("Cuphead window outside of the sreen !")

        # Actions
        # self.actions_list = [["right"],["left"],["up"],["down"],["s"],["z"],["shiftleft"],["z","right"],["z","left"]]   # s correspond à 'still', cuphead ne fait rien
        # self.hold_timings = [0.75,   0.75,  0.3, 0.3,   0.75,  0.75,   0.1,      0.65,          0.65]
        # self.actions_list = [["right"],["left"],["left"],["z"],["z","right"],["shiftleft"]]   # s correspond à 'still', cuphead ne fait rien
        # self.hold_timings = [0.75,   0.75,        0.1,    0.7
        # 5,      0.65,        0.1]
       
        self.actions_list = actions_list  
        self.hold_timings = hold_timings
        self.forward_action_index_list = forward_action_index_list
       
        self.actions_dim = len(self.actions_list)
        self.controls_enabled = controls_enabled   # si True, le programme utilise PyAutoGUI et controle le clavier

        self.current_hp = 3 
        self.last_progress = 0
        self.done = False
        self.reward = 0

        # self.interact = InteractWithKeyboard(actions_list=actions_list, 
        #                                     hold_timings=hold_timings,
        #                                     controls_enabled=controls_enabled,
        #                                     )
        

        # Set correlation_threshold
        self.correlation_threshold = 0.8
        
        # Crop variables
   
        with mss() as sct:
            bgra_array = np.array(sct.grab(self.mon)  , dtype=np.uint8)
            img =  bgra_array[:, :, :3]
  

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

        img_hp = img[self.h_min_hp:self.h_max_hp,self.w_min_hp:self.w_max_hp]
        gray = cv2.cvtColor(img_hp, cv2.COLOR_BGR2GRAY)
        _,self.new_hp = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        _,self.prev_hp = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Images for correlations, resize (originale saved images on 1920x1080 tests)

        def resize_saved_image(img,h,w):
            return cv2.resize(img,(int(img.shape[1]* w/1920),int(img.shape[0]*h/1080)))
        
        self.img_shift = resize_saved_image(cv2.imread('images/shift.png'),h,w)
        self.img_win = resize_saved_image(cv2.imread('images/the_from_win_screen.png'),h,w)
        self.img_1hp_1 = resize_saved_image(cv2.imread('images/1hp_1.png'),h,w)
        self.img_1hp_2 = resize_saved_image(cv2.imread('images/1hp_2.png'),h,w)
        self.img_2hp = resize_saved_image(cv2.imread('images/2hp.png'),h,w)

        # Progression bar threshold
        self.lower_red = np.array([0, 197, 116])
        self.upper_red = np.array([71, 255, 198])

        # Transforms pour les states
        self.resize_w = resize_w
        self.resize_h = resize_h 
        TR_ARRAY2TENSOR = ToTensor()
        TR_COLOR2GRAY = Grayscale()                         # image en float entre 0 et 1, shape B H W
        TR_RESIZE = Resize((self.resize_h,self.resize_w))
        TR_LIST = [TR_ARRAY2TENSOR,TR_COLOR2GRAY,TR_RESIZE]
        self.transform = Compose(TR_LIST)

        # Variables des timing FPS et skip frame
        self.dim_state = dim_state
        self.episode_time_limite = episode_time_limite

        self.sleep_between_state_frames = 1/24  # le jeu est à 24 fps

        # Reward dictionary
        self.reward_dict = reward_dict

        ## Exemple :
        #REWARD_DICT = {
        # 'Health_point_lost':-10,
        # 'GameWin' : 100,
        # 'GameOver' : -20,
        # 'Forward': 1
        #}

    def take_screenshot(self):
        with mss() as sct:
            frame = np.array(sct.grab(self.mon)  , dtype=np.uint8)
            return frame[:, :, :3]  
    
    def reset_episode(self):
        time.sleep(2)
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
                                , self.img_shift, eval('cv2.TM_CCOEFF_NORMED'))
        return (res >= self.correlation_threshold).any()       
    
    def compute_progress(self,img):
        img_progress_bar = img[self.h_min_bar:self.h_max_bar,self.w_min_bar:self.w_max_bar]
        hsv_progress_bar = cv2.cvtColor(img_progress_bar, cv2.COLOR_BGR2HSV)
        hsv_progress_bar = cv2.GaussianBlur(src=hsv_progress_bar,ksize=(3,3),sigmaX=1,sigmaY=1) 
        mask = cv2.inRange(hsv_progress_bar, self.lower_red, self.upper_red) # seuillage
        _, mass_x = np.where(mask >= 255)
        try:
            cX = np.average(mass_x)
        except:
            ValueError('CupHead non-detecté dans la barre de progression')
        progress = cX/mask.shape[1]
        return progress

    def is_GameWin(self,img):
        res = cv2.matchTemplate(img[self.h_min_win:self.h_max_win,self.w_min_win:self.w_max_win] \
                                , self.img_win, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
        return (res >= self.correlation_threshold).any()
    
    # def is_health_point_lost(self,current_hp,img):
    def is_health_point_lost(self,img):
        
        # # Maj new 
        # img_hp = img[self.h_min_hp:self.h_max_hp,self.w_min_hp:self.w_max_hp]
        # gray = cv2.cvtColor(img_hp, cv2.COLOR_BGR2GRAY)
        # _,self.new_hp = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # # Correlation
        # res = cv2.matchTemplate(self.prev_hp, self.new_hp, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
        # # Maj prev
        # self.prev_hp = self.new_hp.copy()

        # return (res < self.correlation_threshold).any()

        img_hp = img[self.h_min_hp:self.h_max_hp,self.w_min_hp:self.w_max_hp]
        if self.current_hp == 3:
            res = cv2.matchTemplate(img_hp \
                                , self.img_2hp, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
            
            return (res >= 0.95).any()   # commenter si version  multithreading 
            
        if self.current_hp == 2:
            res1 = cv2.matchTemplate(img_hp \
                                , self.img_1hp_1, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
            res2 = cv2.matchTemplate(img_hp \
                                , self.img_1hp_2, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
            return (res2 >= self.correlation_threshold).any() or (res1 >=self.correlation_threshold).any()    # commenter si version  multithreading 
        
        return False

    def act_in_environment(self, action_idx):
        keys = self.actions_list[action_idx]
        timing = self.hold_timings[action_idx]
        if self.controls_enabled == True:
            for key in keys:
                pg.keyDown(key)
            time.sleep(timing)
            for key in keys:
                pg.keyUp(key)

    def step(self,action_idx, temps):    # correspond à un épisode, renvoie : next_state, reward, done, trunc, info
        self.done = False
        self.reward = 0
        # t = time.time()

        # Action de l'agent
        act_thread = threading.Thread(target=self.act_in_environment, args=[action_idx])    # parallélisation de l'action
        act_thread.start()

        with mss() as sct:

            # Mise a jour de l'image via capture d'écran 

            bgra_array = np.array(sct.grab(self.mon_for_correlation)  , dtype=np.uint8)
            prev =cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2GRAY)
            bgra_array = np.array(sct.grab(self.mon)  , dtype=np.uint8)
            img =  bgra_array[:, :, :3]     

            # Game Over

            if self.is_GameOver(img) : 
                self.done = True
                print("You Died !")
                time.sleep(3)
                bgra_array = np.array(sct.grab(self.mon)  , dtype=np.uint8)
                img =  bgra_array[:, :, :3]
                progress = self.compute_progress(img)           
                print(f"Progress : {progress:.4f}")
                self.last_progress = progress  
                self.reward +=  self.reward_dict['GameOver']
                pg.press('enter')           # recommencer une partie 
            
            # Game Win

            if self.is_GameWin(img):
                self.done = True
                self.reward += self.reward_dict['GameWin']
                print("GG ! You won !")
                exit()      # dans un 1er temps
            
            # Perte d'HP

            if self.current_hp>1 and self.is_health_point_lost(img=img):
                self.current_hp += -1
                # print(f'HP lost ! HP left : {self.current_hp}')
                self.reward += self.reward_dict['Health_point_lost']

            # Génération de l'état suivant

            next_state = torch.zeros(self.dim_state,self.resize_h,self.resize_w)
            for k in range(self.dim_state):
                bgra_array = np.array(sct.grab(self.mon)  , dtype=np.uint8)
                img_state =  bgra_array[:, :, :3]  # copy pour régler le pb des srides négatifs par géré par torch
                next_state[k] = self.transform(img_state.copy()) 
                time.sleep(self.sleep_between_state_frames)  # 1/24 sec

            # Correlation

            bgra_array = np.array(sct.grab(self.mon_for_correlation)  , dtype=np.uint8)
            next =cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2GRAY)
            res = cv2.matchTemplate(prev,next, eval('cv2.TM_CCOEFF_NORMED')) 
            if (res < 0.9).any() and action_idx in self.forward_action_index_list:
                # print("Moving Forward !")
                self.reward += self.reward_dict['Forward']                      # récompense pour avancer
   
            # Limite de temps pour un épisode atteinte

            if temps > self.episode_time_limite:
                self.done = True
                self.reward += -10 
                print("Time limite reached, reseting...")

            act_thread.join()    # pour synchro / attendre le thread avant de terminer 

            # print(t - time.time())
            return next_state, self.reward, self.done
    

if __name__ == '__main__':

    ACTION_LIST  = [["right"],["left"],["left"],["z"],["z","right"]]   # s correspond à 'still', cuphead ne fait rien
    HOLD_TIMINGS = [0.75,   0.75,        0.1,    0.75,      0.65   ]
    SCREEN_WIDTH, SCREEN_HEIGHT = pg.size() 
    env = CupHeadEnvironment(actions_list=ACTION_LIST,hold_timings=HOLD_TIMINGS, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH)