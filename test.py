### Imports

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
from subprocess import check_output


img = cv2.imread('full_1hp_1.jpg')

h_min_hp = int(img.shape[1]*1008/1920)
h_max_hp = int(img.shape[1]*1041/1920)
w_min_hp = int(img.shape[0]*38/1080)
w_max_hp = int(img.shape[0]*137/1080)
img = img[h_min_hp:h_max_hp,w_min_hp:w_max_hp] 

cv2.imwrite('1hp_2.png',img)

plt.figure()
plt.imshow(img)
plt.show()
exit()






def holdKey(key,seconds=3):
    pg.keyDown(key)
    time.sleep(seconds)
    pg.keyUp(key)

def hold2Keys(key1,key2,seconds=3):
    pg.keyDown(key1)
    pg.keyDown(key2)
    time.sleep(seconds)
    pg.keyUp(key1)
    pg.keyUp(key2)


time.sleep(4)
# keyboard.send('z', do_release=False)

pg.PAUSE = 0 #valeur par défaut = 0.1, temps entre les actions


# for k in range(4):
#     time.sleep(1)
#     pg.keyDown('right')
#     pg.keyDown('z')
#     time.sleep(0.5)
#     pg.keyUp('right')
#     pg.keyUp('z')



# exit()

### Variables et fonctions

actions_binds_list = ["x"    ,"z"   ,"v"      ,"shiftleft","c"   ,"right","left","up","down","tab"          ]
actions_names_list = ["shoot","jump","exshoot","dash"     ,"lock","right","left","up","down","switch_weapon"]
flying_actions_list =       ["x"    ,"v"      ,"z"    ,"shift"]
flying_actions_names_list = ["shoot","special","parry","shrink"]
action_dim = len(actions_binds_list)

# Action réellement utilisées :
actions_list = ["right","left","up","down","still","z","shiftleft",("z","right"),("z","left")]
# si still : ne rien faire
# ajouter le shoot et exshoot dans un second temps, dans un premier temps, shoot en continu. Avec les différentes directions pk pas

mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080} 
# mon = {'top': 0, 'left': 0, 'width': 800, 'height': 400} 

done = False
step_num = 0

def take_screenshot():
    with mss.mss() as sct:
        img = sct.grab(mon)         
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

img = take_screenshot()

w_min_bar = int(img.shape[1]*812/2560)
w_max_bar = int(img.shape[1]*1758/2560)
h_min_bar = int(img.shape[0]*691/1440)
h_max_bar = int(img.shape[0]*915/1440)

w_min_shift = int(img.shape[1]*1759/1920)
w_max_shift = int(img.shape[1]*1854/1920)
h_min_shift = int(img.shape[0]*37/1080)
h_max_shift = int(img.shape[0]*73/1080)

h_min_win = int(img.shape[1]*14/1920)
h_max_win = int(img.shape[1]*220/1920)
w_min_win = int(img.shape[0]*340/1080)
w_max_win = int(img.shape[0]*743/1080)

img_shift = cv2.imread('shift.png')
img_win = cv2.imread('the_from_win_screen.png')

lower_red = np.array([0, 197, 116])
upper_red = np.array([71, 255, 198])

def nothing(x):
    pass

def is_GameOver(img):
    res = cv2.matchTemplate(img[h_min_shift:h_max_shift,w_min_shift:w_max_shift] \
                            , img_shift, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
    return (res >= 0.8).any()

def is_GameWin(img):
    res = cv2.matchTemplate(img[h_min_win:h_max_win,w_min_win:w_max_win] \
                            , img_win, eval('cv2.TM_CCOEFF_NORMED')) # par correlation
    return (res >= 0.8).any()

def compute_progression(img):
    img_progress_bar = img[h_min_bar:h_max_bar,w_min_bar:w_max_bar]
    hsv_progress_bar = cv2.cvtColor(img_progress_bar, cv2.COLOR_BGR2HSV)
    hsv_progress_bar = cv2.GaussianBlur(src=hsv_progress_bar,ksize=(3,3),sigmaX=1,sigmaY=1) 
    mask = cv2.inRange(hsv_progress_bar, lower_red, upper_red) # seuillage
    _, mass_x = np.where(mask >= 255)
    try:
        cX = np.average(mass_x)
    except:
        ValueError('CupHead non-detecté dans la barre de progression')
    progression = cX/mask.shape[1]
    return progression

def reset_episode():
    pg.press('esc')  # vérifier si pas plutôt 'escape'
    pg.press('enter') # besoin d'un sleep entre les deux ?

def step(time_limite=180):                # correspond à un épisode, renvoie : next_state, reward, done, trunc, info

    start_time = time.time()
    prev_frame_time = time.time()
    temps=0
    loop_counter = 0

    while temps<time_limite :  # seuil temps maximal en secondes
        loop_counter += 1


        # Mise a jour de l'image via capture d'écran 

        img = take_screenshot()
        
        # action de l'agent

        # pg.press('esc')
        # pg.press(action_bind)

        # Vérifier si Game Over ou Game Win

        if is_GameOver(img) : 
            done = True
            cv2.imwrite('test.png',img)
            print("You Died !")
            time.sleep(2)
            img = take_screenshot()
            progression = compute_progression(img)
            print("Progression : ") 
            reward = int(100*progression)
            pg.press('enter') # recommencer une partie en pressant entrer sur Retry
            return reward

        if is_GameWin(img):
            done = True
            reward = 200
            print("GG ! You won !")
            return reward
        

        # Calcul FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        os.system('clear')
        print("FPS : ",fps)

        temps = time.time()-start_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Time milite atteinte")
    return 0   # reward en cas d'echec
    
    


if __name__=='__main__':

    print("Go on the game !")
    time.sleep(5)

    step()




# # # Commentaires et notes


# w_min_dead = int(img.shape[1]*35/1920)
# w_max_dead = int(img.shape[1]*144/1920)
# h_min_dead = int(img.shape[0]*1009/1080)
# h_max_dead = int(img.shape[0]*1040/1080)
# img_dead = img[h_min_dead:h_max_dead,w_min_dead:w_max_dead]

# w_min = int(img.shape[1]*1190/2560)
# w_max = int(img.shape[1]*1409/2560)
# h_min = int(img.shape[0]*945/1440)
# h_max = int(img.shape[0]*1014/1440)
# img_retry = img[h_min:h_max,w_min:w_max]

# # Fenêtres, avant la boucle while
# cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Mask', 1024, 1024)
# cv2.namedWindow('Hsv', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hsv', 1024, 1024)

# # Affichage, dans la boucle
# cv2.imshow("Hsv", img)
# cv2.imshow("Hsv", hsv_progress_bar)
# cv2.imshow("Mask", mask)