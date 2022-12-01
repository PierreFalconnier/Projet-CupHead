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

import re
import math
import os
import pygetwindow as gw
import numpy
from PIL import Image



def mss_rgb(im):
    """ Better than Numpy versions, but slower than Pillow. """
    return im.rgb


def numpy_flip(im):
    """ Most efficient Numpy version as of now. """
    frame = numpy.array(im, dtype=numpy.uint8)
    return numpy.flip(frame[:, :, :3], 2)


def numpy_slice(im):
    """ Slow Numpy version. """
    return numpy.array(im, dtype=numpy.uint8)[..., [2, 1, 0]].tobytes()


def pil_frombytes(im):
    """ Efficient Pillow version. """
    return Image.frombytes('RGB', im.size, im.bgra, 'raw', 'BGRX')

def screenshot_process(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img




if __name__=='__main__':

    if os.name == 'nt':
        from mss.windows import MSS as mss
        # import win32gui
        # print(gw.getAllTitles()) # nom de la fenêtre = 'Cuphead'
        # print(gw.getWindowsWithTitle('Cuphead'))
        # window = gw.getWindowsWithTitle('Cuphead')[-1]
        # window.restore()
    else:
        from mss.linux import MSS as mss

    step = 20000
    proba = 0.5
    rate = 0.999965
    print("STEPS needed to reach proba : ",math.log(proba)/math.log(rate))
    print("Rate to reach proba after given step : ", proba**(1/step))

    mon = {'top': 0, 'left': 0, 'width': 1920//2, 'height': 1080//2} 

    with mss() as sct:
        bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
        img1 =  np.flip(bgra_array[:, :, :3], 2)
        img1=cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)

        t1 = time.time()
        cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
        while True :
            bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
            img2 =  np.flip(bgra_array[:, :, :3], 2)
            img2=cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)

            res = cv2.matchTemplate(img2,img1, eval('cv2.TM_CCOEFF_NORMED')) 
            # (res >= 0.8).any()
   
            # cv2.imshow("Image", img2)
            # print(res)

            bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
            img1 =  np.flip(bgra_array[:, :, :3], 2)
            img1=cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)


# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# https://youtu.be/hfXMw2dQO4E

            # cv2.calcOpticalFlowPyrLK(img2, img1, )


            # print("FPS : ", int(1/(time.time()-t1)))
            t1 = time.time()
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
cv2.destroyAllWindows()
        


    


        

        

        





