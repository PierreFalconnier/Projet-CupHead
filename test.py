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

import win32gui
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
   
    # print(gw.getAllTitles()) # nom de la fenêtre = 'Cuphead'
    # print(gw.getWindowsWithTitle('Cuphead'))
    # window = gw.getWindowsWithTitle('Cuphead')[-1]
    # window.activate()

    if os.name == 'nt':
        print("OS = WINDOWS")
    else:
        print("OS = Linux")

    step = 20000
    proba = 0.5
    rate = 0.999965

    print("STEPS needed to reach proba : ",math.log(proba)/math.log(rate))
    print("Rate to reach proba after given step : ", proba**(1/step))

    if os.name == 'nt':
        from mss.windows import MSS as mss
    else:
        from mss.linux import MSS as mss

    with mss() as sct:
        im = sct.grab(sct.monitors[1])
        # rgb = numpy_flip(im)
        rgb = screenshot_process(im)
        from torchvision.transforms import ToTensor, Resize, Grayscale, Compose
        TR_ARRAY2TENSOR = ToTensor()
        TR_COLOR2GRAY = Grayscale()
        TR_RESIZE = Resize((128,128))
        TR_LIST = [TR_ARRAY2TENSOR,TR_COLOR2GRAY]
        transform = Compose(TR_LIST)

        rgb =  transform(rgb)
        # print(rgb)
        print(type(rgb))
        print(rgb.shape)
        print(rgb)

        

        





