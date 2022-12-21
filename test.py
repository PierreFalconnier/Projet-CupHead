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
import numpy
from PIL import Image
import pprint


# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0)) ; exit()


if __name__=='__main__':
    time.sleep(2)
    keys=["z","right"]
    timings = [0.5,1]
    
    # Press differents keys with differents timings
    # s = time.time()

    # max = max(timings)
    # n = len(keys)
    # start = time.time()
    
    # for key in keys:
    #     pg.keyDown(key)
    # while time.time() - start < max:
    #     for i,key in enumerate(keys):
    #         if time.time() - start > timings[i]: pg.keyUp(key)
    # for key in keys:
    #     pg.keyUp(key)

    # print(abs(s-time.time()))
    
    timing = 0.1
    # s = time.time()
    
    for key in keys:
        pg.keyDown(key)
    time.sleep(timing)
    for key in keys:
        pg.keyUp(key)

    keys=["shiftleft"]
    
    for key in keys:
        pg.keyDown(key)
    time.sleep(timing)
    for key in keys:
        pg.keyUp(key)
    
    # print(abs(s-time.time()))


    # for key in keys:
    #         print(key) 
    #         pg.keyDown(key)
    # time.sleep(0.4)
    # for key in keys:
    #     pg.keyUp(key)

    if os.name == 'nt':
        import pygetwindow as gw
        from mss.windows import MSS as mss
        # import win32gui
        # print(gw.getAllTitles()) # nom de la fenêtre = 'Cuphead'
        # print(gw.getWindowsWithTitle('Cuphead'))
        # window = gw.getWindowsWithTitle('Cuphead')[-1]
        # window.restore()
    else:
        from mss.linux import MSS as mss

    

    # from win32gui import FindWindow, GetWindowRect

    # # FindWindow takes the Window Class name (can be None if unknown), and the window's display text. 
    # window_handle = FindWindow(None, "Diablo II")
    # window_rect   = GetWindowRect(window_handle)

    # print(window_rect)


  

    step = 50000
    proba = 0.5
    rate = 0.999965
    print("STEPS needed to reach proba : ",math.log(proba)/math.log(rate))
    print("Rate to reach proba after given step : ", proba**(1/step))
    exit()
    # mon = {'top': 0, 'left': 0, 'width': 1920//2, 'height': 1080//2} 
    mon = {'top': 1080//4, 'left': 0, 'width': 1920//2, 'height': 1080//2} 


    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    with mss() as sct:
        bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
        img1 =  bgra_array[:, :, :3]
        # img1=cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)

        plt.figure()
        plt.imshow(img1)
        plt.show()
        print(type(img1),img1.shape)
        exit()

        t1 = time.time()
        cv2.namedWindow("Image",cv2.WINDOW_NORMAL)





        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        old_frame = img1
        old_gray = old_frame
        p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while True :
            bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
            img2 = bgra_array[:, :, :3]
            img2=cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
            frame = img2


            #  # calculate optical flow
            # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame, p0, None, **lk_params)
            # # Select good points
            # if p1 is not None:
            #     good_new = p1[st==1]
            #     good_old = p0[st==1]
            # # draw the tracks
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #     frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            # img = cv2.add(frame, mask)
            # cv2.imshow('frame', img)


            # p0 = good_new.reshape(-1, 1, 2)



            # calculate correlation
            res = cv2.matchTemplate(frame,old_gray, eval('cv2.TM_CCOEFF_NORMED')) 
            # (res <0.8).any()
   
            cv2.imshow("Image", img2)
            print(res)

            # # https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
            # # https://youtu.be/hfXMw2dQO4E


            old_gray = frame.copy()


            # print("FPS : ", int(1/(time.time()-t1)))
            t1 = time.time()
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
cv2.destroyAllWindows()
        


    


        
# class InteractWithKeyboard(threading.Thread):

#     def __init__(self,actions_list = [], 
#     hold_timings = [],
#     controls_enabled=True,
#     mon_for_correlation=None,):
#         super().__init__()
#         self.actions_list = actions_list  
#         self.hold_timings = hold_timings
#         self.controls_enabled = controls_enabled
#         self.mon_for_correlation = mon_for_correlation

#     def act_in_environment(self, action_idx):
#         keys = self.actions_list[action_idx]
#         timing = self.hold_timings[action_idx]
#         if self.controls_enabled == True:
#             for key in keys:
#                 pg.keyDown(key)
#             time.sleep(timing)
#             for key in keys:
#                 pg.keyUp(key)
        

        





