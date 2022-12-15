import os
from wmctrl import Window
import pprint
import time
import numpy as np
import cv2
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


        

# w= Window.get_active()
# print(w.wm_name)

# print(help(Window))
# pprint.pprint(Window.list())

with mss() as sct:
    while True:
        for w in Window.list():
            if w.wm_name == 'Cuphead':
                print(w.x,w.y,w.w,w.h)
                mon = {'top': w.x, 'left': w.y, 'width': w.w, 'height': w.h} 
                bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
                img1 =  bgra_array[:, :, :3]
                cv2.imwrite('test.png',img1)
            # print(w.wm_name)
        time.sleep(1)