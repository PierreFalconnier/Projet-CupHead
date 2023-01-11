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
            print(help(w)) ; exit()
            if w.wm_name == 'Cuphead':
                print(w.x,w.y,w.w,w.h)
                mon = {'top': max(0,w.y), 'left': max(0,w.x), 'width': min(1920-w.x,w.w), 'height': min(1080-w.y,w.h)} 
                bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
                img1 =  bgra_array[:, :, :3]
                cv2.imwrite('test.png',img1)
                if any([w.y<0,w.x<0,w.x+w.w>1920, w.y+w.h>1080]):
                    print("Cuphead window outside of the sreen !")
                w_active = Window.get_active()
                if w_active.wm_name != 'Cuphead':
                    print('Cuphead window not on screen !')
            # print(w.wm_name)
        time.sleep(1)