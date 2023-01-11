# Print the name and bounding box (x1, y1, x2, y2) for the active window in
# a loop.



import time
from collections import namedtuple

import Xlib
import Xlib.display


disp = Xlib.display.Display()
root = disp.screen().root

NET_ACTIVE_WINDOW = disp.intern_atom('_NET_ACTIVE_WINDOW')
MyGeom = namedtuple('MyGeom', 'x y height width')


def get_active_window():
    win_id = root.get_full_property(NET_ACTIVE_WINDOW,
                                    Xlib.X.AnyPropertyType).value[0]
    try:
        return disp.create_resource_object('window', win_id)
    except Xlib.error.XError:
        pass


def get_absolute_geometry(win):
    """
    Returns the (x, y, height, width) of a window relative to the top-left
    of the screen.
    """
    geom = win.get_geometry()
    (x, y) = (geom.x, geom.y)
    while True:
        parent = win.query_tree().parent
        pgeom = parent.get_geometry()
        x += pgeom.x
        y += pgeom.y
        if parent.id == root.id:
            break
        win = parent
    return MyGeom(x, y, geom.height, geom.width)


def get_window_bbox(win):
    """
    Returns (x1, y1, x2, y2) relative to the top-left of the screen.
    """
    geom = get_absolute_geometry(win)
    x1 = geom.x
    y1 = geom.y
    x2 = x1 + geom.width
    y2 = y1 + geom.height
    return (x1, y1, x2, y2)


def main():
    while True:
        # Protect against races when the window gets destroyed before we
        # have a chance to use it.  Guessing there's a way to lock access
        # to the resource, but for this demo we're just punting.  Good
        # enough for who it's for.
        try:
            win = get_active_window()
            print(win.get_wm_name(), get_window_bbox(win))
        except Xlib.error.BadWindow:
            print("Window vanished")
        time.sleep(1)


if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    import time
    if os.name == 'nt':
        from mss.windows import MSS as mss
    else:
        from mss.linux import MSS as mss

    while True:
        # Protect against races when the window gets destroyed before we
        # have a chance to use it.  Guessing there's a way to lock access
        # to the resource, but for this demo we're just punting.  Good
        # enough for who it's for.
        try:
            time.sleep(3)
            win = get_active_window()
            print(win.get_wm_name(), get_window_bbox(win))
            box = get_window_bbox(win)
            mon = {'top': box[0], 'left': box[1], 'width': box[2], 'height': box[3]} 

            # with mss() as sct:
            #     bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
            #     img1 =  bgra_array[:, :, :3]
            #     cv2.imwrite('test.png', img1)

        except Xlib.error.BadWindow:
            print("Window vanished")
        time.sleep(1)
    

    
        