#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import time


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def main(use_cam = True):
    import os
    if os.name == 'nt':
        import pygetwindow as gw
        from mss.windows import MSS as mss
    else:
        from mss.linux import MSS as mss
    mon = {'top': 100, 'left': 0, 'width': 3*1920//4, 'height': 3*1080//4} 
    # mon = {'top': 7*1080//8, 'left': 1920//4, 'width': 2*1920//4, 'height': 1080//8} 

    import sys
    import cv2
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    with mss() as sct:
        if use_cam:
            cam = cv2.VideoCapture(0)
            _ret, prev = cam.read()
            prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        else :
            bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
            prev =  np.flip(bgra_array[:, :, :3], 2)
            prevgray=cv2.cvtColor(prev, cv2.COLOR_BGRA2GRAY)

        show_hsv = False
        show_glitch = False
        cur_glitch = prev.copy()

        t1 = time.time()
        while True:
            if use_cam:
                _ret, img = cam.read()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                bgra_array = np.array(sct.grab(mon)  , dtype=np.uint8)
                img =  np.flip(bgra_array[:, :, :3], 2)
                gray=cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)


            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray

            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # print(np.pi-np.pi/6, ang.mean(), np.pi+np.pi/6)
            # print(np.pi-np.pi/6 < float(ang.mean()) and (float(ang.mean()) < np.pi+np.pi/6))
            print(mag.mean())
           
            if (np.pi-np.pi/6 < ang.mean() < np.pi+np.pi/6) and mag.mean() > 5 :
                pass
                print("Cuphead Avance !")


          
            cv.imshow('flow', ResizeWithAspectRatio(draw_flow(gray, flow), width=800))
            if show_hsv:
                cv.imshow('flow HSV', draw_hsv(flow))
            if show_glitch:
                cur_glitch = warp_flow(cur_glitch, flow)
                cv.imshow('glitch', cur_glitch)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                print('glitch is', ['off', 'on'][show_glitch])
            # print("FPS : ", int(1/(time.time()-t1)))
            t1 = time.time()

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    main(use_cam=False)
    cv.destroyAllWindows()