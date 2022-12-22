# import pygetwindow as gw
# print( gw.getWindowsWithTitle('Cuphead'))
# window = gw.getWindowsWithTitle('Cuphead')[-1]
# window.restore()

import win32gui

# def callback(hwnd, extra):
#     rect = win32gui.GetWindowRect(hwnd)
#     x = rect[0]
#     y = rect[1]
#     w = rect[2] - x
#     h = rect[3] - y
#     print("Window %s:" % win32gui.GetWindowText(hwnd))
#     print("\tLocation: (%d, %d)" % (x, y))
#     print("\t    Size: (%d, %d)" % (w, h))

# def main():
#     win32gui.EnumWindows(callback, None)

# if __name__ == '__main__':
#     main()


import pygetwindow as gw
from mss.windows import MSS as mss
import win32gui
import time
print(gw.getAllTitles()) # nom de la fenêtre = 'Cuphead'
print(gw.getWindowsWithTitle('Cuphead'))


if len(gw.getWindowsWithTitle('Cuphead')) != 2 :
    print('Game not running, exiting.') ; exit()
window = gw.getWindowsWithTitle('Cuphead')[-1]
window.activate()
print(window.size)
print(window.topleft)

p = window.topleft
x = p.x+2
y = p.y+25
h= window.height-29
w = window.width-5
print(x,y)

import pyautogui as pg

# pg.displayMousePosition() 

pg.moveTo((x+w,y+h))