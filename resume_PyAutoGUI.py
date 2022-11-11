import pyautogui as pg
import time
import keyboard  # utile pour taper certain caractères comme le "#" qui ne sont pas supportés par pyautogui
import matplotlib.pyplot as plt


# im1 = pg.screenshot()       # capture d'écran, donner un path pour la sauvgarder
# plt.figure()
# plt.imshow(im1)
# plt.show() 
# time.sleep(3)

# print(pg.size())        # afficher résolution écran
# print(pg.position())   # position de la souris

# pg.moveTo(41,606)        # bouger le curseur position et temps de parcour 0 par défaut
# pg.moveRel(10,10)        # bouger le curseur relativement à la poition actuelle

# pg.leftClick()           # clicker
# pg.rightClick()
# pg.doubleClick()
# pg.tripleClick()
# pg.click(500, 500, 3,2,button="left")  # trois clicks, 2s pour aller à la position
# pg.scroll(-500)

# pg.mouseDown(300,400,button="left")     # laisser enfoncé le button
# pg.moveTo(800,400)
# pg.mouseUp()

# # Exemple dessiner une spirale
# time.sleep(1)
# distance = 300
# while distance>0:
#     pg.dragRel(distance,0,1,button="left")   # ecore une fois, position, temps de parcour, bouton
#     distance = distance -20
#     pg.dragRel(0,distance,1,button="left")
#     pg.dragRel(-distance,0,1,button="left")
#     distance = distance -20
#     pg.dragRel(0,-distance,1,button="left")


# # Exemple recerche sur firefox
# pg.leftClick(41,606, 1)           # clicker une fois à la poisiotn donnée, ici celle de l'icone du browser
# time.sleep(0.5)
# with pg.hold("ctrl"):           # on peut aussi utiliser keyUp() puis keyDown()
#     pg.press('k')
# pg.typewrite("https://www.youtube.com/results?search_query=pyautogui+python+tutorial")                      # pareil que pg.write()
# time.sleep(0.5)
# pg.press("enter")                      # presser la touche entrer


# # Localiser sur l'écran
# edit_button = pg.locateOnScreen("edit.png") # renvoie un rectange qui correspond à la position 
# print(edit_button)
# edit_button = pg.locateCenterOnScreen("edit.png") # renvoie la position du centre du rectangle
# print(edit_button)
# pg.moveTo(edit_button)

# Fonction pour maintenir une touche un certain temps
def holdKey(key,seconds=1):
    pg.keyDown(key)
    time.sleep(seconds)
    pg.keyUp(key)

# # Capture d'écran
# im1 = pg.screenshot()       # capture d'écran, donner un path pour la sauvgarderzzzzz
# plt.figure()
# plt.imshow(im1)
# plt.show() 



""" KeyBoard Keys PyAutoGUI
['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
'8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
'a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
'browserback', 'browserfavorites', 'browserforward', 'browserhome',
'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
'command', 'option', 'optionleft', 'optionright']
"""