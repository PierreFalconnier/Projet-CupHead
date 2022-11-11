# Imports et fonctions

import pyautogui as pg
import matplotlib.pyplot as plt
import time 


# Maintenir une touche un certain temps
def holdKey(key,seconds=3):
    pg.keyDown(key)
    time.sleep(seconds)
    pg.keyUp(key)



# Actions
actions_binds_list = ["x"    ,"z"   ,"v"      ,"shiftleft","c"   ,"right","left","up","down","tab"          ]
actions_names_list = ["shoot","jump","exshoot","dash"     ,"lock","right","left","up","down","switch_weapon"]

flying_actions_list =       ["x"    ,"v"      ,"z"    ,"shift"]
flying_actions_names_list = ["shoot","special","parry","shrink"]

action_dim = len(actions_binds_list)

# Environment class

class CupHeadEnvironment(object):

    def __init__(self) -> None:
        self.actions_binds_list = actions_binds_list
    


    def reset(self):
        pass

    def step(self):
        pass




if __name__ == '__main__':
    # Tests sur l'environnement
    a = CupHeadEnvironment()
    print(a.actions_binds_list)