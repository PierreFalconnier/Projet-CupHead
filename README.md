esume_PyAutoGUI.py  : prise en main de PyAutoGUI qui permet l'intéraction avec le jeu

agent.py  : implémentation de la classe CupHead, qui représente l'agent, avec la méthode act(), qui a un état renvoie la décision à prendre (avec epsilon greedy)

mode.py  : implémente le model CupHeadNet (CNN) qui prend l'état en entrée (succession d'images) et donne en sortie les valeurs associées aux actions (la plus grande est celle à choisir)
            deux models, celui online qui s'entraine, et celui target, copie du online et actualiser réugulièrement, qui permet le calcul de la loss


