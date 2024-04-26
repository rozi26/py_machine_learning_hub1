import pygame as pg

class Game:
    def __init__(self):
        self.done = True

    def reset(self):
        self.done = False
    def game_over(self):
        self.done = True
    def render(self, mode=None): #render the game
        if(self.gui):
            pg.display.update()

    def user_play(self):
        while(True):
            self.player_step()
            if(self.done):
                self.reset()
            if(self.gui):
                pg.display.update()
    
class MC_Game(Game):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
    
    def get_observation_space(self):
        print("ERROR! try to observe date from the intarface, probobly you game class doesn't have \"get_observe_data\" method")