import keyboard
import numpy as np

class Player():
    def get_move(self, data):
        pass

class KeyboardPlayer(Player):
    def __init__(self,keys:list=[],neutral=0,press_block=True) -> None:
        
        if(not type(keys[0]) == list):
            fix = []
            for key in keys:
                fix.append([key])
            keys = fix
            
        self.keys = keys
        self.neutral = neutral
        self.press_block = press_block
        self.prev = None

    def get_move(self, data=None):
        output = np.zeros(len(self.keys))
        for i,key in enumerate(self.keys):
            for cr in key:
                if(keyboard.is_pressed(cr)):
                     output[i] = 1
                     break
        if(self.press_block):
            if((not self.prev is None) and np.array_equal(output,self.prev)): return self.neutral
            self.prev = output
        if(np.sum(output)) != 1: return self.neutral
        for i,v in enumerate(output):
            if(v): return i
