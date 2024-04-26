import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
import pygame as pg
import random
from Games2.Player import Player, KeyboardPlayer
from Games2.Games_Inter import Game, MC_Game
from gym.spaces import Discrete,Box
from Models.RALS.RL1 import *

BOARD_COLOR = (10,255,10)
FOOD_COLOR = (255,50,0)
SNAKE_COLOR = (30,60,255)
HEAD_COLOR=(130,160,255)

clock = pg.time.Clock()

class Snakes(MC_Game):
    def __init__(self,player:Player,gui=True,size=(100,100),FPS=30,block_size=10,action_space=None,observation_space=None):
        super().__init__(action_space,observation_space)
        self.player = player
        self.gui = gui
        self.size = size
        self.fps = FPS
        
        if(gui):
            self.surface = pg.display.set_mode((size[0] * block_size,size[1] * block_size))
            self.surface.fill(BOARD_COLOR)
            self.block_size = block_size
        self.reset()

    def reset(self):
        super().reset()
        self.length = 1
        self.direction = random.randint(0,3) - 1
        if(self.direction <= 0): self.direction -= 1
        self.snake = []
        self.snake.append((self.size[0] // 2,self.size[1] // 2))
        self.eat_food()
        if(self.gui): pg.draw.rect(self.surface,BOARD_COLOR,pg.Rect((0,0),self.board_loc_to_pixels(self.size)))
        self.eat_food()

    #events
    def eat_food(self):
        food_loc = None
        while(food_loc == None or  food_loc in self.snake):
            food_loc = (random.randint(0,self.size[0] - 1),random.randint(0,self.size[1] - 1))
        if(self.gui):
            pg.draw.rect(self.surface,FOOD_COLOR,pg.Rect(self.board_loc_to_pixels(food_loc),(self.block_size,self.block_size)))
        self.food_loc = food_loc
        self.length += 1
        self.snake.append(self.calculate_next(self.snake[-1],-self.direction))
    
    def player_step(self):
        return self.step(self.player.get_move())
    def step(self,action):
        if(self.gui and self.fps != 0): clock.tick(self.fps)
        if(action == 0):
            self.direction *= -2
            if(self.direction == 4): self.direction = -1
            elif(self.direction == -4): self.direction = 1
        elif(action == 2):
            self.direction *= 2
            if(self.direction == -4): self.direction = 1
            elif(self.direction == 4): self.direction = -1

        if(self.snake[0] == self.food_loc):
            self.eat_food()

        head_loc = self.calculate_next(self.snake[0],self.direction)
        tail_loc = self.snake.pop(-1)

        if(head_loc in self.snake or head_loc[0] < 0 or head_loc[1] < 0 or head_loc[0] >= self.size[0] or head_loc[1] >= self.size[1]):
            self.game_over()

        if(self.gui): 
            pg.draw.rect(self.surface,BOARD_COLOR,pg.Rect(self.board_loc_to_pixels(tail_loc),(self.block_size,self.block_size)))
            pg.draw.rect(self.surface,SNAKE_COLOR,pg.Rect(self.board_loc_to_pixels(self.snake[0]),(self.block_size,self.block_size)))
            pg.draw.rect(self.surface,HEAD_COLOR,pg.Rect(self.board_loc_to_pixels(head_loc),(self.block_size,self.block_size)))
            
        self.snake.insert(0,head_loc)


    #help methods
    def board_loc_to_pixels(self, board_loc):
        return (board_loc[0] * self.block_size,board_loc[1] * self.block_size)
    def calculate_next(self,block,daraction):
        a = block[0]
        b = block[1]
        if(daraction % 2 == 1):
            a += daraction
        else:
            b += 1 if daraction == 2 else -1
        return (a,b)

class SnakesMC(Snakes):
    def __init__(self, player: Player, gui=True, size=(100, 100), FPS=120, block_size=10,pixel_observe=False,eat_score=100,walk_to_food_score=1,lose_score=-250):
        self.pixel_observe = pixel_observe
        
        self.eat_score =eat_score
        self.walk_to_food_score = walk_to_food_score
        self.lose_score = lose_score

        action_space = Discrete(3)
        box_shape = size[0] * size[1] if pixel_observe else 7
        observation_space = Box(low=np.full(box_shape,-1),high=np.ones(box_shape),dtype=np.int8)


        super().__init__(player, gui, size, FPS, block_size,action_space,observation_space)

    def reset(self):
        super().reset()
        return self.get_observation_space()
     
    def step(self, action):
        food_loc = self.food_loc
        super().step(action)
        
        score = self.eat_score if (food_loc != self.food_loc) else 0
        report = self.get_observation_space()
        
        if(self.pixel_observe):
            head = self.snake[0]
            if(self.direction % 2 == 1 and head[0] != self.food_loc[0]):
                score += (1 if ((head[0] < self.food_loc[0]) ^ (self.direction == -1)) else -2) * self.walk_to_food_score
            elif(self.direction % 2 == 0 and head[1] != self.food_loc[1]):
                score += (1 if ((head[1] < self.food_loc[1]) ^ (self.direction == -2)) else -2) * self.walk_to_food_score
        else:
            a = report[5] + report[6]
            if(a == -1): a = -2
            score += a * self.walk_to_food_score
        if(self.done): score = self.lose_score

        return report,score,self.done,{}
    def isHitSomething(self, loc):
        return loc[0] < 0 or loc[0] == self.size[0] or loc[1] < 0 or loc[1] == self.size[1] or loc in self.snake[1:]
    def get_observation_space(self):
        """def getDistanceUntilHit(vartical, step):
            loc = (self.snake[0][0] + (step if vartical == 0 else 0),self.snake[0][1] + (step if vartical == 1 else 0))
            return 1 if self.isHitSomething(loc) else 0

        report = np.zeros(11,dtype=np.int0)
        report[0 if self.snake[0][0] > self.food_loc[0] else 1] = 1
        report[2 if self.snake[0][1] > self.food_loc[1] else 3] = 1
        report[int(self.direction + 6 if self.direction < 0 else self.direction + 5)] = 1
        BLOCK_SIZE = 1
        report[8] = getDistanceUntilHit(1 if int(self.direction) % 2 == 0 else 0,(-BLOCK_SIZE if self.direction > 0 else BLOCK_SIZE))
        report[9] = getDistanceUntilHit(1 if (self.direction % 2 == 1)else 0,(BLOCK_SIZE if self.direction > 0 else -BLOCK_SIZE))
        report[10] = getDistanceUntilHit(1 if int(self.direction) % 2 == 0 else 0,(BLOCK_SIZE if self.direction > 0 else -BLOCK_SIZE))
        self.state = report
        return report"""
        
        if(self.pixel_observe):
            arr = np.zeros(self.size,dtype=np.int8)
            for i,loc in enumerate(self.snake):
                if(i == 0):continue
                arr[loc[0]][loc[1]] = 1
            if(not self.done):
                s_lose = self.isHitSomething(self.calculate_next(self.snake[0],self.direction))
                arr[self.snake[0][0]][self.snake[0][1]] = 10 if s_lose else 2
            arr[self.food_loc[0]][self.food_loc[1]] = -1
            arr = arr.flatten()
            return arr
        arr = np.zeros(7)
        head = self.snake[0]

        b = []
        if(self.direction % 2 == 1):
            arr[0] = self.direction
            b.append(self.calculate_next(head,self.direction * -2))
            b.append(self.calculate_next(head,self.direction))
            b.append(self.calculate_next(head,self.direction * 2))
            
            arr[5] = 1 if ((head[0] < self.food_loc[0]) ^ (self.direction == -1)) else -1
            #if(head[0] != self.food_loc[0]):
        else: 
            arr[1] = self.direction // 2
            b.append(self.calculate_next(head,self.direction // 2))
            b.append(self.calculate_next(head,self.direction))
            b.append(self.calculate_next(head,self.direction // -2))
            #if(head[1] != self.food_loc[1]):
            arr[6] = 1 if ((head[1] < self.food_loc[1]) ^ (self.direction == -2)) else -1
        
        for i,v in enumerate(b):
            if(v[0] == -1 or v[0] == self.size[0] or v[1] == -1 or v[1] == self.size[1]):
                arr[i + 2] = 1
                continue
            if(b in self.snake):
                arr[i + 2] = 1
        self.state = arr
        return arr
                

        

player = KeyboardPlayer(["a","w","d"],1)
game = SnakesMC(player,size=(500,500),block_size=3,FPS=0,pixel_observe=False,gui=False)

model = getModel_S2(game)
agent = getAgent_1(model,game)

rl = RL1(game,agent,"snakes_v2")
rl.train(verbose=1)