import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")
from gym.spaces import Discrete, Box
from IPython import display
from Game_ex import Game
from keras.models import load_model

import matplotlib.pyplot as plt
import pygame
import random
import numpy as np

USER_PLAY = False

MARK_PROGRESS = True
KEYBORAD_ON = True
COMPUTER_FPS = 120


BORAD_SIZE = 1000
BLOCK_SIZE = 10
FOOD_SIZE = 10

BORAD_COLOR = (110,100,5)
FOOD_COLOR = (255,255,0)
SNAKE_COLOR = (0,255,255)
SNAKE_HEAD_COLOR = (0,0,0)

DONE_EAT_FAIL = 1000
PLAYER_TICK_SPEED = 10

pygame.init()
clock = pygame.time.Clock()
class snakes_game(Game):

    #building methods
    def __init__(self):
        action_space = Discrete(3)
        observation_space = Box(low=np.zeros(11),high=np.ones(11),dtype=np.int0)
        super().__init__(action_space,observation_space)
        
        self.surface = pygame.display.set_mode((BORAD_SIZE,BORAD_SIZE))
        self.surface.fill(BORAD_COLOR)


        self.FPS = COMPUTER_FPS
        
        self.length = 1
        self.direction = 1
        self.dontEat = 0
        self.foodLoc = (0,0)
        self.snakeLocs = []

        self.state = self.reset()
        pygame.display.flip()
    def reset(self): #resat the game
        for p in self.snakeLocs:
            pygame.draw.rect(self.surface,BORAD_COLOR,pygame.Rect(p,(BLOCK_SIZE,BLOCK_SIZE)))
        self.snakeLocs = [(BORAD_SIZE / 2,BORAD_SIZE / 2)]
        self.length = 1
        self.direction = 1
        self.dontEat = 0
        self.eatFood()
        self.eatFood()
        self.eatFood()
        self.gameRun = True
        self.done = False
        report = self.getBotObservation()
        return report
    
    #game events
    def eatFood(self): #when the sanek eat food
        self.dontEat = 0
        snakeTail = self.snakeLocs[len(self.snakeLocs) - 1]
        newTail = self.getNextLocation(snakeTail,-self.direction)
        self.snakeLocs.append(newTail)
        self.length += 1
        #delete the old food and draw the new food
        position = (random.randint(0,BORAD_SIZE - FOOD_SIZE),random.randint(0,BORAD_SIZE - FOOD_SIZE))
        prevFood = pygame.Rect((self.foodLoc[0],self.foodLoc[1]),(FOOD_SIZE,FOOD_SIZE))
        pygame.draw.rect(self.surface,BORAD_COLOR,prevFood)
        newFood = pygame.Rect((position[0],position[1]),(FOOD_SIZE,FOOD_SIZE))
        pygame.draw.rect(self.surface,FOOD_COLOR,newFood)
        self.foodLoc = position  
    def gameOver(self): #when the game is over
        self.gameRun = False
    
    """def step(self, action): #for the machine learning
        # 0 - unclockwise 1 - same 2 - clockwise
        if(self.FPS != 0): clock.tick(self.FPS)
        if(action == 0): 
            if(self.direction % 2 == 1): self.direction += self.direction
            else: self.direction = -(self.direction / 2)
        elif(action == 2):
            if(self.direction % 2 == 1): self.direction = -self.direction*2
            else: self.direction /= 2
        reward = self.snakeMove()
        if(self.dontEat == DONE_EAT_FAIL): self.gameOver()
        #get the report for the machine learning
        return self.getMCReport(), reward, self.done, {} """
    
    def runFrame(self):# run frame of the game (the input is declare in player_input method or by the bot)
        PHL = self.snakeLocs[0]
        reward = (1 if self.direction * (self.snakeLocs[0][1] - self.foodLoc[1]) <= 0 else -1) if self.direction % 2 == 1 else (1 if self.direction * (self.snakeLocs[0][0] - self.foodLoc[0]) <= 0 else -1)
        if(PHL[0] + BLOCK_SIZE>= self.foodLoc[0] and PHL[0] <= self.foodLoc[0] + FOOD_SIZE and PHL[1] + BLOCK_SIZE >= self.foodLoc[1] and PHL[1] <= self.foodLoc[1] + FOOD_SIZE):
            self.eatFood()
            reward = 25
        else: self.dontEat += 1
        NHL = self.getNextLocation(PHL,self.direction)
        if(self.isHitSomething(NHL)): 
            self.gameOver() 
            reward = -100
        else:
            self.snakeLocs.insert(0,NHL)
            if(self.FPS > 0): #draw the snake
                pygame.draw.rect(self.surface, BORAD_COLOR,pygame.Rect(self.snakeLocs[len(self.snakeLocs) - 1],(BLOCK_SIZE,BLOCK_SIZE)))
                pygame.draw.rect(self.surface,SNAKE_HEAD_COLOR,pygame.Rect(NHL,(BLOCK_SIZE,BLOCK_SIZE)))
                pygame.draw.rect(self.surface,SNAKE_COLOR,pygame.Rect(PHL,(BLOCK_SIZE,BLOCK_SIZE)))
            self.snakeLocs.pop()
        return reward
    
    #help methods
    def isHitSomething(self, NHL):
        return (NHL in self.snakeLocs[1:len(self.snakeLocs) - 1]) or NHL[0] >= BORAD_SIZE or NHL[0] < 0 or NHL[1] >= BORAD_SIZE or NHL[1] < 0
    def getNextLocation(self, block, direction):
        addY = ((direction) if direction % 2 == 1 else 0) * BLOCK_SIZE
        addX = ((direction) if direction % 2 == 0 else 0) * BLOCK_SIZE / 2
        return (block[0] + addX,block[1] + addY)
    
    #user methods
    def player_input(self):
        for event in pygame.event.get():
            if(event.type == pygame.KEYDOWN):
                if(event.key == pygame.K_w or event.key == pygame.K_UP): self.direction = -1
                elif(event.key == pygame.K_s or event.key == pygame.K_DOWN): self.direction = 1
                elif(event.key == pygame.K_d or event.key == pygame.K_RIGHT): self.direction = 2
                elif(event.key == pygame.K_a or event.key == pygame.K_LEFT): self.direction = -2
                elif(event.key == pygame.K_r): self.reset()
                elif(event.key == pygame.K_q): exit()
                elif(event.key == pygame.K_p): 
                    if(self.FPS == 0):
                        pygame.draw.rect(self.surface,BORAD_COLOR,pygame.Rect((0,0),(BORAD_SIZE,BORAD_SIZE)))
                        pygame.draw.rect(self.scores,FOOD_COLOR,pygame.Rect(self.foodLoc),(FOOD_SIZE,FOOD_SIZE))
                        if(len(self.snakeLocs) > 0):
                            pygame.draw.rect(self.surface,SNAKE_HEAD_COLOR,pygame.Rect(self.snakeLocs[0],(BLOCK_SIZE,BLOCK_SIZE)))
                            for i in range(1,len(self.snakeLocs)):
                                pygame.draw.rect(self.surface,SNAKE_COLOR,pygame.Rect(self.snakeLocs[i],(BLOCK_SIZE,BLOCK_SIZE)))
                        self.FPS = COMPUTER_FPS
                    else: self.FPS = 0
    def render(self, mode): #render the game
        if(self.FPS > 0): pygame.display.update()

    #machine learning methods
    def canContinue(self):
        return self.dontEat < DONE_EAT_FAIL
    def adouptBotInput(self, action):
        if(self.FPS != 0): clock.tick(self.FPS)
        if(action == 0): 
            if(self.direction % 2 == 1): self.direction += self.direction
            else: self.direction = -(self.direction / 2)
        elif(action == 2):
            if(self.direction % 2 == 1): self.direction = -self.direction*2
            else: self.direction /= 2
    def getBotObservation(self): #return the input for the bot
        def getDistanceUntilHit(vartical, step):
            loc = (self.snakeLocs[0][0] + (step if vartical == 0 else 0),self.snakeLocs[0][1] + (step if vartical == 1 else 0))
            return 1 if self.isHitSomething(loc) else 0

        report = np.zeros(11,dtype=np.int0)
        report[0 if self.snakeLocs[0][0] > self.foodLoc[0] else 1] = 1
        report[2 if self.snakeLocs[0][1] > self.foodLoc[1] else 3] = 1
        report[int(self.direction + 6 if self.direction < 0 else self.direction + 5)] = 1
        report[8] = getDistanceUntilHit(1 if int(self.direction) % 2 == 0 else 0,(-BLOCK_SIZE if self.direction > 0 else BLOCK_SIZE))
        report[9] = getDistanceUntilHit(1 if (self.direction % 2 == 1)else 0,(BLOCK_SIZE if self.direction > 0 else -BLOCK_SIZE))
        report[10] = getDistanceUntilHit(1 if int(self.direction) % 2 == 0 else 0,(BLOCK_SIZE if self.direction > 0 else -BLOCK_SIZE))
        self.state = report
        return report

    #visual methods
    def getGameScore(self):
        return self.length

if(MARK_PROGRESS): plt.ion()
def createProgressBar(scores, avrage):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.title("train stats")
    plt.xlabel("number of games")
    plt.ylabel("scores")
    plt.plot(scores)
    plt.plot(avrage)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    #plt.text(len(avrage)-1, avrage[-1], str(avrage[-1]))
    plt.autoscale()
    plt.show(block=False)
    plt.pause(.1)
    

game = snakes_game()
def runGame():
    while(game.gameRun):
        clock.tick(PLAYER_TICK_SPEED)
        if(KEYBORAD_ON): game.player_input()
        game.runFrame()
        if(not game.gameRun): 
            game.getMCReport()
            game.reset()
        pygame.display.update()
if(USER_PLAY): runGame()
else:
    from Models.RALS.RL1 import getAgent_1,getModel_S2,RL1
    
    model = getModel_S2(game)
    #model = load_model("F:\\programing\\python\\machine_learning_hub\\_not code_\\RL_MODELS\\snakes_1\\model")
    agent = getAgent_1(model,game)
    
    RL = RL1(game,agent,"snakes")
    RL.train(250000)
    RL.save()
    
    print("done")
    