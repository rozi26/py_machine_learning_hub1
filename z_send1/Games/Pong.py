import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

import pygame as pg
import numpy as np

from Games.General.ObjectMove import BoundsMover, Retangle, Line
from Games.Game_ex import Game
from gym.spaces import Discrete, Box

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

RACKET_WIDTH = 30
RACKET_HEIGHT = 100
RACKET_SPEED = 10
RACKET_MARGIN = 5
RACKET_COLOR = (0,0,255)

BALL_SPEED = 30
BALL_SIZE = 15
BALL_COLOR = (255,255,255)
SPIN_EFFECT = 0.03
SPEED_INCRISE = 1
RANDOM_CHANGE = 0.01

BACKGROUND_COLOR = (10,10,10)

USER_FPS = 30
COMPUTER_FPS = 240

T_RACKET1 = 1
T_RACKET2 = 2

VS_AGENT_L = "F:\\programing\\python\\machine_learning_hub\\_not code_\\RL_MODELS\\pong_4\\model"
VS_AGENT_R = "F:\\programing\\python\\machine_learning_hub\\_not code_\\RL_MODELS\\pong_6\\model_5"

AGENT_L = None
AGENT_R = None

timer = pg.time.Clock()

#rackets
class Racket():
    def __init__(self, spawn):
        self.y = spawn[1]
        self.x = spawn[0]
        self.size = (RACKET_WIDTH,RACKET_HEIGHT)
        self.lastMove = 0

    def move(self, duraction = 0):
        if(duraction == 1):
            self.y = max(self.y - RACKET_SPEED,0)
        elif(duraction == -1):
            self.y = min(self.y + RACKET_SPEED,SCREEN_HEIGHT - self.size[1])
        self.lastMove = duraction

    def hitBall(self, p1, p2):
        v = max(p1[0],p2[0]) > self.x and min(p1[0],p2[0]) < self.x + self.size[0]
        h = max(p1[1],p2[1]) > self.y and min(p1[1],p2[1]) < self.y + self.size[1]
        return v and h
        
class User_Racket(Racket): #racket that move by user input
    def __init__(self, spawn, controls = 0):
        super().__init__(spawn)
        self.controls = controls
    
    def move(self):
        keys = pg.key.get_pressed()
        duraction = 0
        if((self.controls != 1 and keys[pg.K_UP]) or (self.controls != -1 and keys[pg.K_w])): duraction += 1
        if((self.controls != 1 and keys[pg.K_DOWN]) or (self.controls != -1 and keys[pg.K_s])): duraction -= 1
        super().move(duraction)

class Smart_Racket(Racket): #racket that move by an algorithm
    def __init__(self, spawn, game):
        super().__init__(spawn)
        self.defend_right = spawn[0] > SCREEN_WIDTH / 2
        self.game = game
        self.last_ball_location = (0,0)
    
    def move(self):
        report = self.game.getBotObservation()
        
        #check if the ball move to my duraction
        to_me = (report[3] > 0.5) ^ self.defend_right
        if(not to_me): return

        ball_loc = (report[4],report[5])
        line = Line(self.last_ball_location,ball_loc)
        self.last_ball_location = ball_loc
        if(line.vartical): return

        ball_y = line.getY(self.x + (0 if self.defend_right else self.size[0]))
        pos = None
        if(ball_y < 0):
            pos = abs(ball_y)
        elif(ball_y > SCREEN_HEIGHT):
            pos = SCREEN_HEIGHT * 2 - ball_y
        else: pos = ball_y

        if(self.y + self.size[1] / 2 > pos):
            super().move(1)
        else:
            super().move(-1)
        
class MC_Racket(Racket): #racket for machine learning in traing
    def __init__(self, spawn):
        super().__init__(spawn)
        self.command = 0
    def move(self):
        super().move(self.command)
    
class Agent_Racket(Racket): #racket for machine learning
    def __init__(self, spawn, agent, game):
        super().__init__(spawn)
        self.game = game
        self.agent = agent
    
    def move(self):
        super().move(self.agent.forward(game.getBotObservation()) - 1)

class Ball(BoundsMover):
    def __init__(self, game):
        start_degree = np.random.uniform(0,0.125) + 0.1875
        if(np.random.randint(0,2) == 0): start_degree += 0.5
        super().__init__(SCREEN_WIDTH,SCREEN_HEIGHT,(BALL_SIZE,BALL_SIZE),start_degree ,BALL_SPEED)
        self.loc = ((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.prevLoc = self.loc
        self.game = game

    def move(self, rackets):
        self.prevLoc = self.loc

        retangels = []
        for racket in rackets:
            retangels.append(Retangle(racket.size, (racket.x,racket.y)))
        self.loc, hits = super().nextPoint(self.loc,retangels,self.degree)

        if(hits[4] and rackets[0].lastMove != 0):
            self.changeDegree(self.degree + SPIN_EFFECT * rackets[0].lastMove)
            self.speed += SPEED_INCRISE
        if(hits[5] and rackets[1].lastMove != 0):
            self.changeDegree(self.degree + SPIN_EFFECT * rackets[1].lastMove)
            self.speed += SPEED_INCRISE
        if(hits[2] or hits[1]):
            self.degree += RANDOM_CHANGE * np.random.uniform(-1,1)
        self.degree %= 1
        return hits
        
class Pong(Game):

    def __init__(self):
        action_space = Discrete(3)
        observation_space = Box(low=np.zeros(6),high=np.ones(6),dtype=np.float32)
        super().__init__(action_space,observation_space)

        self.surfce = pg.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.surfce.fill(BACKGROUND_COLOR)

        global AGENT_L, AGENT_R
        if((T_RACKET1 == 3 and AGENT_L == None)  or (T_RACKET2 == 3 and AGENT_R == None)):
            from keras.models import load_model
            from Models.RALS.RL1 import getAgent_1
            if(T_RACKET1 == 3):
                AGENT_L = getAgent_1(load_model(VS_AGENT_L),self)
            if(T_RACKET2 == 3):
                AGENT_R = getAgent_1(load_model(VS_AGENT_R),self)

        self.reset()
    
    def reset(self):
        self.ball = Ball(self)
        self.gameRun = True
        pg.draw.rect(self.surfce,BACKGROUND_COLOR,pg.Rect((0,0),(SCREEN_WIDTH,SCREEN_HEIGHT)))

        #set the rackets
        r1_spawn = ((RACKET_MARGIN,(SCREEN_HEIGHT - RACKET_HEIGHT) // 2))
        r2_spawn = (SCREEN_WIDTH - RACKET_WIDTH - RACKET_MARGIN,r1_spawn[1])
        def getRacketType(id, left):
            spawn = r1_spawn if left else r2_spawn
            if(id == 0):
                return User_Racket(spawn,(-1 if T_RACKET2 == 0 else 0) if left else (1 if T_RACKET1 == 0 else 0))
            if(id == 1):
                return Smart_Racket(spawn,self)
            if(id == 2):
                rack = MC_Racket(spawn)
                self.MC_RACKET = rack
                return rack
            if(id == 3):
                return Agent_Racket(spawn,AGENT_L if left else AGENT_R,self)
        
        r1 = getRacketType(T_RACKET1,True)
        r2 = getRacketType(T_RACKET2,False)
        self.racket1 = r1
        self.racket2 = r2
        return self.getBotObservation()

    def gameOver(self):
        self.gameRun = False
    #runing

    #run frame for the machine learning
    def runFrame(self):
        self.moveRackets()
        hits = self.moveBall()
        score = 0
        HIT_ADD = 0 if T_RACKET1 == 2 else 1
        if(hits[0] or hits[1]):
            #score = 100 * (-1.2/((self.ball.speed - BALL_SPEED) + 1) if hits[1] else 1)
            score = 100 * (-1 if(hits[HIT_ADD]) else 1)
            self.gameOver()
        if(hits[4 + HIT_ADD]):
            score = 2
        return score


    def moveRackets(self):
        self.racket1.move()
        self.racket2.move()
    def moveBall(self):
        return self.ball.move([self.racket1,self.racket2])

    #rendering
    def render(self, mode=None):
        timer.tick(COMPUTER_FPS if ((T_RACKET1 == 2 or T_RACKET2 == 2) and T_RACKET1 != 0 and T_RACKET2 != 0) else USER_FPS)
        self.renderRacket(self.racket1)
        self.renderRacket(self.racket2)
        self.renderBall()
        pg.display.update()
        pg.event.pump()

    def renderRacket(self, racket):
        pg.draw.rect(self.surfce,BACKGROUND_COLOR,pg.Rect((racket.x,0),(racket.size[0],SCREEN_HEIGHT)))
        pg.draw.rect(self.surfce,RACKET_COLOR,pg.Rect((racket.x,racket.y),(racket.size)))

    def renderBall(self):
        pg.draw.rect(self.surfce,BACKGROUND_COLOR,pg.Rect((self.ball.prevLoc),(BALL_SIZE,BALL_SIZE)))
        pg.draw.rect(self.surfce,BALL_COLOR,pg.Rect((self.ball.loc),(BALL_SIZE,BALL_SIZE)))


    #AI
    def getBotObservation(self):
        report = np.zeros(6)

        report[0] = self.racket1.y
        report[1] = self.racket2.y
        report[2] = self.ball.speed
        report[3] = self.ball.degree
        report[4] = self.ball.loc[0]
        report[5] = self.ball.loc[1]
        return report
        
    def adouptBotInput(self,action):
        self.MC_RACKET.command = action - 1

    def canContinue(self):
        return True
    def getGameScore(self):
        return (1 if((self.ball.loc[0] < SCREEN_WIDTH / 2) ^ (T_RACKET1 == 2)) else -1)* (self.ball.speed - BALL_SPEED)


#declare the game
game = Pong()
print(AGENT_L)

if(T_RACKET1 != 2 and T_RACKET2 != 2):
    while(True):
        game.runFrame()
        if(not game.gameRun): game.reset()
        game.render()
    exit()

from Models.RALS.RL1 import getAgent_1,getModel_S1,RL1

model = getModel_S1(game)
agent = getAgent_1(model,game)
    
RL = RL1(game,agent,"pong")
for i in range(200):
    RL.train(100000,visualize=False)
    RL.save(True)

print("done")