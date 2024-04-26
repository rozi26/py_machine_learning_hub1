from colorsys import yiq_to_rgb
from math import fabs
import re
import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

import time
import pygame
import numpy as np

from PIL import Image
from Games.Game_ex import Game
from Games.Racer.Car import Car, SmartCar ,MAX_SPEED
from gym.spaces import Box,MultiDiscrete
from Models.RALS.RL1 import *
from keras.models import load_model

MAX_MOVES = 2000
USER_FPS = 120
COMPUTER_FPS = 1200

USER_PLAY = True


RACER_FOLDER = "F:\\programing\\python\\machine_learning_hub\\Games\\Racer\\"
clock = pygame.time.Clock()

def getMapArr(id): 
    return np.asarray(Image.open(RACER_FOLDER + "MAPS\\map" + str(id) + ".png",'r'),dtype=np.uint8)[:, :, :3]

class Map():
    def __init__(self, arr):
        self.height = len(arr)
        self.width = len(arr[0])
        self.speedMap = np.zeros((self.height,self.width),dtype=np.uint8)
        self.walls = np.zeros((self.height,self.width),dtype=np.uint0)
        
        def startLineColor(color):
            return color[0] == 255 and color[1] == 0 and color[2] == 0
        def wallColor(color):
            return color[2] == 255 and color[0] == 0 and color[1] == 0
        
        start_line_found = False
        self.start_line_top = None
        self.start_line_bottom = None
        for y in range(len(arr)):
            for x in range(len(arr[0])):
                color = arr[y][x]
                self.speedMap[y][x] = 64 + round(color[0] * 0.75)
                if(wallColor(color)):
                    self.walls[y][x] = 1
                elif(not start_line_found and startLineColor(color)):
                    start_line_found = True
                    print("found top: " + str(x) + ", " + str(y))
                    self.start_line_top = (y,x)
                    y_clone = y + 1
                    x_clone = x + 1
                    while(startLineColor(arr[y_clone][x])): y_clone += 1
                    while(startLineColor(arr[y][x_clone])): x_clone += 1
                    self.start_line_bottom = (y_clone - 1,x_clone - 1)
     
        self.walls[0].fill(1)
        self.walls[:,0] = 1
        #self.walls[len(self.walls) - 1].fill(1)
        
        self.start_line_width = self.start_line_bottom[0] - self.start_line_top[0]
        print("top " + str(self.start_line_top) + ", bottom " + str(self.start_line_bottom))                

    def pointLigall(self, point): #return if the move from a to b was ligal
        if(round(point[0]) >= len(self.walls) or round(point[1]) >= len(self.walls[0])): return False 
        return True if (self.walls[round(point[0])][round(point[1])] == 0) else False
   
    def pointOnStartingLine(self, point):
       def inside(axis):
           return point[axis] >= self.start_line_top[axis] and point[axis] <= self.start_line_bottom[axis]
       return inside(0) and inside(1) 
    def squreOnStartingLine(self, points):
        def inside(axis):
            lowest = points[0][axis]
            highest = points[0][axis]
            for i in range(1,len(points)):
                lowest = min(lowest,points[i][axis])
                highest = max(highest,points[i][axis])
            #print("  [lowest: " + str(lowest) + ", heighest: " + str(highest) + "]",end="")
            return lowest <= self.start_line_bottom[axis] and highest >= self.start_line_top[axis]      
        return inside(0) and inside(1)

    def pointInMap(self, point):
        return 0 < point[0] < len(self.walls) and 0 < point[1] < len(self.walls[0])


class Racer(Game):
    def __init__(self, arr):
        action_space = Discrete(7)
        observation_space = Box(low=np.zeros(9),high=np.array([MAX_SPEED] + [1000] * 8),dtype=np.float32)
        super().__init__(action_space,observation_space,showGraph=False)
        arr = np.rot90(arr)
        arr = np.flipud(arr)
        
        self.map = Map(arr)
        self.surface = pygame.display.set_mode((self.map.height,self.map.width))
        
        image = pygame.surfarray.make_surface(arr)
        image = pygame.transform.scale(image,(self.map.height,self.map.width))
        #image = pygame.transform.flip(image,flip_x=False,flip_y=True)
        self.background = image
        self.renderBackground()
        
        self.cars = []
        self.cars.append(Car(self.map.start_line_top,(0,255,0)))
        self.cars.append(SmartCar(self.map.start_line_top,(0,127,127),self))
        self.userCar = self.cars[0] #choose the car the user has control of
         
        self.reset()#set the locations of the cars
        self.carsOnLine = np.zeros(len(self.cars),dtype=np.uint0)
       
        
        self.startTime = time.time()
        self.cantMove = False
        self.moves = 0
        self.rounds = 0
    #game methods
    def reset(self):
        self.gameRun = True
        
        #reset props
        self.cantMove = False
        self.moves = 0
        
        #reset the cars locations
        car_width = self.cars[0].size[0]
        car_height = self.cars[0].size[1] + 10
        cars_in_line = self.map.start_line_width // car_width # how many cars you can put in one line
        cars_space = (self.map.start_line_width - cars_in_line * car_width) // cars_in_line
        for i in range(len(self.cars)):
            spawn_x = self.map.start_line_top[0] + (cars_space + car_width) * (i % cars_in_line) + cars_space
            spawn_y = self.map.start_line_top[1] - 200 + car_height * (i // cars_in_line)
            self.cars[i].reset((spawn_x,spawn_y))
        return self.getBotObservation()
    def gameOver(self):
        self.gameRun = False
        
    def runFrame(self):
        self.moveAllCars(True)
        self.moves += 1
        free = self.userCar.move(self.map)
        if(not free):
            self.cantMove = True
            self.gameOver()
            
        return self.userCar.speed  - 2 + 2 * self.userCar.efficency # return bot reward for the frame

    #game helpers
    def checkPass(self, i): # check if the car in index i pass the finish line
        car = self.cars[i]
        onLine = self.map.squreOnStartingLine(car.wheels)
        if(onLine):
            backWheels = self.map.pointOnStartingLine(car.wheels[2]) or self.map.pointOnStartingLine(car.wheels[3])
            if(backWheels and self.carsOnLine[i] and car.wheels[0][1] < self.map.start_line_top[1]):
                self.carsOnLine[i] = 0
                return True
                
            elif(not backWheels):
                self.carsOnLine[i] = 1
        else: self.carsOnLine[i] = 0
        return False
            
    def moveAllCars(self, withoutUser=False):
        for car in self.cars:
            if(withoutUser and self.userCar != car):
                car.move(self.map)
            
    #rendering
    def renderBackground(self):
        self.surface.blit(self.background,(0,0))
    def renderCar(self, car):
        object = car.getRect()
        self.surface.blit(object[0],object[1])
    
    def render(self, mode=""):
        if(not USER_PLAY): clock.tick(COMPUTER_FPS)
        self.renderBackground()
        for car in self.cars:
            self.renderCar(car)
        pygame.display.update()
        pygame.event.pump()
    
    #user methods
    def player_input(self):
        car = self.userCar
        keys = pygame.key.get_pressed()
        car.actions[0] = 1 if (keys[pygame.K_UP] or keys[pygame.K_w]) else 0
        car.actions[1] = 1 if (keys[pygame.K_DOWN] or keys[pygame.K_s]) else 0
        car.actions[2] = 1 if (keys[pygame.K_LEFT] or keys[pygame.K_a]) else 0
        car.actions[3] = 1 if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) else 0

    def getGameScore(self): #return the game score when the game is over
        return time.time() - self.startTime
    
    
    #machine learning methods
    def adouptBotInput(self,action):
        #print("A: [" + str(action) + "]")
        self.userCar.actions.fill(0)
        if(action != 0):
            if(action < 5):
                self.userCar.actions[action - 1] = 1
            else:
                self.userCar.actions[0] = 1
                self.userCar.actions[action - 3] = 1
        
    def canContinue(self):
        return self.moves < MAX_MOVES and not self.checkPass(0)
    
    def getBotObservation(self, car = None):
        if(car == None): car = self.userCar;
        def getDistanteOfRoad(point_o, M):
            point = (point_o[0],point_o[1])
            count = 0
            XM, YM = M
            while(self.map.pointInMap(point) and self.map.speedMap[round(point[0]),round(point[1])] == 255):
                point = (point[0] + XM, point[1] + YM)
                count += 1
            return count
        def getDistanceFree(point_o, M):
            point = (point_o[0],point_o[1])
            count = 0
            XM, YM = M
            while(self.map.pointInMap(point) and self.map.pointLigall(point)):
                point = (point[0] + XM, point[1] + YM)
                count += 1
            return count
        
        degree = car.degree
        
        report = np.zeros(9,dtype=np.float32)
        report[0] = car.speed # the bot's car speed

        F_M = car.getDegreeMultypliers()
        L_M = car.getDegreeMultypliers(degree - 0.25)
        R_M = car.getDegreeMultypliers(degree + 0.25)
        B_M = car.getDegreeMultypliers(degree + 0.5)
        start_point = car.wheels[0]
        
        report[1] = getDistanteOfRoad(start_point,F_M) #length of road forward
        report[2] = getDistanteOfRoad(start_point,L_M) #length of road from the left
        report[3] = getDistanteOfRoad(start_point,R_M) #length of road from the right
        report[4] = getDistanteOfRoad(start_point,B_M) #length of road backward
        
        report[5] = getDistanceFree(start_point,F_M)
        report[6] = getDistanceFree(start_point,L_M)
        report[7] = getDistanceFree(start_point,R_M)
        report[8] = getDistanceFree(start_point,B_M)
        
        #print("\r report: [" + str(report[2]) + "],[" + str(report[3]) + "],[" + str(report[4]) + "],[" + str(report[5]) + "],[" + str(report[6]) + "],[" + str(report[7]) + "],[" + str(report[8]) + "],[" + str(report[9]) + "]                    ",end="")
        return report
    
    def getGameOverReward(self):
        if(self.moves >= MAX_MOVES): return -250
        elif(self.cantMove): return -1500
        return 10000000 / (time.time() - self.startTime())

race = Racer(getMapArr(1))

if(USER_PLAY):
    while(True):
        clock.tick(USER_FPS) # wait for frame
        race.player_input() # get user input
        race.runFrame() # run frame
        race.render() # render the game
        
        #just for report
        race.getBotObservation()
        
        if(not race.gameRun):
            race.reset()
else:
    model = getModel_S2(race)
   # model = load_model("F:\\programing\\python\\machine_learning_hub\\_not code_\\RL_MODELS\\car_race_4\\model")
    agent = getAgent_1(model,race)
    rl = RL1(race,agent,name="car_race")
    rl.train(1000000)
    rl.save()