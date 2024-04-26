from re import S
import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")
from turtle import speed
from pygame import Rect, Surface, SRCALPHA
from pygame.transform import rotate, scale


import numpy as np
import math

MIN_SPEED = 0
MAX_SPEED = 10 # the maximum pixels to move per frame
ECELERATION = 0.2 # the speed gain per frame while ecelerating

BREAKING_SPEED = 0.2 # the speed lose per frame while breaking

TURNING_RADIUS = 0.005 # the change in degree per frame while turning

DRAG_CONSTENT = 0.99

class Car():
    def __init__(self, spawn, color):
        self.size = (20,60)
        self.location = spawn
        self.color = color
        
        self.object = Surface(self.size,SRCALPHA)
        self.object.fill(color)
        self.object = scale(self.object,self.size)
        
        self.actions = np.zeros(4,dtype=np.uint0)
        
        self.speed = MIN_SPEED
        self.degree = 0
        
        self.center_to_whell_distance = math.sqrt(((self.size[0] * self.size[0]) + (self.size[1] * self.size[1])) / 4)
        self.center_to_whell_slopeConstent = 1 - (math.atan(self.size[0] / self.size[1]) * (0.5 / math.pi))
        self.wheels = self.getWheelsLocations()
        print(self.center_to_whell_slopeConstent)
    
    #setting
    def reset(self, location):
        self.location = location
        self.degree = 0
        self.speed = 0
    
    #moveing
    def move(self, map, sef=None):
        self.wheels = self.getWheelsLocations()
        self.efficency = self.getWheelsEfficency(self.wheels,map.speedMap)
        
        #print("\r" + str(self.wheels[0]) + " (" + str(efficency) + ") [" + str(self.speed) + "]",end="")
        
        CURRENT_MAX_SPEED = MAX_SPEED * self.efficency
        if(self.actions[0] == 1):
            self.speed += ECELERATION * self.efficency
            if(self.speed > CURRENT_MAX_SPEED): self.speed = CURRENT_MAX_SPEED
        if(self.actions[1] == 1):
            self.speed -= BREAKING_SPEED
        if(self.actions[2] == 1):
            self.degree -= TURNING_RADIUS
            self.degree %= 1
        if(self.actions[3] == 1):
            self.degree += TURNING_RADIUS
            self.degree %= 1

        self.speed *= DRAG_CONSTENT
        if(self.speed < MIN_SPEED): self.speed = MIN_SPEED
        
        XM, YM = self.getDegreeMultypliers()
        
        if(not self.moveLigal(map,self.wheels,XM,YM)):
            self.speed = 0
            return False
            
        ADD_X = self.speed * XM
        ADD_Y = self.speed * YM
        self.location = (self.location[0] + ADD_X,self.location[1] + ADD_Y)
        
        return True
        
    def getDegreeMultypliers(self, degree=None):
        M = self.degree % 1 if degree == None else degree % 1
        xm = 0
        ym = 0
        if(M <= 0.25):  xm = M * 4
        elif(M <= 0.75):xm = 1 - (M - 0.25) * 4
        else:           xm = (M - 0.75) * 4 - 1
        
        if(M <= 0.5):   ym = M * 4 - 1
        else:           ym = 1 - (M - 0.5) * 4
        return xm,ym    
    
    def getWheelsEfficency(self,wheels, speedMap):
        sum = 0
        for wheel in wheels:
            if(wheel[0] < len(speedMap) and wheel[1] < len(speedMap[0])): sum += speedMap[wheel[0],wheel[1]]
        return sum / (len(wheels) * 255)   
    def getWheelsLocations(self):
        middle = (self.location[0] + self.size[0] // 2,self.location[1] + self.size[1] // 2)
        def getWhell(degree):
            XM, YM = self.getDegreeMultypliers(degree)
            return (round(middle[0] + XM * self.center_to_whell_distance), round(middle[1] + YM * self.center_to_whell_distance))
        wheels = []
        deg = self.degree + self.center_to_whell_slopeConstent
        wheels.append(getWhell(deg)) # the top left whell
        deg -= self.center_to_whell_slopeConstent * 2
        wheels.append(getWhell(deg)) # the top right wheel
        deg += 0.5
        wheels.append(getWhell(deg)) # the bottom left wheel
        deg -= self.center_to_whell_slopeConstent * 2
        wheels.append(getWhell(deg))
        return wheels
    
    def moveLigal(self, map, wheels, MX, MY):
        def pointMoveLigal(point):
            count = 0
            if(not map.pointLigall(point)): return False
            while(count < self.speed):
                point = (point[0] + MX,point[1] + MY)
                count += 1
                if(not map.pointLigall(point)): return False
            return True
        for whell in wheels:
            if(not pointMoveLigal((whell[0],whell[1]))): return False
        return True
    #renderung
    def getRect(self):
       rotate_image = rotate(self.object,(1 - self.degree) * 360)
       new_rect = rotate_image.get_rect(center=self.object.get_rect(topleft=self.location).center)
       return rotate_image, new_rect
    
    #props
    def getLocation(self):
        return (round(self.location[0]),round(self.location[1]))
    
class SmartCar(Car): #drive itself by an algorithm
    def __init__(self, spawn, color, racer):
        self.racer = racer
        super().__init__(spawn, color)
        
    def move(self,map):
        def getDistanteOfRoad(point_o, M):
            point = (point_o[0],point_o[1])
            count = 0
            XM, YM = M
            while(map.pointInMap(point) and map.speedMap[round(point[0]),round(point[1])] == 255):
                point = (point[0] + XM, point[1] + YM)
                count += 1
            return count
        
        self.actions.fill(0) #reset the previus inputs
        
        inputs = self.racer.getBotObservation(self)
        breakingDistance = (self.speed / 2) * (self.speed // BREAKING_SPEED + 1);
        print("\r" + str(breakingDistance) + " - " + str(inputs[1]),end="")
        
        if(inputs[1] > breakingDistance):
            self.actions[0] = 1
        if(inputs[2] >= inputs[3]):
            dis = getDistanteOfRoad(self.wheels[0], self.getDegreeMultypliers(self.degree - TURNING_RADIUS))
            if(dis >= inputs[1]):
                self.actions[2] =1
                if(dis >= breakingDistance): self.actions[0] = 1
        else:
            dis = getDistanteOfRoad(self.wheels[0], self.getDegreeMultypliers(self.degree + TURNING_RADIUS))
            if(dis >= inputs[1]):
                self.actions[3] =1
                if(dis >= breakingDistance): self.actions[0] = 1
        if(not self.actions[0]): self.actions[1] = 1
            
        
        super().move(map)
        
        
    