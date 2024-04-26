import math
import numpy as np

#linar line
class Line():

    def __init__(self, p1,p2):
        self.x_min = min(p1[0],p2[0])
        self.x_max = max(p1[0],p2[0])

        dev = (p1[0] - p2[0])
        self.vartical = dev == 0
        if(self.vartical):
            self.y_max = max(p1[1],p2[1])
            self.m = 0
        else:
            self.m = (p1[1] - p2[1]) / dev
            
        self.b = self.m * -p1[0] + p1[1]
    
    #return the point where this line hit "line" (if the lines never hit return None)
    def getHit(self, line):

        if((self.vartical and line.vartical) or (self.m == line.m and not self.vartical and not line.vartical)): return None
        if(self.vartical or line.vartical):
            if(self.vartical): return line.getVarticalHit(self)
            else: return self.getVarticalHit(line)

        #if both lines are not vartical
        hit_x = (line.b - self.b) / (self.m - line.m)
        if(not self.getXInRange(hit_x) or not line.getXInRange(hit_x)): return None
        return (hit_x,self.getY(hit_x))

    def getY(self, x):
        return self.m * x + self.b
    def getXInRange(self, x):
        return x >= self.x_min and x < self.x_max
    def __str__(self):
        return "y = " + str(self.m) + "x + " + str(self.b) + " {x >= " + str(self.x_min) + "} {x <= " + str(self.x_max) + "}"

    def getVarticalHit(self, varticalLine):
        if(varticalLine.x_min < self.x_min or varticalLine.x_min > self.x_max): return None
        hit_y = self.getY(varticalLine.x_min)
        if(hit_y > varticalLine.y_max or hit_y < varticalLine.b): return None
        return (varticalLine.x_min,hit_y)


class Retangle():
    def __init__(self, size, loc):
        self.size = size
        self.loc = loc

    #return the point and the engale that object that move from p1 to p2 will hit
    def getHit(self, p1, p2):
        line = Line(p1,p2)
        
        global hit_point
        global hit_distance
        hit_point = None
        hit_distance = 0
        def betterPoint(new_point):
            
            global hit_point
            global hit_distance
            if(new_point == None): return
            distance = getDistanceBetweenPoints(p1,new_point)
            if(hit_point == None or distance < hit_distance):
                
                #input("press enter to ")
                hit_point = new_point
                hit_distance = distance
        
        c1 = self.loc
        c2 = (c1[0],c1[1] + self.size[1])
        c4 = (c1[0] + self.size[0],c1[1])
        c3 = (c4[0],c2[1])
       # print("c1: " + str(c1) + ",c2: " + str(c2) + ",c3: " + str(c3) + ",c4: " + str(c4))

        betterPoint(line.getHit(Line(c1,c2)))
        betterPoint(line.getHit(Line(c2,c3)))
        betterPoint(line.getHit(Line(c4,c3)))
        betterPoint(line.getHit(Line(c4,c1)))
    
        if(hit_point == None): return None
        return hit_point,0

class Mover():

    def __init__(self, degree = 0, speed=0):
        self.degree = degree
        self.speed = speed

        self.degreeChanged = False
        self.mx, self.my = getDegreeMultypliers(self.degree)

    def changeDegree(self, to):
        self.degree = to
        self.degreeChanged = True

    def nextPoint(self,loc, degree=None, speed=None):
        
        if(degree != None and degree != self.degree):
            self.changeDegree(degree)
        if(speed != None): 
            self.speed = speed

        if(self.degreeChanged):
            self.degreeChanged = False
            self.mx, self.my = getDegreeMultypliers(self.degree)

        return (loc[0] + self.speed * self.mx,loc[1] + self.speed * self.my)

class BoundsMover(Mover):
    def __init__(self, screen_width, screen_height,size=(0,0), degree=0, speed=0):
        super().__init__(degree, speed)
        self.width = screen_width
        self.height = screen_height
        self.size = size

    def nextPoint(self, loc,opsticals=None, degree=None, speed=None):

        #record the hits [0] - left vartical [1] - right vartical [2] - top horizontal [3] - bottom horizonal [>3] opsticals
        hits = np.zeros(4 + (0 if opsticals == None else len(opsticals)),dtype=np.uint0)

        next = super().nextPoint(loc,degree,speed)
            
        #hit opsticals
        hits_record = 4
        for opstical in opsticals:
            res = opstical.getHit(loc,next)
            if(res != None):
                hits[hits_record] = 1
                hit, slope = res
                super().changeDegree(getBounceDegree(self.degree,slope))
                if(slope == 0):
                    next = (next[0] + ((hit[0] - next[0]) * 2),next[1])
            hits_record += 1


        #hit the horizontal wall
        if(next[0] < 0 or next[0] + self.size[0] >= self.width):
            super().changeDegree(getBounceDegree(self.degree,0))
            hits[(0 if next[0] < 0 else 1)] = 1
        #hit the vartical line
        if(next[1] < 0 or next[1] + self.size[1] >= self.height):
            super().changeDegree(getBounceDegree(self.degree,0.5))
            hits[2 + (0 if next[1] < 0 else 1)] = 1

        return next, hits


def getDegreeMultypliers(degree):
        M = degree % 1

        AX = (M % 0.25) * 4
        A = 1 / math.sqrt(AX * AX * 2 - AX * 2 + 1)

        xm = 0
        ym = 0
        if(M <= 0.25):  xm = M * 4
        elif(M <= 0.75):xm = 1 - (M - 0.25) * 4
        else:           xm = (M - 0.75) * 4 - 1
        
        if(M <= 0.5):   ym = M * 4 - 1
        else:           ym = 1 - (M - 0.5) * 4
        return xm * A,ym * A    

#work only for y=0 or x=0 lines
def getBounceDegree(mover_degree, solid_degree):
    return (1 - (solid_degree) - mover_degree) % 1

def getDistanceBetweenPoints(p1, p2):
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    return math.sqrt(d1*d1 + d2*d2)