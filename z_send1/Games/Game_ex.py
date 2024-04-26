import matplotlib.pyplot as plt
from IPython import display

def printSelfError(messege):
        print("\n\nSELF ERORR!!!: [" + messege + "]\n\n")
        
class Game():
    def __init__(self, actionSpace, observationSpace, loseScore=0, showGraph=True, showBotConnaction=True):
        self.action_space = actionSpace # the operations the get can do
        self.observation_space = observationSpace # the size on the input the bot get
        self.state = 0 # the state of the game
        self.gameRun = True
        self.lose_score = loseScore
        self.show_graph = showGraph
        if(showGraph):
            plt.ion()
            display.display(plt.gcf())
            plt.title("train stats")
            plt.xlabel("number of games")
            plt.ylabel("scores")
            plt.ylim(ymin=0)
            self.scores = []
            self.avrages = []
            self.totalScores = 0
        
    
    def step(self, action): #step from the bot
        self.adouptBotInput(action)
        reward = self.runFrame()
        if(self.gameRun and not self.canContinue()): self.gameOver()
        if(not self.gameRun):
            if(self.lose_score == -1):
                reward = self.getGameOverReward()
            elif(self.lose_score != 0):
                reward = self.lose_score
                
            if(self.show_graph): self.addToGraph()
        return self.getBotObservation(), reward, not self.gameRun, {}
        
     
    def addToGraph(self):
        
       score = self.getGameScore()
       self.scores.append(score)
       self.totalScores += score
       self.avrages.append(self.totalScores / len(self.scores))
       
       display.clear_output(wait=True)
       plt.plot(self.scores)
       plt.plot(self.avrages)
       plt.text(len(self.scores)-1, self.scores[-1], str(self.scores[-1]))
       #plt.text(len(avrage)-1, avrage[-1], str(avrage[-1]))
       plt.autoscale()
       plt.show(block=False)
       plt.pause(.1)
       
    
    
    #methods that all the implomented class must have
    
    def render(self, mode): #render the game graphics
        printSelfError("render() method is called from game_ex class")
    def adouptBotInput(self,action): #convert array of boolean to the game's input format
        printSelfError("adouptBot() input method is called from game_ex class")
    def runFrame(self): #run single frame of the game
        printSelfError("runFrame() method is called from game_ex class")
    def reset(self): # restart the game
        printSelfError("reset() method is called from game_ex class")
    def canContinue(self): # if the game didn't over from technical resones
        printSelfError("canContinue() method is called from game_ex class")
    def gameOver(self): # call that when the game over
        printSelfError("gameOver() method is called from game_ex class")
    def getBotObservation(self): # return the propaties of the game
        printSelfError("getBotObservation() method is called from game_ex class")
    def getGameScore(self): # return the score of the end game
       printSelfError("getGameScores() method is called from game_ex class")