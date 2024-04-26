import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")
from gym import Env
from gym.spaces import Discrete, Box
from Games.Game_ex import Game
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from Data.Files_meneger import getAvailableFolder

SAVE_FOLDER = "F:\\programing\\python\\machine_learning_hub\\_not code_\\RL_MODELS\\"

LEARNING_RATE = 0.01

class RL1():
    def __init__(self, game, agent, name="namless"):
        self.GAME = game
        self.AGENT = agent
        
        self.path = None
        self.name = name
    
    def train(self, steps=50000,visualize=True,verbose=1):
        self.AGENT.fit(self.GAME,nb_steps=steps,visualize=visualize,verbose=verbose)
        
    def save(self, newFile=False):
        if(self.path == None):
            self.path = getAvailableFolder(SAVE_FOLDER + self.name)
        if(newFile):
            path = getAvailableFolder(self.path + "\\model")
            self.AGENT.model.save(path)
            return
        self.AGENT.model.save(self.path + "\\model")
    


#get models
def getModel_S1(game = None,inputs=None,actions=None):
    if(inputs == None or actions == None):
        inputs = (1,) + game.observation_space.shape
        actions = game.action_space.n
    
    model = Sequential()
    model.add(Dense(24,activation='relu',input_shape=inputs))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    model.add(Flatten())
    model.build()
    print(model.summary())
    return model

def getModel_S2(game = None,inputs=None,actions=None):
    if(inputs == None or actions == None):
        inputs = (1,) + game.observation_space.shape
        actions = game.action_space.n
    
    model = Sequential()
    model.add(Dense(32,activation='relu',input_shape=inputs))
    model.add(Dense(32,activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    model.add(Flatten())
    model.build()
    print(model.summary())
    return model

#get agents
def getAgent_1(model, game=None, actions=None):
    if(actions == None):
        actions = game.action_space.n
    
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000,window_length=1)
    
    agent = DQNAgent(model=model,memory=memory,policy=policy,nb_actions=actions,target_model_update=LEARNING_RATE)
    agent.compile(Adam(lr=LEARNING_RATE),metrics=['mae'])
    return agent