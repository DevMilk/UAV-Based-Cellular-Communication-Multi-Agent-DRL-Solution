#!/usr/bin/env python
# coding: utf-8
import math
import scipy.constants as constant
from scipy.stats import truncnorm
import pickle
import json
import os
import keras
from keras import Model
from keras.layers import AveragePooling2D, Flatten, Conv2D, Input, concatenate, Dense
from keras.losses import Huber
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from random import randint
import random as rand 
import tensorflow as tf
from time import sleep
from collections import deque 
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.axes import Axes

sns.set_style("darkgrid")

# Carrier Frequency 
FC = 2e9  

# Speed of Light
C = constant.c  

# Power Spectral Density
N0 = 1e-20  

# Bandwidth
B = 1e6  

# Noise
STD = B * N0  

# Environmental Variable 1
B1 = 0.36  

# Environmental Variable 2
B2 = 0.21  

# Path Loss Exponent
ALPHA = 2   

# Additional path loss for LineOfSight
U_LOS = 10**(3/10)  

# Additional path loss for NonLineOfSight
U_NLOS = 10**(23/10)  

K0 = (4 * math.pi * FC / C) ** 2

# Environmental Variable 3
ZETA = 0  

# Transmission power of an UAV
P = 0.08  

# Discount Rate
GAMMA = 0.95  

# Learning Rate
LR = 0.001  

# Initial Epsilon Greedy Value
EPSILON = 1  

# Epsilon Greedy Decay Rate
EPSILON_DECAY = 0.95  

# Minimum Epsilon Greedy
EPSILON_MIN = 0.1  

# Minimum altitude to fly
MIN_ALTITUDE = 200

# Initial Altitude
ALTITUDE = 400  

# Maximum altitude to fly
MAX_ALTITUDE = 800  

# Penalty value
PENALTY = -100   

# UAV speed rate according to humans
UAV_MOVEMENT_SCALE= 20  

STEP_COUNT_TRAINED_ON = 64
UNIT =20
UE_COUNT = 200

env_dim = (100,100)

alt = ALTITUDE
mid_height, mid_width = env_dim[1]//2, env_dim[0]//2


uav_paths = [ 
              [mid_width, mid_height,alt ],  \
              [mid_width, mid_height,alt-1], \
              [mid_width, mid_height,alt-2], 
            ]

# Helper Function for normal distribution
def get_truncated_normal(mean=0, low=0, upp=10):
    return truncnorm(
        (low - mean), (upp - mean), loc=mean, scale=1).rvs()

# Normal Distribution    
def gaussian(end,begin=0):
    normal_random = round(get_truncated_normal(mean=int((begin+end)/2),low=begin,upp=end))
    return max(0,min(normal_random,end))

# Function to plot step graph on execution
def plot_step_graph(sum_rates_per_step,ax,lim=100):
      #ax.clear()
      ax.set_xlabel("Step Number (Time)")
      ax.set_ylabel("Sum Rate(bits/s/Hz)")
      ax.set_ylim(6,19)
      ax.set_xlim(1,lim+1)
      t = len(sum_rates_per_step)
      if(t>1):
        ax.plot([t-1,t],sum_rates_per_step[t-2:],c="blue")
      #ax.plot(range(len(sum_rates_per_step)),sum_rates_per_step)
      ax.set_title("Sum rates per step")

# Execution function to execute trained models
def test_env(file_name,user_data_path,random_env=False,fig=None,step_count=64,plot_trajectories=True):
    
    agent,env = semiConvAgentForTest, UavSimuEnvForTest

    new_env = env(uav_paths,range(UE_COUNT),(100,100),uav_class=agent)
    new_env.load(file_name,just_for_test=True)
    
    sum_rates_per_step = []
    if(fig==None):
      fig = plt.figure(figsize=(10,10),dpi=100)   
      ax = Axes3D(fig)
      fig2,ax1 = plt.subplots(figsize=(5,5),dpi=100)
      ax1.set_yticks([i for i in range(5,20)])
      
    #TESTING
    total_sum_rate = 0
    new_env.test_mode = not random_env
    new_env.reset(user_data_path)
    for counter in range(step_count):

        if counter%UAV_MOVEMENT_SCALE==0:
          new_env.step_UEs()
        #Step UAVs
        new_env.step_UAVs(isTest=True)

        #Get Error
        sum_rate = new_env.calculate_sum_rate()
        sum_rates_per_step.append(sum_rate)
        total_sum_rate += sum_rate
        #Step UEs 
        new_env.render(ax=ax,plot_trajectories=plot_trajectories) 
        #Plot resource errors by step and interation
        plot_step_graph(sum_rates_per_step,ax1,lim=step_count)

        """if(counter>1):
            plot_trajectories(new_env,ax3,counter)"""
        yield 

    
    score = (total_sum_rate/counter,new_env.calculate_sum_rate())
    result_dict = {"scores":score
                    }
    yield result_dict


# Main Agent
class Agent(): 

    def __init__(self,state_size,action_count,batch_size=200,maxlen=10000):

        self.batch_size = batch_size
        self.input_size   = state_size 
        self.action_count = action_count 
        self.lr = LR
        self.gamma = GAMMA #Discount factor = continuous tasklarda return'un sonsuza gitmemesi için
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.model = self.build_model() 
        self.memory = deque(maxlen=maxlen)

    def build_model(self):
        """
            Create Neural network as Deep Q Network
        """    
        model = Sequential()
        model.add(Dense(128, input_dim= self.input_size, activation = "relu"))   
        model.add(Dense(64,activation="relu"))   
        model.add(Dense(64,activation="relu"))   
        model.add(Dense(64,activation="relu"))  
        model.add(Dense(64,activation="relu"))  
        model.add(Dense(self.action_count,activation="linear"))
        model.compile(Adam(learning_rate = self.lr ,clipnorm=1.0),loss=Huber())
        return model  
 

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
         

    def act(self,state,cannot_random=False):
        if (rand.uniform(0,1) <= self.epsilon and not cannot_random):
            return rand.randint(0,self.action_count-1) 
        else:
            act_values = self.model.predict(state) 
            return np.argmax(act_values[0])
        
        
    def is_memory_enough(self):
        return not (len(self.memory)< self.batch_size)
    
    
    def replay(self):
        
        if(not self.is_memory_enough()):
            return
        
        minibatch = rand.sample(self.memory,self.batch_size)
        
        for state,action, reward,next_state, done in minibatch:  
            
          if done:
            target = reward
          else:  
            target = reward + self.gamma*np.amax(self.model.predict(next_state))  
           
          train_target = self.model.predict(state)
          train_target[0][action] = target 
          
          self.model.fit(state,train_target,verbose=0,workers=8,use_multiprocessing=True)
        

    def adaptiveEGreedy(self):
        if(self.epsilon > self.epsilon_min and self.is_memory_enough()):
            self.epsilon *= self.epsilon_decay   


# General class for moving objects
class MovingObject():
    def __init__(self,initial_location):
        self.current_location = initial_location.copy()
        self.starting_location = initial_location.copy()


    def reset(self):
        self.current_location = self.starting_location.copy()


# Uav class
class UAV(Agent,MovingObject): 
    
    def __init__(self,state_size,action_count,batch_size,initial_location,maxlen=10000):
        Agent.__init__(self,state_size,action_count,batch_size=batch_size,maxlen=maxlen)
        MovingObject.__init__(self,initial_location)
        self.transmission_power = P


# User Class
class UE(MovingObject):

    def __init__(self,initial_location,env_dim,movement_vectors,define_path_first = True,step_count = 10000,movement_function=gaussian):
        super().__init__(initial_location)

        self.movement_vectors =movement_vectors[:5].copy()  # altitude vektörleri hariç
        rand.shuffle(self.movement_vectors)

        self.movement_function = movement_function
        self.maxY , self.maxX = env_dim
        self.path = []
        self.path_determined = define_path_first

        if define_path_first:
            self.initial = 0
            for _ in range(step_count):
                self.path.append(movement_function(len(self.movement_vectors)-1))
    
    def move(self):
        action_index = 0
        if self.path_determined:
            action_index = self.path[self.initial]
            self.initial+=1
            if self.initial==len(self.path):
                self.initial = 0
        else:
            action_index = self.movement_function(len(self.movement_vectors)-1)

        action = self.movement_vectors[int(action_index)]
        self.current_location= [self.current_location[i] + action[i] for i in range(3)]


# Main Environment
class UavSimuEnv:
  actions = [(1,0,0),(-1,0,0),(0,0,0),(0,1,0),(0,-1,0)] #Actionlar
  coord_count = 2
    
  def distance_func(self,v1,v2): 
    return (sum(((p-q)*UNIT)**2 for p, q in zip(v1[:2], v2[:2])) + (v1[2]-v2[2])**2)** .5

  def get_all_uavs(self):
        return [uav for uav in self.map["uav_set"]]
        
  def save(self,env_name): 
    for index,uav in enumerate(self.get_all_uavs()):
        model = uav.model
        path = "modelss/{}/".format(env_name)
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = "{}uav{}".format(path,index)
        model.save(model_name+".h5")
        pickle.dump(uav.memory, open(model_name+"memory.pkl", 'wb'))
            
            
  def load(self,env_name,just_for_test=True):
    """
      Load trained uav models to environment
    """
    for index in range(len(self.map["uav_set"])):
        path = "modelss/{}/".format(env_name)
        model_name = "{}uav{}".format(path,index)
        self.getUAV(index).model = keras.models.load_model(model_name+".h5",compile=just_for_test)
        self.getUAV(index).memory = pickle.load(open(model_name+"memory.pkl", "rb"))

            
  def get_input_size(self,uav_count):
    return uav_count*self.coord_count + math.ceil(self.step_count/UAV_MOVEMENT_SCALE)

  def is_collect_step(self):
    return self.getUAV(0).is_memory_enough() 
               
  def get_current_epsilon(self):
    return self.getUAV(0).epsilon
    
  def __init__(self,uav_paths,ue_set,env_dim = (100,100),batch_size=200,max_memory_per_agent=10000,uav_class=UAV):
    self.step_count = STEP_COUNT_TRAINED_ON
    self.uav_count = len(uav_paths) 
    self.env_dim = env_dim 
    self.input_size = self.get_input_size(len(uav_paths))
    self.map = {"uav_set": self.init_uav(uav_paths,batch_size,max_memory_per_agent,uav_class), "ue_set": ue_set}

  def init_uav(self,uav_paths,batch_size,max_memory_per_agent,uav_class):
    """
      Agent'ları initialize eder
    """
    return [uav_class(self.input_size, len(self.actions), batch_size, begin,maxlen=max_memory_per_agent) 
            for begin in uav_paths]


  def get_distance_list(self, dest_list, location,isObject=False):
    """
      Verilen konum dizisi ile verilen konum arasındaki mesafelerin dizisini döndürür
    """
    if (isObject):
      return [self.distance_func(location,loc.current_location) for loc in dest_list]
    return [self.distance_func(location,loc) for loc in dest_list] 
    
  def reset(self):
    """
      Environmentin state'sini en başa alır
    """
    def reset_set(key):
      for index, val in enumerate(self.map[key]):
        self.map[key][index].reset()
    
    reset_set("ue_set")
    reset_set("uav_set")
    self.initial_time = 0 

  def get_state_input(self,isNext=False):
    """
      verilen Agent'in stateye dair inputunu döndürür

    """
    all_uav_coordinates = []

    for uav in self.map["uav_set"]:
      all_uav_coordinates.extend(uav.current_location[:self.coord_count]) 
    
    time_range = [0 for _ in range(math.ceil(self.step_count/UAV_MOVEMENT_SCALE))]
    time_range[self.initial_time // UAV_MOVEMENT_SCALE] = 1
    return self.reshape(all_uav_coordinates +time_range)

    
  def get_distance_matrix(self,map_obj=None):
    """
      Tüm UAV-BS ve UE-AV arasındaki mesafe matrislerini ve ilişki sayısı vektörünü döndürür
    """
    if(map_obj==None):
        map_obj = self.map
        
    def minIndex(lst):
      
        return lst.index(min(lst))    

    def get_distances(lst1, lst2, isLst2Object = True):
      distance_matrix = []
      assoc_matrix = []
      for _ in range(len(lst2)):
        assoc_matrix.append([])

      for index,member in enumerate(lst1):
        distances = self.get_distance_list(lst2,member.current_location,isLst2Object)
        distance_matrix.append(distances)
        index_of_min = minIndex(distances)
        assoc_matrix[index_of_min].append(index)

      return distance_matrix, assoc_matrix

    ue_uav_matrix, assoc_matrix_uav = get_distances(map_obj["ue_set"],map_obj["uav_set"]) 

    return {"ue_uav_matrix": ue_uav_matrix, "assoc_matrix_uav" : assoc_matrix_uav}


  def calculate_sum_rate(self,map_obj=None):
    
    if(map_obj==None):
        map_obj = self.map
        
    uav_count = len(map_obj["uav_set"])
    ue_count = len(map_obj["ue_set"])
    distance_state_dict = self.get_distance_matrix(map_obj)
    sumrate = 0
    transmit_powers = []
    channel_gain_matrix = []
    
    def calculate_transmit_power(uav_index):
      """
        calculate transmit power of uav
      """
      uav = map_obj["uav_set"][uav_index]
      uav_assoc_count = len(distance_state_dict["assoc_matrix_uav"][uav_index])
    
      #Eğer uav hiçbir servis yapmıyorsa transmit gücünü kullanmayacağı için 0 olacak
      if(uav_assoc_count==0):
            return 0 
      p = uav.transmission_power / uav_assoc_count

      return p

    
    def calculate_channel_gain(uav_index,user_index):
      """
        calculate channel gain between uav and user
      """
      uav = map_obj["uav_set"][uav_index]
      d = distance_state_dict["ue_uav_matrix"][user_index][uav_index]
      theta = math.asin(uav.current_location[-1] / d )
      Plos = B1 * (180 * theta / math.pi - ZETA ) ** B2
      
      Pnlos = 1 - Plos
      g = (K0 ** (-1)) * (d ** (- ALPHA)) * ((Plos*U_LOS + Pnlos*U_NLOS) **(-1))
      return g #gainin negarif olmaması için Plos negatif olmalı


    #First calculate all channel gains and transmit powers of all combination of uav and users
    for uav_index in range(uav_count):

      p = calculate_transmit_power(uav_index)
      transmit_powers.append(p)
      channel_gain_list = [calculate_channel_gain(uav_index,user_index) for user_index in range(ue_count)]
      channel_gain_matrix.append(channel_gain_list)
    

    def calculate_interference(uav_index,user_index):
      """
        Calculate interference between uav and user caused by other uavs
      """
      """return sum([transmit_powers[other_uav_index] * calculate_channel_gain(other_uav_index,user_index)                   for other_uav_index in range(uav_count) if other_uav_index!=uav_index])"""
      I = 0

      for other_uav_index in range(uav_count):
        if (other_uav_index==uav_index):
          continue
        p = transmit_powers[other_uav_index]
        g = channel_gain_matrix[other_uav_index][user_index]
        I +=  p*g 
        
      return I
    

    for uav_index in range(uav_count):

        p = transmit_powers[uav_index]
        users_of_uav = distance_state_dict["assoc_matrix_uav"][uav_index]
        for user_index in users_of_uav:
          
          I = calculate_interference(uav_index,user_index)
          g = channel_gain_matrix[uav_index][user_index]
          SINR = p*g/(I + STD)
          sumrateOfUser = B * math.log2(1+SINR)
          sumrate += sumrateOfUser 
            
    return sumrate*1e-7

  def step_UEs(self):
    """
      Simulasyondaki tüm UE'leri hareket ettirir
    """
    ue_set_length = len(self.map["ue_set"])
    for ue in self.map["ue_set"]:
      ue.move()

      #Eğer ue'lerin konumları input olarak varsa, bunu yapamayız
      if (not self.isInside(ue.current_location,True)):
        self.map["ue_set"].remove(ue)


  def isCollides(self,uav_index,new_location): 
    for index,uav in enumerate(self.map["uav_set"]): 
        if(index!=uav_index and np.array_equal(uav.current_location, new_location)):
            return True
    return False


  def step_env(self,action_indexes):
        
        old_sum_rate = self.calculate_sum_rate()
        penalty = 0
        done = False
        for uav_index in range(len(self.map["uav_set"])):
            current = self.getUAV(uav_index).current_location
            action_index = action_indexes[uav_index]
            new_location = [current[i] + self.actions[action_index][i]
                                                         for i in range(3)]
            if(not self.isInside(new_location)):
                penalty += PENALTY
                done = True
            else:
                self.getUAV(uav_index).current_location = new_location
        
        for uav_index in range(len(self.map["uav_set"])):
            if(self.isCollides(uav_index,self.getUAV(uav_index).current_location)):
                done = True
                penalty +=PENALTY
                
        new_sum_rate = self.calculate_sum_rate()
        
        reward = (new_sum_rate-old_sum_rate) + penalty #penalty negatif
        done = done or self.initial_time == self.step_count
        return self.get_state_input(), reward, done
    
     
  def getUAV(self,uav_index):
    return self.map["uav_set"][uav_index]


  def step_UAVs(self,isTest=False,isCollectStep=False):
    """
      Tüm uav'leri hareket ettirir, 
      hepsinin bu stepteki rewardlarının toplamını
      simulasyonun reward listesine ekler
    """
    uav_set_length = len(self.map["uav_set"])
    
    state = self.get_state_input()
    
    #Paralel
    action_indexes = [self.getUAV(uav_index).act(state,cannot_random = isTest)                for uav_index in range(uav_set_length)]
        
    next_state, reward, Done = self.step_env(action_indexes)
    
    
    #Paralel
    if(not isTest):
        for uav_index in range(uav_set_length):
            self.getUAV(uav_index).store(state,action_indexes[uav_index],reward,next_state,Done) 
            
            if(not isCollectStep):
                self.getUAV(uav_index).replay()
                if(self.initial_time==self.step_count-1):
                    self.getUAV(uav_index).adaptiveEGreedy() 
    self.initial_time +=1
        
            
            
  def reshape(self, data):
    return np.reshape(data, [1,self.input_size]) 


  def isInside(self,location,isOnGround=False):
    """
      Verilen koordinatın simulasyonun dışına çıkmadığını döndürür
    """
    return (0 <= location[0] < self.env_dim[1] and 0<= location[1] < self.env_dim[0] ) and (isOnGround or  MIN_ALTITUDE <= location[2] <= MAX_ALTITUDE)

  def get3DMap(self):
      def getCoords(map):
        x,y,z = [],[],[]
        for obj in map:
          x.append(obj.current_location[0])
          y.append(obj.current_location[1])
          z.append(obj.current_location[2])
        return x,y,z
      return getCoords(self.map["uav_set"]),getCoords(self.map["ue_set"])

    
  def render(self):

    """
      Simulasyonu görsel olarak yazdırır
    """
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    uav_coords,ue_coords = self.get3DMap()
    clear_output(wait=True)
    ax.set_xlim3d(0,self.env_dim[0])
    ax.set_ylim3d(0,self.env_dim[1])
    ax.set_zlim3d(0,MAX_ALTITUDE)
    ax.scatter(*uav_coords,c="green")
    ax.scatter(*ue_coords,c="red")
    distance_map = self.get_distance_matrix()["assoc_matrix_uav"]
    for uav_index in range(self.uav_count):
        coord_of_UAV = self.getUAV(uav_index).current_location
        for ue_index in distance_map[uav_index]: 
            coord_of_UE = self.map["ue_set"][ue_index].current_location
            ax.plot([coord_of_UE[0],coord_of_UAV[0]],[coord_of_UE[1],coord_of_UAV[1]],[coord_of_UE[2],coord_of_UAV[2]],c="green",alpha=.1)
    plt.show()


# Env with independent execution instead of jpint execution 
class IndependentAISimuEnv(UavSimuEnv):
    
    def step(self,uav_index,action_index):
        
        #Get current and new locations of uav
        current_loc = self.getUAV(uav_index).current_location
        new_location = [current_loc[i] + self.actions[action_index][i]
                                                         for i in range(3)]
        
        #Calculate Reward
        reward = 0
        done = False
        if(not self.isInside(new_location) or self.isCollides(uav_index,new_location)):
            reward = PENALTY
            done = True
        else:
            old_sum = self.calculate_sum_rate()
            self.getUAV(uav_index).current_location = new_location
            new_sum = self.calculate_sum_rate()
            reward = (new_sum - old_sum)
        
        #If uav is the last one then increase step
        if(uav_index==self.uav_count-1):
            self.initial_time+=1
            
        #Get next state (Next uav's next input = next observation of next agent)
        next_obs = self.get_state_input(uav_index)
        done = done or self.initial_time == self.step_count
        return reward,next_obs,done 
    
    def train_agents(self):
        for uav_index in range(self.uav_count):
            self.getUAV(uav_index).replay() 
            if(self.initial_time==self.step_count-1):
                self.getUAV(uav_index).adaptiveEGreedy() 
                    
    def step_UAVs(self,save_reward=True,isTest=False,isCollectStep=False):  
        
        #For each uav, get (observation,action,reward,next_state)
        for uav_index in range(self.uav_count):
            
            observation = self.get_state_input(uav_index) 
            
            #Get action index of uav
            action = self.getUAV(uav_index).act(observation, cannot_random=isTest)
            
            #step env
            reward,next_obs,done = self.step(uav_index,action)
            
            #If training, agent stores
            if(not isTest):
                self.getUAV(uav_index).store(observation,action,reward,next_obs,done) 
                
        
        
        #After a step finishes, agent replays
        if(not isTest and not isCollectStep):
            self.train_agents()


# Agent class same as agent classes of trained models
class semiConvAgent(UAV):
    
    def build_model(self):
        inputA = Input(shape=(self.input_size[0],))
        inputB = Input(shape=list(self.input_size[1])+[1])
        
        uav_coord_input = Dense(64, activation="relu")(inputA)
        uav_coord_input = Model(inputs=inputA, outputs=uav_coord_input)
        
        ue_coord_input = AveragePooling2D(pool_size=(4, 4),padding="same")(inputB)
        ue_coord_input = Flatten()(ue_coord_input)
        ue_coord_input = Model(inputs=inputB, outputs=ue_coord_input)

        combined = concatenate([uav_coord_input.output, ue_coord_input.output])
        
        z = Dense(512, activation="relu")(combined)
        z = Dense(512, activation="relu")(z)
        z = Dense(256, activation="relu")(z)
        z = Dense(self.action_count,activation="linear")(z)
        
        model = Model(inputs=[uav_coord_input.input, ue_coord_input.input], outputs=z)
        model.compile(Adam(learning_rate = self.lr ,clipnorm=1.0),loss=Huber())
        return model  


# Environment class that compatible with trained models
class UserScalableEnvWithTimeInfo(IndependentAISimuEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def reshape(self, data):
        return np.reshape(data, (1,) + data.shape) 
    
    def get_input_size(self,uav_count):
        return (uav_count*self.coord_count+1,self.env_dim)
    
    #Point process Distribution
    def create_random_user_locations(self):
        total_intensity = self.env_dim[0]*self.env_dim[1]*(np.exp(1)-1) / 5
        max_intensity = np.exp(1)/5

        number_points = np.random.poisson(self.env_dim[0] * self.env_dim[1] * max_intensity)
        ue_coords = []
        ue_count = len(self.map["ue_set"])
        
        while(len(ue_coords)<ue_count):
            x = randint(0, self.env_dim[1]-4)
            y = randint(0, self.env_dim[0]-4)
            intensity = np.exp(y / self.env_dim[1]) / 5
            if intensity >= np.random.uniform(0, max_intensity):
                ue_coords.append([x,y,0])
        return ue_coords
    
    
    def reset(self,user_data):
        """
          Environmentin state'sini en başa alır
        """
        def reset_set(key):
          for index, val in enumerate(self.map[key]):
            self.map[key][index].reset()
            
        ue_set = []
        
        if(self.test_mode==True):
            a = UE
            with open(user_data, 'rb') as data: 
                ue_set = pickle.load(data)
        else:
            random_locs = self.create_random_user_locations()
            for coord in random_locs:
                ue_set.append(UE(coord,env_dim,UavSimuEnv.actions,define_path_first=True,step_count=self.step_count//UAV_MOVEMENT_SCALE))
        
        self.map["ue_set"] = ue_set
        reset_set("uav_set")
        self.initial_time = 0 

    def get_state_input(self,uav_index):
        all_uav_coordinates = []

        for uav in self.map["uav_set"]:
          all_uav_coordinates.extend(uav.current_location[:self.coord_count]) 

        user_map = np.zeros((env_dim[0],env_dim[1]))
        for user in self.map["ue_set"]:
            row = user.current_location[0]
            col = user.current_location[1]
            user_map[row][col] = 1
            
        return [self.reshape(np.array(all_uav_coordinates + [max(0,self.step_count-self.initial_time)])),self.reshape(user_map)]
 

# Agent class for only execution of trained models
class semiConvAgentForTest(semiConvAgent):
    def act(self,*args,**kwargs):
        try:
            self.location_history
        except:
            self.location_history = []
            
        action_index = semiConvAgent.act(self,*args,**kwargs)
        self.location_history.append(self.current_location)
        return action_index


# Environment class for only execution of trained models
class UavSimuEnvForTest(UserScalableEnvWithTimeInfo):
  actions = [(1,0,0),(-1,0,0),(0,0,0),(0,1,0),(0,-1,0),(0,0,-UAV_MOVEMENT_SCALE),(0,0,UAV_MOVEMENT_SCALE)] #Actionlar

  def get_input_size(self,uav_count):
        return (uav_count*self.coord_count+1,self.env_dim)
  
  def load(self,path,just_for_test=True,load_memory=False):
    for index in range(len(self.map["uav_set"])):
        model_name = "{}/uav{}".format(path,index)
        self.getUAV(index).model = keras.models.load_model(model_name+".h5",compile=not just_for_test)
        if(load_memory):
            self.getUAV(index).memory = pickle.load(open(model_name+"memory.pkl", "rb"))

  
  def get_state_input(self,uav_index):
        all_uav_coordinates = []

        for uav in self.map["uav_set"]:
          all_uav_coordinates.extend(uav.current_location[:self.coord_count]) 

        user_map = np.zeros((env_dim[0],env_dim[1]))
        for user in self.map["ue_set"]:
            row = user.current_location[0]
            col = user.current_location[1]
            user_map[row][col] = 1
            
        return [self.reshape(np.array(all_uav_coordinates + [max(0,self.step_count-self.initial_time)])),self.reshape(user_map)]

  def step_UAVs(self,save_reward=True,isTest=False,isCollectStep=False):  
        
        #For each uav, get (observation,action,reward,next_state)
        for uav_index in range(self.uav_count):
            
            observation = self.get_state_input(uav_index) 
            
            #Get action index of uav
            action = self.getUAV(uav_index).act(observation, cannot_random=isTest)
            
            #step env
            self.step(uav_index,action)

  def step(self,uav_index,action_index):
        
        #Get current and new locations of uav
        current_loc = self.getUAV(uav_index).current_location
        new_location = [current_loc[i] + self.actions[action_index][i]
                                                         for i in range(3)]

        if(not (not self.isInside(new_location) or self.isCollides(uav_index,new_location))):
            self.getUAV(uav_index).current_location = new_location
        
        #If uav is the last one then increase step
        if(uav_index==self.uav_count-1):
            self.initial_time+=1
  def render(self,ax=None,plot_trajectories=True):

      """
        Simulasyonu görsel olarak yazdırır
      """
    
      ax.clear()
      uav_coords,ue_coords = self.get3DMap()
      ax.set_xlim3d(0,self.env_dim[0])
      ax.set_ylim3d(0,self.env_dim[1])
      ax.set_zlim3d(0,MAX_ALTITUDE)
      
      ax.set_xlabel("X (unit)")
      ax.set_ylabel("Y (unit)")
      ax.set_zlabel("Z (meters)")
      ax.scatter(*uav_coords,c="green")
      ax.scatter(*ue_coords,c="red")

      
      if(plot_trajectories):
        for uav_index,uav in enumerate(self.get_all_uavs()):
            trajectory_map_X = []
            trajectory_map_Y = []
            trajectory_map_Z = []
            for location in uav.location_history:
                trajectory_map_X.append(location[0])
                trajectory_map_Y.append(location[1])
                trajectory_map_Z.append(location[2])
            
            ax.scatter(trajectory_map_X,trajectory_map_Y,trajectory_map_Z,c="purple",alpha=.3)

      distance_map = self.get_distance_matrix()["assoc_matrix_uav"]
      for uav_index in range(self.uav_count):
          coord_of_UAV = self.getUAV(uav_index).current_location
          for ue_index in distance_map[uav_index]: 
              coord_of_UE = self.map["ue_set"][ue_index].current_location
              ax.plot([coord_of_UE[0],coord_of_UAV[0]],[coord_of_UE[1],coord_of_UAV[1]],[coord_of_UE[2],coord_of_UAV[2]],c="green",alpha=.1)
