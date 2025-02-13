"""
Script that contains details how the DQN agent learns, updates the target network, 
selects actions and save/loads the model
"""

import random
import numpy as np

import torch
import torch.nn.functional as F

from model import DQNNet
from replayMemory import ReplayMemory


class DQNAgent:
    """
    Class that defines the functions required for training the DQN agent
    """
    def __init__(self, device, inputSize = 18, actionSize = 4, 
                    discount=1.01,    
                    eps_max=1.0, 
                    eps_min=0.01, 
                    eps_decay=0.9999, 
                    memory_capacity=10000, 
                    lr=1e-2, 
                    train_mode=True): #Discount skal muligvis være over 1. 
        self.device = device

        # for epsilon-greedy exploration strategy
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        # for defining how far-sighted or myopic the agent should be
        self.discount = discount

        # size of the state vectors and number of possible actions
        self.inputSize = inputSize
        self.actionSize = actionSize

        # instances of the network for current policy and its target
        self.onlineNet = DQNNet(input_size=self.inputSize, output_size=self.actionSize, lr=lr).to(self.device)
        self.targetNet = DQNNet(input_size=self.inputSize, output_size=self.actionSize, lr=lr).to(self.device)
        self.targetNet.eval() # since no learning is performed on the target net
        
        if not train_mode:
            self.onlineNet.eval()

        # instance of the replay buffer
        self.memory = ReplayMemory(capacity=memory_capacity)


    def update_target_net(self):
        """
        Function to copy the weights of the current online net into the (frozen) target net
        Parameters
        ---
        none
        Returns
        ---
        none
        """

        self.targetNet.load_state_dict(self.onlineNet.state_dict())


    def update_epsilon(self):
        """
        Function for reducing the epsilon value (used for epsilon-greedy exploration with annealing)
        Parameters
        ---
        none
        Returns
        ---
        none
        """
        
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


    def select_action(self, state):
        """
        Uses epsilon-greedy exploration such that, if the randomly generated number is less than epsilon then the agent performs random action, else the agent executes the action suggested by the policy Q-network
        """
        """
        Function to return the appropriate action for the given state.
        During training, returns a randomly sampled action or a greedy action (predicted by the policy network), based on the epsilon value.
        During testing, returns action predicted by the policy network
        Parameters
        ---
        state: vector or tensor
            The current state of the environment as observed by the agent
        Returns
        ---
        none
        """

        if random.random() <= self.epsilon: # amount of exploration reduces with the epsilon value
            return random.randrange(self.actionSize)


        if not torch.is_tensor(state):
            state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            action = self.onlineNet.forward(state)

        tingmin = torch.argmin(action).item() # since actions are discrete, return index that has highest Q
        return tingmin # since actions are discrete, return index that has highest Q


    def learn2(self, state, next_state, action, reward, done):
        """
        Function to perform the updates on the neural network that runs the DQN algorithm.
        Parameters
        ---
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from
        Returns
        ---
        none
        """
        # select n samples picked uniformly at random from the experience replay memory, such that n=batchsize
        # if len(self.memory) < batchsize:
        #     return
        # states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)

       # states = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        
        state = torch.from_numpy(np.array(state)).float()
        action = torch.from_numpy(np.array(action)).float()
        
        # get q values of the actions that were taken, i.e calculate qpred; 
        # actions vector has to be explicitly reshaped to nx1-vector
        #actionView = actions.view(-1, 1)
        #actionView = actionView.type(torch.int64)  #DEtte virker måske IKKE grundet mangel på dtype!!!
        #q_pred = self.onlineNet.forward(states).gather(1, actionView) 
        
        actionView = action.view(-1, 1)  
        actionView = actionView.type(torch.int64)  #DEtte virker måske IKKE grundet mangel på dtype!!!   
        q_pred = self.onlineNet.forward(state).gather(1, actionView)
        print("Q pred", q_pred)
        bestAction = q_pred[action]
        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj

        q_target = self.targetNet.forward(next_state).min(dim=1).values # because max returns data structure with values and indices
        # Probably needs to be changed
        print(q_target)
        q_target[done] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state
        
        y_j = reward + (self.discount * q_target)
        y_j = y_j.view(-1, 1)
        
        # calculate the loss as the mean-squared error of yj and qpred
        self.onlineNet.optimizer.zero_grad()
        loss = F.mse_loss(y_j, bestAction).mean()
        loss.backward()
        self.onlineNet.optimizer.step()
        

    def learn(self, batchsize):
        """
        Function to perform the updates on the neural network that runs the DQN algorithm.
        Parameters
        ---
        batchsize: int
            Number of experiences to be randomly sampled from the memory for the agent to learn from
        Returns
        ---
        none
        """
        
        # select n samples picked uniformly at random from the experience replay memory, such that n=batchsize
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)

        # get q values of the actions that were taken, i.e calculate qpred; 
        # actions vector has to be explicitly reshaped to nx1-vector
        actionView = actions.view(-1, 1)
        actionView = actionView.type(torch.int64)  #DEtte virker måske IKKE grundet mangel på dtype!!!     
        q_pred = self.onlineNet.forward(states).gather(1, actionView) 
        
        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        #Overvej omkring min eller max
        q_target = self.targetNet.forward(next_states).min(dim=1).values # because max returns data structure with values and indices
        # Probably needs to be changed
        q_target[dones] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state
        
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)
        
        # calculate the loss as the mean-squared error of yj and qpred
        self.onlineNet.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.onlineNet.optimizer.step()
        

    def save_model(self, fileName = 'model.txt', filePath = "./Algorithms/DQNGitTing/saveModel/"):
        """
        Function to save the policy network
        Parameters
        ---
        filename: str
            Location of the file where the model is to be saved        
        Returns
        ---
        none
        """

        self.onlineNet.save_model(fileName = filePath+fileName)

    def load_model(self, fileName = 'model.txt', filePath = './saveModel/'):
        """
        Function to load model parameters
        Parameters
        ---
        filename: str
            Location of the file from where the model is to be loaded
        Returns
        ---
        none
        """

        self.onlineNet.load_model(device=self.device, fileName=filePath+fileName)