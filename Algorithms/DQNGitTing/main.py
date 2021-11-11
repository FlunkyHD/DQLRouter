"""
Script containing the training and testing loop for DQNAgent
"""

import os
import csv
import argparse
import numpy as np
import pickle
from environment import *
from Graph import Node, Road



from agent import DQNAgent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# While loop should probably be changed to a counting loop where done is not the condition. 
def fill_memory(env, dqn_agent, num_memory_fill_eps):
    """
    Function that performs a certain number of episodes of random interactions
    with the environment to populate the replay buffer

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_memory_fill_eps: int
        Number of episodes of interaction to be performed

    Returns
    ---
    none
    """

    for _ in range(num_memory_fill_eps):
        done = False
        #state = env.reset()
        state = env.randomReset(1, False)
        while not done:
            action = env.sampleAction()
            next_state, reward, done = env.step(state,action)
            dqn_agent.memory.store(state=state, 
                                action=action, 
                                next_state=next_state, 
                                reward=reward, 
                                done=done)
            state = next_state


def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, results_basepath="NO"):
    """
    Function to train the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_train_eps: int
        Number of episodes of training to be performed
    num_memory_fill_eps: int
        Number of episodes of random interaction to be performed
    update_frequency: int
        Number of steps after which the target models must be updated
    batchsize: int
        Number of transitions to be sampled from the replay buffer to perform a learning step
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment
    
    Returns
    ---
    none
    """

    fill_memory(env, dqn_agent, num_memory_fill_eps)
    print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    
    reward_history = []
    epsilon_history = []

    step_cnt = 0
    best_score = np.inf

    for ep_cnt in range(num_train_eps):
        epsilon_history.append(dqn_agent.epsilon)

        done = False
        state = env.randomReset(1, False)
        #state = env.reset()

        ep_score = 0

        while not done:
            #print(state)
            action = dqn_agent.select_action(state)
            #print(action)
            next_state, reward, done = env.step(state, action)
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)

            dqn_agent.learn(batchsize)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-100:]) # moving average of last 100 episodes

        print('Ep: {}, Total Steps: {}, Ep: Score: {}, Avg score: {}; Epsilon: {}'.format(ep_cnt, step_cnt, ep_score, current_avg_score, epsilon_history[-1]))
        
        if current_avg_score <= best_score:
            # dqn_agent.save_model('{}/dqn_model'.format(results_basepath))
            best_score = current_avg_score
        if(ep_cnt % 100 == 0) :
            dqn_agent.save_model('NotWorkingSimpleNetwork.txt')
            

def testRandomTestsReturnAverage(env, dqnAgent, maxGoal, alwaysCapGoals = False, numberOfTests = 100) :
    dqnAgent.epsilon = 0
    reward_history = []
    stepHistory = []

    for i in range(numberOfTests):

        done = False
        state = env.randomReset(maxGoal, alwaysCapGoals)
        step_cnt = 0

        ep_score = 0

        while not done:
            action = dqnAgent.select_action(state)
            next_state, reward, done = env.step(state, action)

            state = next_state
            ep_score += reward
            step_cnt += 1
            if(step_cnt > len(env.graph) *3 + 1) :
                print("Svær opgave")
                print(state)
                break 

        reward_history.append(ep_score)
        stepHistory.append(step_cnt)

    averageReward = sum(reward_history) / numberOfTests
    print(averageReward)

def test(env, dqn_agent, num_test_eps, seed, results_basepath="no"):
    """
    Function to test the agent

    Parameters
    ---
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_test_eps: int
        Number of episodes of testing to be performed
    seed: int
        Value of the seed used for testing
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment

    Returns
    ---
    none
    """

    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.randomReset(1, False)
        #state = env.reset()
        while not done:

            action = dqn_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

    with open('{}/test_reward_history_{}.pkl'.format(results_basepath, seed), 'wb') as f:
        pickle.dump(reward_history, f)
        

if __name__ ==  '__main__':

        np.random.seed()
        torch.manual_seed(1)
        
        graph = []
        node0 = Node(False, 0)
        node1 = Node(False, 1)
        node2 = Node(False, 2)
        node3 = Node(False, 3)
        node4 = Node(False, 4)
        
        node0.addConnection(Road(1000,node0))
        node0.addConnection(Road(2,node1))
        node1.addConnection(Road(2,node0))
        node1.addConnection(Road(2,node2))
        node2.addConnection(Road(2,node1))
        node2.addConnection(Road(2,node3))
        node3.addConnection(Road(2,node2))
        node3.addConnection(Road(2,node4))
        node4.addConnection(Road(2,node3))
        node4.addConnection(Road(1000,node4))

        graph.append(node0)
        graph.append(node1)
        graph.append(node2)
        graph.append(node3)
        graph.append(node4)

        actionsSpace = 2

        env = environment(graph, actionsSpace)
        dqn_agent = DQNAgent(device, 
                                inputSize=len(graph) * 2, 
                                actionSize=actionsSpace, 
                                discount= 0.99, 
                                train_mode=True)
        dqn_agent.load_model(textFileName="NotWorkingSimpleNetwork.txt")
        
        train(env=env, 
                dqn_agent=dqn_agent, 
                # results_basepath=args.results_folder, 
                num_train_eps=20000, 
                num_memory_fill_eps=200, 
                update_frequency=20,
                batchsize=5)

        testRandomTestsReturnAverage(env, dqn_agent, 1, alwaysCapGoals = False, numberOfTests = 100)
    
    
    
    