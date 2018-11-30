import numpy as np
import random
from problem3_env import Env
import matplotlib.pyplot as plt

class Sarsa():

    def __init__(self):
        self.environment = Env()   #Environment class with the model

        self.num_moves = 10000000
        self.num_episodes = 100

        self.QValues = np.zeros(( self.environment.NUM_STATES,  self.environment.NUM_ACTIONS))

        #PARAMETERS
        #self.epsilon_max = 0.1  #egreedy policy
        self.epsilon_min = 1e-3
        self.epsilon = 0.1
        '''
        Epsilon setting
        0: eps = eps_min + eps_max/t
        1: smooth step wise reduction
        2: keep constant high epsilon
        '''
        self.epsilon_setting = {0,1,2}
        self.epsilon_list = {0.1}   # can try different values of epsilon.
         
        self.epsilon_t = np.zeros(self.num_moves)     # epsilon with time

        self.alpha = 0.05    #learning rate

        self.sum_rewards = 0
        self.EPISODIC = 0
        self.ValueF_init = np.zeros(self.num_moves)
        # keep the value function for the initial state
        self.init_state_indx = self.environment.states_mapping[((0,0), (3,3))]


    def egreedy_policy(self, state_indx):
        '''
        Choose an action based on a epsilon greedy policy.
        A random action is selected with epsilon probability, else select the best action.
        '''

        if np.random.random() < self.epsilon:
            #first find possible actions(robber's movements)
            robber_moves_permitted = self.environment.check_wall_constraint('robber')
            return random.sample(robber_moves_permitted, 1)
        else:
            robber_moves_permitted = self.environment.check_wall_constraint('robber')
            best_action_ind = np.argmax(self.QValues[state_indx, robber_moves_permitted])
            return robber_moves_permitted[best_action_ind]
    ###NOTE: all states mentions refer to indexes, not the actual positions
    
    
    def plot_result(self, i,collected_rewards):
        plt.close()
        xx = np.linspace(0,i, i+1)
        yy = collected_rewards
        plt.plot(xx,yy)
        plt.title('Sum of rewards v/s eposides') 
        plt.ylabel('Sum of rewards')
        plt.xlabel('Episode number')
        plt.show()

    def plot_ValueF_epsilon(self, curr_epsilon, curr_epsilon_setting):
        '''
        plot yy throughout the game
        '''
        moves = range(self.num_moves)
        f1 = plt.figure(1)
        #plt.subplot(211)
        plt.plot(moves, self.ValueF_init)
        plt.title('Value fn at initial state v/s time')
        plt.xlabel('time')
        plt.ylabel('Value fn at initial state')

        #plt.subplot(212)
        f2 = plt.figure(2)
        plt.plot(moves, self.epsilon_t)
        plt.title('Epsilon v/s time')
        plt.xlabel('time')
        plt.ylabel('Epsilon')
        #plt.savefig('Figures/problem3_%s.png' %arg)
        
        f1.show()
        f2.show()
        f1.savefig('prob3_sarsa_valueF_' + str(curr_epsilon_setting)+'_'+str(curr_epsilon)+'.png')
        f2.savefig('prob3_sarsa_epsilon_'+ str(curr_epsilon_setting)+'_'+ str(curr_epsilon)+'.png')

    
    def sarsa_algo(self):
        '''
        Implement SARSA Algorithm. STEPS:
        
        Initially Q = np.zeros(NUM_STATES x NUM_ACTIONS). 
        1. Reset environment and compute action_probabilities for initial state
            state = env.reset()
            action_probs = policy(state)
        2. choose action based on above prob distribution
        3. for i in NITER (10 million), take a step and derive next state amd ots reward
        4. do TD learning steps
        5. current_action <- new_action, current_state <- new_state 
        '''
        for epsilon_setting in self.epsilon_setting:
            print('----------------------------------------------')
            print('Epsilon Setting: ' +str(epsilon_setting))
            for epsilon in self.epsilon_list:
                print('Choose Epsilon: ' + str(epsilon))
                self.epsilon = epsilon  # need to set this before we take 1st action 
                reduction = 0
                # initialize position of robber and police
                cur_state_indx = self.environment.reset_game()
                # choose an action
                action = self.egreedy_policy(cur_state_indx)
                
                #collected_rewards = []
                self.sum_rewards = 0    # CHECK!!!

                for i in range(self.num_moves):
                    # update epsilon 
                    if epsilon_setting == 0:    # Epsilon reduces quickly to MIN
                        if i == 0:
                            temp = 0
                        self.epsilon = temp * self.epsilon_min + epsilon / (i+1)
                    elif epsilon_setting == 1:  # Epsilon smoothly reduces to MIN
                        if i%(self.num_moves/100) == 0:   #Every 100000 steps
                            reduction += 1 
                            self.epsilon = epsilon/reduction
                    elif epsilon_setting == 2:  # Epsilon remains constant at MAX value
                        self.epsilon = epsilon

                    if i%(self.num_moves/100) == 0:   #Every 100000 steps
                        print('Iteration: ' +str(i) + ', current epsilon: '+ str(self.epsilon))
                    
                    # store epsilon with time (for plotting later) 
                    self.epsilon_t[i] = self.epsilon

                    # perform action --> move to new state and get reward
                    new_state_indx = self.environment.next_step(action)
                    
                    # pick the next action
                    new_action = self.egreedy_policy(new_state_indx)
                    # Update statisticss
                    self.sum_rewards += self.environment.reward
                    
                    # TD Update
                    td_target = self.environment.reward + self.environment.lamda * self.QValues[new_state_indx][new_action]
                    td_delta = td_target - self.QValues[cur_state_indx][action]
                    self.QValues[cur_state_indx][action] += self.alpha * td_delta

                    action = new_action
                    cur_state_indx = new_state_indx

                    #update valueF for the initial state
                    self.ValueF_init[i] = np.amax(self.QValues[self.init_state_indx])
                self.plot_ValueF_epsilon(epsilon, epsilon_setting)


def test():

    sarsa_obj = Sarsa()
    sarsa_obj.sarsa_algo()

test()








