import numpy as np
import random
from Lab1.problem3_env import Env

class QLearning():

    def __init__(self):

        self.num_moves = 10000000
        self.environment = Env()  # Environment class with the model

        self.QValues = np.zeros(( self.environment.NUM_STATES,  self.environment.NUM_ACTIONS))

        #PARAMETERS
        self.epsilon = 0.1  #egreedy policy
        self.alpha = 0.5    #learning rate

        self.sum_rewards = 0


    def egreedy_policy(self, state_indx):
        '''
        Choose an action based on a epsilon greedy policy.
        A random action is selected with epsilon probability, else select the best action.
        '''

        robber_moves_permitted = self.environment.check_wall_constraint('robber')
        if np.random.random() < self.epsilon:
            #first find possible actions(robber's movements)
            return random.sample(robber_moves_permitted, 1)
        else:
            return np.argmax(self.QValues[state_indx, robber_moves_permitted])

    ###NOTE: all states mentions refer to indexes, not the actual positions
    def apply_qlearning(self):
        '''
        Implement Q-Learning Algorithm
        '''

        # initialize position of robber and police
        cur_state_indx = self.environment.reset_game()

        for i in range(self.num_moves):

            # choose action
            action = self.egreedy_policy(cur_state_indx)
            # perform action --> move to new state and get reward
            new_state_indx = self.environment.next_step(action)
            self.sum_rewards += self.environment.reward
            # find state value in next_state
            new_value = self.environment.reward + self.environment.lamda * np.max(self.QValues[new_state_indx])
            # compare it with action value in cur_state and action
            difference = new_value - self.QValues[cur_state_indx, action]
            # update QValue in cur_state
            self.QValues[cur_state_indx,action] += difference

            #update cur_state
            cur_state_indx = new_state_indx


def test():

    qLearning_obj = QLearning()
    qLearning_obj.apply_qlearning()

test()


