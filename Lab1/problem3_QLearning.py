import numpy as np
import matplotlib.pyplot as plt
from Lab1.problem3_env import Env

class QLearning():

    def __init__(self):

        self.num_moves = 10000000
        self.environment = Env()  # Environment class with the model

        self.QValues = np.zeros(( self.environment.NUM_STATES,  self.environment.NUM_ACTIONS))
        self.num_updates = np.zeros(( self.environment.NUM_STATES,  self.environment.NUM_ACTIONS))  #keep the number of updates of each value (Q(s,a))

        #PARAMETERS
        self.epsilon = 0.1  #egreedy policy
        self.alpha = 0.5    #learning rate

        self.sum_rewards = 0
        self.sum_rewards_list = []

        self.step_size = 0

        # keep the value function for the initial state
        self.init_state_indx = self.environment.states_mapping[((0,0), (3,3))]

        self.ValueF_init = np.zeros(self.num_moves)


    def egreedy_policy(self, state_indx):
        '''
        Choose an action based on a epsilon greedy policy.
        A random action is selected with epsilon probability, else select the best action.
        '''

        # first find possible actions(robber's movements)
        robber_moves_permitted = self.environment.check_wall_constraint('robber')
        if np.random.random() < self.epsilon:
            return np.random.choice(robber_moves_permitted )
        else:
            QValues_permitted = self.QValues[state_indx, robber_moves_permitted]    #robber_moves_permitted
            indx = np.argmax(QValues_permitted)  #index in QValues permitted
            return robber_moves_permitted[indx]

    def plot_ValueF(self, arg):
        '''
        plot yy throughout the game
        '''
        moves = range(self.num_moves)
        if arg == 'valueF':
            plt.plot(moves, self.ValueF_init)
        elif arg == 'sumRewards':
            plt.plot(moves, self.sum_rewards_list)

        plt.savefig('Figures/problem3_%s.png' %arg)
        plt.show()

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
            self.sum_rewards_list.append(self.sum_rewards)
            # find state value in next_state
            new_value = self.environment.reward + self.environment.lamda * np.max(self.QValues[new_state_indx])
            # compare it with action value in cur_state and action
            difference = new_value - self.QValues[cur_state_indx, action]
            # define step size
            self.num_updates[cur_state_indx, action] += 1
            self.step_size = float(1/pow(self.num_updates[cur_state_indx,action], float(2/3)))
            # update QValue in cur_state
            self.QValues[cur_state_indx,action] += self.step_size * difference
            print('state: %d' %cur_state_indx, 'action: %d'%action, 'QValue:%f'% self.QValues[cur_state_indx,action])
            #update cur_state
            cur_state_indx = new_state_indx

            #update valueF for the initial state
            self.ValueF_init[i] = np.amax(self.QValues[self.init_state_indx])


        self.myplot('valueF')
        self.myplot('sumRewards')



def test():

    qLearning_obj = QLearning()
    qLearning_obj.apply_qlearning()


test()


