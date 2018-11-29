'''
Define the environment: states, actions, rewards, functions for finding the next state
'''

import numpy as np
import itertools
import random
import copy


class Env():

    def __init__(self):

        '''Grid'''
        # Origin at Top left
        self.GRID_SIZE_COL = 4
        self.GRID_SIZE_ROW = 4


        # state: (robber position, police position)

        self.robber = (0,0)
        self.police = (3,3)

        self.bank = (1,1) #location of bank - doesnt change

        # Container for walls
        self.walls = np.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL, 4), dtype=np.int)

        '''States'''
        self.define_states()

        '''Actions:
        0:  Left
        1:  Up
        2:  Right
        3:  Down
        4:  Stay
        '''
        self.actions_robber = [0, 1, 2, 3, 4]
        self.NUM_ACTIONS = len(self.actions_robber)
        self.police_movements = [0, 1, 2, 3]
        self.num_movs_robber = len(self.actions_robber)
        self.num_movs_police = len(self.police_movements)

        '''Reward'''
        self.reward = 0

        '''Infinite Discounted Time Horizon'''
        self.lamda = 0.8

        self.define_wall()

    def define_states(self):

        self.NUM_STATES = (self.GRID_SIZE_COL * self.GRID_SIZE_ROW) ** 2
        all_positions = [positions_pair for positions_pair in
                         itertools.product(range(self.GRID_SIZE_ROW), range(self.GRID_SIZE_COL))]
        self.all_states = [states_pair for states_pair in itertools.product(all_positions, all_positions)]
        self.states_mapping = dict(zip(list(self.all_states), range(0, self.NUM_STATES)))



    def define_wall(self):
        '''
        Wall definition:
        Every cell has 4 flags (0:free, 1:bocked by wall)
        0:  Left
        1:  Up
        2:  Right
        3:  Down
        For eg, self.walls = (0,0) = [0,1,1,0], this means that
        or cell 0,0:    Up and Right sides are blocked by walls
        '''

        self.walls[0,0] = [1,1,0,0]                                     # TOP LEFT
        self.walls[self.GRID_SIZE_ROW-1,self.GRID_SIZE_COL-1] = [0,0,1,1]   # BOTTOM RIGHT
        self.walls[self.GRID_SIZE_ROW-1, 0] = [1,0,0,1]
        self.walls[0,self.GRID_SIZE_COL-1] = [0,1,1,0]

        # cells between TOP LEFT and TOP RIGHT
        self.walls[0, 1:self.GRID_SIZE_COL-1] = [0,1,0,0]
        # cells between BOTTOM LEFT and BOTTOM RIGHT
        self.walls[self.GRID_SIZE_ROW-1, 1:self.GRID_SIZE_COL-1,] = [0,0,0,1]
        # cells between TOP LEFT and BOTTOM LEFT
        self.walls[1:self.GRID_SIZE_ROW-1,0] = [1,0,0,0]
        # cells between TOP RIGHT and BOTTOM RIGHT
        self.walls[1:self.GRID_SIZE_ROW-1, self.GRID_SIZE_COL-1] = [0,0,1,0]


    def move_agent(self, agent, action):
        '''
        returns location (x_hat, y_hat) after moving the agent(robber or thief) according to the input action
        '''

        if agent == 'robber':
            x = copy.copy(self.robber[0])
            y = copy.copy(self.robber[1])
        elif agent == 'police':
            x = copy.copy(self.police[0])
            y = copy.copy(self.police[1])
        else:
            print('Robber or police!')
            return

        if action == 0: #LEFT
            next_location = (x,y-1)
        elif action == 1: #UP
            next_location = (x-1,y)
        elif action == 2: #RIGHT
            next_location = (x,y+1)
        elif action == 3:  #DOWN
            next_location = (x+1,y)
        else: #STAY
            next_location = (x,y)

        return next_location

    def check_wall_constraint(self, agent):
        '''
        returns a list of valid movements from a current position
        agent: robber or police
        '''

        forbidden_actions = np.empty(self.num_movs_police)
        agent_position = (-1,-1)

        if agent == 'robber':
            agent_position = self.robber
            forbidden_actions = np.zeros(self.num_movs_robber)
        elif agent == 'police':
            agent_position = self.police
            forbidden_actions = np.zeros(self.num_movs_police)

        if self.walls[agent_position[0], agent_position[1], 0] == 1:  # agent's LEFT side is blocked
            forbidden_actions[0] = 1
        if self.walls[agent_position[0], agent_position[1], 1] == 1:  # agent's TOP side is blocked
            forbidden_actions[1] = 1
        if self.walls[agent_position[0], agent_position[1], 2] == 1:  # agent's RIGHT side is blocked
            forbidden_actions[2] = 1
        if self.walls[agent_position[0], agent_position[1], 3] == 1:  # agent's BOTTOM side is blocked
            forbidden_actions[3] = 1

        actions_permitted = [value[0] for value in np.argwhere(forbidden_actions == 0).tolist()]

        return actions_permitted   #

    def next_step(self, action):
        '''
        apply input action to current state: (self.robber, self.bank)
        input: action = robber's movement
        return: next state and reward
        '''

        #find next location of robber
        self.robber = self.move_agent('robber', action)

        #todo: move police closer to the robber
        wall_permitted_actions_police = self.check_wall_constraint('police')
        police_movement = np.random.choice(wall_permitted_actions_police)   #police moves randomily!
        self.police = self.move_agent('police', police_movement)

        self.assign_reward()

        new_state_indx = self.states_mapping[(self.robber, self.police)]

        return new_state_indx

    def assign_reward(self):
        '''
        assign reward for moving to state (self.robber, self.police)
        '''

        # the robber got caught by the police
        if np.all(self.robber == self.police):
            self.reward = -10
        # the robber is in the bank but the police is not there
        elif np.all(self.robber == self.bank):
            self.reward = 1
        # the robber and the police are in neighbour cells
        elif np.linalg.norm(np.array([self.robber[0], self.robber[1]])-np.array([self.police[0],self.police[1]])) == 1 :
            self.reward = -5


    def reset_game(self):

        self.robber = (0,0)
        self.police = (3,3)

        state_indx = self.states_mapping[(self.robber, self.police)]

        return state_indx



