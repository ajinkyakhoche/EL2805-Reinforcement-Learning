import numpy as np
from math import sqrt

class MDP():

    def __init__(self):
        #Origin at Top left
        self.GRID_SIZE_X = 6
        self.GRID_SIZE_Y = 5

        # Number of possible actions for player and Minotaur 
        self.n_actions_p = 5    
        self.n_actions_m = 5    #NOTE:THIS CAN BE SET TO 4/5 for part (c for eg.)

        '''State'''
        # position of player
        self.p = (0,0)
        # position of minotaur
        self.m = (4,4)
        # dead state
        self.dead = 0
        # win state
        self.win = 0

        '''Actions:
        0:  Left
        1:  Up
        2:  Right
        3:  Down
        4:  Stay
        '''
        self.action_p = 4   # for player
        self.action_m = 4   # for minotaur

        # Container for walls
        self.walls = np.zeros((self.GRID_SIZE_X, self.GRID_SIZE_Y, 4), dtype=np.int)
        
        # State transition matrix
        self.p_sHat = np.zeros((self.n_actions_p,self.n_actions_m))

        '''Member Function''' 
        self.define_wall()
        # State transition probabilities
        self.calc_transition_prob()

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
        # define corners
        self.walls[0,0] = [1,1,0,0]                                     # TOP LEFT
        self.walls[self.GRID_SIZE_X-1,self.GRID_SIZE_Y-1] = [0,0,1,1]   # BOTTOM RIGHT
        self.walls[self.GRID_SIZE_X-1, 0] = [0,1,1,0]
        self.walls[0,self.GRID_SIZE_Y-1] = [1,0,0,1]

        # cells between TOP LEFT and TOP RIGHT
        self.walls[1:self.GRID_SIZE_X-2,0] = [0,1,0,0]
        # cells between BOTTOM LEFT and BOTTOM RIGHT
        self.walls[1:self.GRID_SIZE_X-2,self.GRID_SIZE_Y-1] = [0,0,0,1]
        # cells between TOP LEFT and BOTTOM LEFT
        self.walls[0,1:self.GRID_SIZE_Y-2] = [1,0,0,0]
        # cells between TOP RIGHT and BOTTOM RIGHT
        self.walls[self.GRID_SIZE_X-1,1:self.GRID_SIZE_Y-2] = [0,0,1,0]

        '''Custom wall cells: depend on map
            SET MANUALLY!!
        '''
        self.walls[1,:0:3, 2] = 1
        self.walls[2,:0:3, 0] = 1

        self.walls[1:5,self.GRID_SIZE_Y-1,1] = 1
        self.walls[1:5,self.GRID_SIZE_Y-2,3] = 1

        self.walls[3,self.GRID_SIZE_Y-1,2] = 1
        self.walls[4,self.GRID_SIZE_Y-1,0] = 1

        self.walls[3,1:3,2] = 1
        self.walls[4,1:3,0] = 1

        self.walls[4:self.GRID_SIZE_X, 1,3] = 1
        self.walls[4:self.GRID_SIZE_X, 2,1] = 1
    
    def check_wall_constraint(self, forbidden_actions):
        if self.walls[self.p, 0] == 1:      #player's LEFT side is blocked
            forbidden_actions[0] = 1
        elif self.walls[self.p, 1] == 1:      #player's TOP side is blocked
            forbidden_actions[1] = 1
        elif self.walls[self.p, 2] == 1:      #player's RIGHT side is blocked
            forbidden_actions[2] = 1
        elif self.walls[self.p, 3] == 1:      #player's BOTTOM side is blocked
            forbidden_actions[3] = 1
        
        return forbidden_actions

    def calc_transition_prob(self):
        '''
        This function takes self.p, self.m as inputs
        -   it examines the state and calculates set of possible actions
        -   The set of possible actions lead to a state transition matrix

        State transition matrix:    p_sHat (5x5 because 5 actions possible for p and m each)

        p_actions →        0                   1              2                 3               4   
        m_actions ↓
                    0   p_left,m_left       p_up,m_left     ...
                    1   p_left,m_up           p_up,m_up     .
                    2   p_left,m_right     p_up,m_right     p_right,m_right          
                    3   p_left,m_down       p_up,m_down     .                    p_down,m_down
                    4   p_left,m_stay       p_up,m_stay     .                                    p_stay,m_stay

        so if p_sHat[2,1] = 1/25, this means probability that player moved up and minotaur moved right is 1/25          
        '''
        # Container for storing forbidden actions for player: 1 means forbidden, free otherwise
        forbidden_actions = np.zeros((self.n_actions_p,), dtype=np.int)
        
        dist = sqrt( (self.m[0] - self.p[0])**2 + (self.m[1] - self.p[1])**2 ) 
        
        if dist == 1:   #minotaur is in KILLING ZONE!
            if self.p[0] == self.m[0]:    # their x coordinates align
                if self.p[1] - self.m[1] == 1:            #Minotaur is on the LEFT of player
                    forbidden_actions[0] = 1    #LEFT motion forbidden
                    forbidden_actions[4] = 1    
                elif self.p[1] - self.m[1] == -1:         #Minotaur is on the RIGHT of player
                    forbidden_actions[2] = 1    #RIGHT motion forbidden    
                    forbidden_actions[4] = 1
            if self.p[1] == self.m[1]:    # their y coordinates align
                if self.p[1] - self.m[1] == 1:            #Minotaur is on TOP of player
                    forbidden_actions[1] = 1    #TOP action forbidden
                    forbidden_actions[4] = 1
                elif self.p[1] - self.m[1] == -1:         #Minotaur is on the BOTTOM of player
                    forbidden_actions[3] = 1    #BOTTOM action forbidden
                    forbidden_actions[4] = 1

        forbidden_actions = self.check_wall_constraint(forbidden_actions)

        n_possible_actions = np.count_nonzero(forbidden_actions==0) #number of possible safe actions for player

        probability_parameter = 1.0/(n_possible_actions * self.n_actions_m)  #since Minotaur can always have all actions

        movable_positions = np.argwhere(forbidden_actions == 0)

        for j in range(movable_positions):
            self.p_sHat[:,movable_positions[j]] = probability_parameter



