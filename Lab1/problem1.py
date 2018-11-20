import numpy as np
from math import sqrt
import itertools

class MDP():

    def __init__(self):
        #Origin at Top left
        self.GRID_SIZE_X = 6
        self.GRID_SIZE_Y = 5

        # Number of possible actions for player and Minotaur
        self.n_actions_p = 5
        self.n_actions_m = 5    #NOTE:THIS CAN BE SET TO 4/5 for part (c for eg.)

        self.reward = 0

        '''State'''
        # position of player
        self.p = (0,0)
        # position of minotaur
        self.m = (4,4)
        # dead state
        self.dead = 0
        # win state for player
        self.win = (4,4)

        #number of STATES
        #self.NUM_STATES = (self.GRID_SIZE_X * self.GRID_SIZE_Y) ^ 2
        self.NUM_STATES = (self.GRID_SIZE_X * self.GRID_SIZE_Y) ** 2

        '''Actions:
        0:  Left
        1:  Up
        2:  Right
        3:  Down
        4:  Stay
        '''
        self.actions = [0, 1, 2, 3, 4]
        self.index_actions = ['left', 'up', 'right', 'down', 'stay']

        #self.action_p = 5  # for player //
        #self.action_m = 5   # for minotaur //

        # Container for walls
        self.walls = np.zeros((self.GRID_SIZE_X, self.GRID_SIZE_Y, 4), dtype=np.int)

        # State transition matrix: each cell (x,y) contains the probability of performing the action (x,y)
        self.p_sHat = np.zeros((self.n_actions_p,self.n_actions_m))

        '''Member Function'''
        self.define_wall()

        # State transition probabilities
        self.value_iteration()

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


    def find_next_location(self, current_location, action):
        '''
        returns location (x_hat, y_hat) after applying input action to current_location
        '''
        x = current_location[0]
        y = current_location[1]

        next_location = (x, y)
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


    def check_wall_constraint(self, forbidden_actions):
        if self.walls[self.p[0],self.p[1],0] == 1:      #player's LEFT side is blocked
            forbidden_actions[0] = 1
        elif self.walls[self.p[0],self.p[1],1] == 1:      #player's TOP side is blocked
            forbidden_actions[1] = 1
        elif self.walls[self.p[0],self.p[1],2] == 1:      #player's RIGHT side is blocked
            forbidden_actions[2] = 1
        elif self.walls[self.p[0],self.p[1],3] == 1:      #player's BOTTOM side is blocked
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

        # Generate all next possible locations for player and minotaur
        next_locations_player = [self.find_next_location(self.p, mov) for mov in self.actions]
        next_locations_minotaur = [self.find_next_location(self.m, mov) for mov in self.actions]

        # Generate all next states when all actions are applied to current state
        all_next_states = [pair for pair in itertools.product(next_locations_player, next_locations_minotaur)]

        # Container for storing forbidden actions for player: 1 means forbidden, free otherwise
        forbidden_actions_player = np.zeros((self.n_actions_p), dtype=np.int)

        for next_p, next_m in all_next_states:

            dist = sqrt( (next_m[0] - next_p[0])**2 + (next_m[1] - next_p[1])**2 )

            if dist == 1:   #minotaur is in KILLING ZONE!
                if next_p[0] == next_m[0]:    # their x coordinates align
                    if next_p[1] - next_m[1] == 1:            #Minotaur is going to be on the LEFT of player
                        forbidden_actions_player[0] = 1    #LEFT motion for player forbidden
                        forbidden_actions_player[4] = 1
                    elif next_p[1] - next_m[1] == -1:         #Minotaur is going to be  on the RIGHT of player
                        forbidden_actions_player[2] = 1    #RIGHT motion for player forbidden
                        forbidden_actions_player[4] = 1
                if next_p[1] == next_m[1]:    # their y coordinates align
                    if next_p[1] - next_m[1] == 1:            #Minotaur is going to be  on TOP of player
                        forbidden_actions_player[1] = 1    #TOP action for player forbidden
                        forbidden_actions_player[4] = 1
                    elif next_p[1] - next_m[1] == -1:         #Minotaur is going to be  on the BOTTOM of player
                        forbidden_actions_player[3] = 1    #BOTTOM action for player forbidden
                        forbidden_actions_player[4] = 1

        forbidden_actions_player = self.check_wall_constraint(forbidden_actions_player)

        n_possible_actions_player = np.count_nonzero(forbidden_actions_player == 0) #number of possible safe actions for player

        probability_parameter = 1.0/(n_possible_actions_player * self.n_actions_m)  #since Minotaur can always have all actions

        acceptable_actions_player = list(np.argwhere(forbidden_actions_player == 0))  #acceptable actions for the player

        acceptable_actions = [pair for pair in itertools.product(acceptable_actions_player, self.actions)]  #acceptable actions for (player, minotaur)


        #update transition probability matrix for current state
        for j in range(len(acceptable_actions)):
            self.p_sHat[acceptable_actions[j], :] = probability_parameter  #otherwise probability is zero


    def update_state(self, action):
        '''
        Update input state after input action
        action: tuple representing (player_action, minotaur_action)
        returns the next state (player_location, minotaur_location)
        '''

        player_action = action[0]
        next_player_location = self.find_next_location(self.p, player_action)

        minotaur_action = action[1]
        next_minotaur_location = self.find_next_location(self.m, minotaur_action)

        return (next_player_location, next_minotaur_location)


    def value_iteration(self):

        #Generate all possible states (player_location, minotaur_location)
        all_positions = [positions_pair for positions_pair in itertools.product(range(self.GRID_SIZE_X), range(self.GRID_SIZE_Y))]  #all positions for player/minotaur
        all_states = [states_pair for states_pair in itertools.product(all_positions, all_positions)]

        # Generate all possible actions in pairs (player action, minotaur action)
        all_actions = [pair for pair in itertools.product(self.actions, self.actions)]

        state_values = np.zeros(self.NUM_STATES)   #keep the best state values computed in each iteration
        policy = np.zeros(self.NUM_STATES + 1)   #store the best sequence of actions for the player

        while True:
            delta = 0.0
            for i in range(self.NUM_STATES):

                #change current state
                self.p = all_states[i][0]
                self.m = all_states[i][1]

                #generate state transition matrix self.p_sHat for state
                self.calc_transition_prob()

                action_returns = []
                for action_p, action_m in all_actions:

                    ######perform one step of policy evaluation
                    #apply action to state and get next_state
                    next_state = self.update_state((action_p, action_m))

                    #find reward for current state: if state is WIN then reward is 1, 0 otherwise
                    if self.p == self.win:
                        self.reward = 1

                    #define the probability of performing that action
                    transition_prob = self.p_sHat[action_p, action_m]

                    #compute expected_reward for the pair (state, action)
                    expected_reward = transition_prob * (self.reward + state_values[next_state])   #gamma?
                    action_returns.append(expected_reward)

                # greedy improvement policy: keep maximum expected reward
                new_state_value = np.max(action_returns)
                delta += np.abs(state_value[state] - new_state_value)

                # update state value
                state_value[i] = new_state_value

                # keep player's action that lead to the best state value
                policy[i] = all_actions[np.argmax(np.round(action_returns, 5))][0]

            if delta < 1e-9:
                break

        #transform action index to strings
        policy_player_final = [self.index_actions[action_indx] for action_indx in policy]

        print(policy_player_final)



def test():

    mdp_obj = MDP()


test()
