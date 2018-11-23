import numpy as np
from math import sqrt
import itertools

class MDP():

    def __init__(self):

        # Max time steps. 
        self.T =15

        #Origin at Top left
        self.GRID_SIZE_X = 6
        self.GRID_SIZE_Y = 5

        self.reward = 0

        '''State'''
        # position of player
        self.p = (0,0)
        # position of minotaur
        self.m = (4,4)
        # dead state
        self.dead = ((-100, -100), (-100, -100))
        # win state for player
        self.win = ((100, 100), (100, 100))

        #number of STATES
        self.NUM_STATES = (self.GRID_SIZE_X * self.GRID_SIZE_Y) ** 2 + 2 # add WIN and DEAD state

        '''Actions:
        0:  Left
        1:  Up
        2:  Right
        3:  Down
        4:  Stay
        5:  move to dead state
        6:  move to win state
        '''
        self.actions_p = [0, 1, 2, 3, 4, 5, 6]
        self.index_actions_p = ['left', 'up', 'right', 'down', 'stay', 'move to dead', 'move to win']

        self.actions_m = [0, 1, 2, 3, 4]
        self.index_actions_m = ['left', 'up', 'right', 'down', 'stay']

        # Number of possible actions for player and Minotaur
        self.n_actions_p = len(self.actions_p)
        self.n_actions_m = len(self.actions_m)  # NOTE:THIS CAN BE SET TO 4/5 for part (c for eg.)
        
        # List of acceptable actions that can be taken by the player (Change with every state)
        self.acceptable_actions_player = []

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
        self.walls[1:self.GRID_SIZE_X-1,0] = [0,1,0,0]
        # cells between BOTTOM LEFT and BOTTOM RIGHT
        self.walls[1:self.GRID_SIZE_X-1,self.GRID_SIZE_Y-1] = [0,0,0,1]
        # cells between TOP LEFT and BOTTOM LEFT
        self.walls[0,1:self.GRID_SIZE_Y-1] = [1,0,0,0]
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

        #self.walls = np.transpose(self.walls, (1,0,2))

    def find_next_location(self, current_location, action, agent):
        '''
        returns location (x_hat, y_hat) after applying input action to current_location
        '''
        x = current_location[0]
        y = current_location[1]

        # if action == 0: #LEFT
        #     next_location = (x,y-1)
        # elif action == 1: #UP
        #     next_location = (x-1,y)
        # elif action == 2: #RIGHT
        #     next_location = (x,y+1)
        # elif action == 3:  #DOWN
        #     next_location = (x+1,y)
        # else: #STAY
        #     next_location = (x,y)

        #if agent == 'player':
        if action == 0: #LEFT
            next_location = (x-1,y)
        elif action == 1: #UP
            next_location = (x,y-1)
        elif action == 2: #RIGHT
            next_location = (x+1,y)
        elif action == 3:  #DOWN
            next_location = (x,y+1)
        else: #STAY
            next_location = (x,y)
    
        if agent == 'minotaur':
            if next_location[0] < 0:    # it moved in -ve along X axis
                next_location = (self.GRID_SIZE_X-1, next_location[1])
            if next_location[0] >= self.GRID_SIZE_X:
                next_location = (0, next_location[1])
            if next_location[1] < 0:   # agent moved -ve in Y axis
                next_location = (next_location[0],self.GRID_SIZE_Y-1)
            if next_location[1] >= self.GRID_SIZE_Y:
                next_location = (next_location[0],0)

            #print('reached here')

        return next_location


    def check_wall_constraint(self, forbidden_actions):
        #NOTE: walls[0,-1] actually means walls[0,5]!!!
        if self.walls[self.p[0],self.p[1],0] == 1:      #player's LEFT side is blocked
            forbidden_actions[0] = 1
        if self.walls[self.p[0],self.p[1],1] == 1:      #player's TOP side is blocked
            forbidden_actions[1] = 1
        if self.walls[self.p[0],self.p[1],2] == 1:      #player's RIGHT side is blocked
            forbidden_actions[2] = 1
        if self.walls[self.p[0],self.p[1],3] == 1:      #player's BOTTOM side is blocked
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

        #dead variable is in tuple form, transform p, m to tuple too
        if np.all(((self.p[0],self.p[1]), (self.m[0], self.m[1])) == self.dead):  #alredy in the dead state, stay there
            self.p_sHat = np.zeros((self.n_actions_p, self.n_actions_m))
            self.p_sHat[5, :] = 0.2 * np.ones(self.p_sHat.shape[1])   #player does a jump to DEAD state, indepentently of minotaur
            return

        #win variable is in tuple form, transform p, m to tuple too
        if np.all(((self.p[0],self.p[1]), (self.m[0], self.m[1])) == self.win):   #player arrived in the winning position, stayed there
            self.p_sHat = np.zeros((self.n_actions_p, self.n_actions_m))
            self.p_sHat[6, :] = 0.2 * np.ones(self.p_sHat.shape[1])      #player does a jump to WIN state, indepentently of minotaur
            return

        if np.all(self.p == self.m):    #player is being killed, moving directly to dead state
            self.p_sHat = np.zeros((self.n_actions_p, self.n_actions_m))
            self.p_sHat[5, :] = 0.2 * np.ones(self.p_sHat.shape[1])      #player does a jump to DEAD state, indepentently of minotaur
            return


        #### CHECK only MOVABLE ACTIONS!
        # Generate all next possible locations for player and minotaur -- save the player's movement too!
        #next_locations_player = [(self.find_next_location(self.p, mov, 'player'), mov) for mov in (self.actions_p[0:5])]  #dont take into account dead and win state
        next_locations_player = [self.find_next_location(self.p, mov, 'player') for mov in (self.actions_p[0:5])]
        next_locations_minotaur = [self.find_next_location(self.m, mov, 'minotaur')  for mov in self.actions_m]

        # Generate all next states when all actions are applied to current state
        all_next_states = [pair for pair in itertools.product(next_locations_player, next_locations_minotaur)]

        # Container for storing forbidden actions for player: 1 means forbidden, free otherwise
        forbidden_actions_player = np.zeros((self.n_actions_p-2), dtype=np.int)   #dont take into account dead and win state

        # check wall restrictions from current state
        forbidden_actions_player = self.check_wall_constraint(forbidden_actions_player)

        self.p_sHat = np.zeros((self.n_actions_p, self.n_actions_m))   #initialize p_sHat
        
        for a_p in forbidden_actions_player:
            if forbidden_actions_player[a_p] == 0:
                for a_m in self.actions_m:
                    if np.all(next_locations_player[a_p] == next_locations_minotaur[a_m]):
                        forbidden_actions_player[a_p] = 1

        # for next_p_tuple, next_m in all_next_states:

        #     next_p = next_p_tuple[0]  #location
        #     p_mov = next_p_tuple[1]   #movement leading to that location


        #     dist = sqrt( (next_m[0] - next_p[0])**2 + (next_m[1] - next_p[1])**2 )


            #cbeck killing-zone restrictions from the next state
            # if dist <= 1:   #minotaur is in KILLING ZONE! very close positions or the same position
            #     if next_p[0] == next_m[0]:    # their x coordinates align
            #         if next_p[1] - next_m[1] == 1:            #Minotaur is going to be on the LEFT of player
            #             forbidden_actions_player[p_mov] = 1
            #             print('minotaur is on the left')
            #         elif next_p[1] - next_m[1] == -1:         #Minotaur is going to be  on the RIGHT of player
            #             forbidden_actions_player[p_mov] = 1
            #             print('minotaur is on the right')
            #         elif next_p[1] - next_m[1] == 0:          #Minotaur and player in the same position
            #             forbidden_actions_player[p_mov] = 1                               #
            #             print('killing position is next')
            #     if next_p[1] == next_m[1]:    # their y coordinates align
            #         if next_p[1] - next_m[1] == 1:            #Minotaur is going to be  on TOP of player
            #             forbidden_actions_player[p_mov] = 1
            #             print('minotaur is on the top')
            #         elif next_p[1] - next_m[1] == -1:         #Minotaur is going to be  on the BOTTOM of player
            #             forbidden_actions_player[p_mov] = 1
            #             print('minotaur is on the bottom')

                
            #     if next_p[0] == next_m[0]:    # their x coordinates align
            #         if next_p[1] - next_m[1] == 1:            #Minotaur is going to be on the LEFT of player
            #             forbidden_actions_player[p_mov] = 1
            #         elif next_p[1] - next_m[1] == -1:         #Minotaur is going to be  on the RIGHT of player
            #             forbidden_actions_player[p_mov] = 1
            #         elif next_p[1] - next_m[1] == 0:          #Minotaur and player in the same position
            #             forbidden_actions_player[p_mov] = 1                               #
            #             print('killing position is next')
            #     if next_p[1] == next_m[1]:    # their y coordinates align
            #         if next_p[1] - next_m[1] == 1:            #Minotaur is going to be  on TOP of player
            #             forbidden_actions_player[p_mov] = 1
            #         elif next_p[1] - next_m[1] == -1:         #Minotaur is going to be  on the BOTTOM of player
            #             forbidden_actions_player[p_mov] = 1

        n_possible_actions_player = np.count_nonzero(forbidden_actions_player == 0) #number of possible safe actions for player

        probability_parameter = 1.0/(n_possible_actions_player * self.n_actions_m)  #since Minotaur can always have all actions

        #self.acceptable_actions_player = list(np.argwhere(forbidden_actions_player == 0)[0])  #acceptable actions for the player
        self.acceptable_actions_player = [value[0] for value in np.argwhere(forbidden_actions_player == 0).tolist()]
        #acceptable_actions = [pair for pair in itertools.product(self.acceptable_actions_player, self.actions_m)]  #acceptable actions for (player, minotaur)


        #update transition probability matrix for current state
        for j in range(len(self.acceptable_actions_player)):
            self.p_sHat[self.acceptable_actions_player[j], :] = probability_parameter  #otherwise probability is zero
        
        if(np.sum(self.p_sHat)!=1):
            print('p_sHat NOT summing to 1 !!!')

        a = self.acceptable_actions_player[0]

    def update_state(self, action):
        '''
        Update input state after input action
        action: tuple representing (player_action, minotaur_action)
        returns the next state (player_location, minotaur_location)
        '''

        player_action = action[0]
        next_player_location = self.find_next_location(self.p, player_action, 'player')

        minotaur_action = action[1]
        next_minotaur_location = self.find_next_location(self.m, minotaur_action, 'minotaur')


        return (next_player_location, next_minotaur_location)


    def value_iteration(self):

        #Generate all possible states (player_location, minotaur_location)
        all_positions = [positions_pair for positions_pair in itertools.product(range(self.GRID_SIZE_X), range(self.GRID_SIZE_Y))]  #all positions for player/minotaur
        all_states = [states_pair for states_pair in itertools.product(all_positions, all_positions)]
        all_states.append(self.dead)   #append dead state
        all_states.append(self.win)       #win state

        #assign an index to every state - useful for accessing the state value of other states
        states_mapping = dict(zip(list(all_states), range(0,self.NUM_STATES)))
        
        # # Generate all possible actions in pairs (player action, minotaur action)
        # all_actions = [pair for pair in itertools.product(self.actions_p, self.actions_m)]

        ## Change all_positions and all_states to numpy array
        all_states = (np.array(all_states)).reshape((self.NUM_STATES,4))
        #all_actions = np.array(all_actions)
       
        state_values = np.zeros((self.NUM_STATES, self.T))   #keep the best state values computed in each iteration
        policy = np.zeros((self.NUM_STATES, self.T))   #store the best sequence of actions for the player

        #####base case of dynamic programming - compute state value at timestep T
        # WIN state
        win_state_indx = states_mapping[self.win]
        state_values[win_state_indx , self.T-1] = 1

        # states (4,4,_,_) are equivalent to WIN state
        winning_position_states_indx = [states_mapping[((4,4), minotaur_pos)] for minotaur_pos in all_positions]
        state_values[winning_position_states_indx, self.T-1] = 1

        # in all other states, terminal reward is zero

        for t in range(self.T-2, -1, -1):
            for i in range(self.NUM_STATES):

                #change current state
                self.p = all_states[i,0:2]
                self.m = all_states[i,2:4]

                #generate state transition matrix self.p_sHat for state
                self.calc_transition_prob()
                
                #if np.count_nonzero(self.p_sHat[0,4,:]) == 0:
                if np.sum(self.p_sHat[5,:]) == 1: # For sure dead x(
                    state_values[i,t] = 0
                    policy[i,t] = 5 #dead state
                else:
                    # Generate all possible actions in pairs (player action, minotaur action)
                    all_acceptable_actions = [pair for pair in itertools.product(self.acceptable_actions_player, self.actions_m)]
                    #all_acceptable_actions = np.array(all_actions)

                    action_returns = []
                    for action_p, action_m in all_acceptable_actions:

                        ######perform one step of policy evaluation
                        #apply action to state and get next_state
                        next_state = self.update_state((action_p, action_m))

                        #find reward for current state: if state is WIN then reward is 1, 0 otherwise
                        if np.all(self.p == self.win):
                            self.reward = 1

                        #define the probability of performing that action
                        transition_prob = self.p_sHat[action_p, action_m]

                        #if next_state in states_mapping.keys():   #next state belongs to our state space
                        next_state_indx = states_mapping[next_state]
                        expected_reward = transition_prob * (self.reward + state_values[next_state_indx, t+1 ])
                        action_returns.append(expected_reward)
                       # else:
                        #    expected_reward = 0
                         #   action_returns.append(expected_reward)

                        # greedy improvement policy: keep maximum expected reward
                        new_state_value = np.max(action_returns)
                        #delta += np.abs(state_values[i, t] - new_state_value)

                        # update state value
                        state_values[i, t] = new_state_value

                        # keep player's action that lead to the best state value
                        policy[i, t] = all_acceptable_actions[np.argmax(np.round(action_returns, 5))][0]

            #if delta < 1e-9:
            #    break

        #transform action index to strings
        policy_player_final = [self.index_actions_p[action_indx] for action_indx in policy]

        print(policy_player_final)



def test():

    mdp_obj = MDP()


test()
