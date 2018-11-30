import numpy as np
from math import sqrt
import itertools
import matplotlib.pyplot as plt

class MDP():

    def __init__(self):

        # Max time steps. 
        self.T =15

        #Origin at Top left
        self.GRID_SIZE_COL = 6
        self.GRID_SIZE_ROW = 5

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


        #states in form (x,y,x,y) correspond to killing position. all these states are modelled with state ((-100,-100),(-100,-100))
        self.killing_states = [(position, position) for position in
                          itertools.product(range(self.GRID_SIZE_ROW), range(self.GRID_SIZE_COL)) if np.all(position != (4,4))]  #state ((4,4)(4,4)) is included in winning set

        # states in form (4,4,x,y) correspond to winning positions. all these states are modelled with state ((100,100),(100,100))
        self.winning_states = [((4, 4), position) for position in
                          itertools.product(range(self.GRID_SIZE_ROW), range(self.GRID_SIZE_COL))]

        # number of STATES
        # remove states that belong to killing and winning set
        self.NUM_STATES = (self.GRID_SIZE_COL * self.GRID_SIZE_ROW) ** 2 + 2  - (len(self.killing_states) + len(self.winning_states))# add WIN and DEAD state

        '''Actions:
        0:  Left
        1:  Up
        2:  Right
        3:  Down
        4:  Stay
        '''
        self.actions_p = [0, 1, 2, 3, 4]
        self.index_actions_p = ['left', 'up', 'right', 'down', 'stay']

        self.actions_m = [0, 1, 2, 3, 4]
        self.index_actions_m = ['left', 'up', 'right', 'down', 'stay']

        self.win_mov = 10  #leads to win state
        self.dead_mov = -10  #leads to dead state

        # Number of possible actions for player and Minotaur
        self.n_actions_p = len(self.actions_p)
        self.n_actions_m = len(self.actions_m)  # NOTE:THIS CAN BE SET TO 4/5 for part (c for eg.)

        # List of acceptable actions that can be taken by the player (Change with every state)
        self.acceptable_actions_player = []

        # Container for walls
        self.walls = np.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL, 4), dtype=np.int)

        # State transition matrix: each cell (x,y) contains the probability of performing the action (x,y)
        self.p_sHat = np.zeros((self.n_actions_p, self.n_actions_m))

        '''Member Function'''
        self.define_wall()
        # simulate game (movement between player and minotaur)
        self.game_grid = []
        self.define_grid()

        # State transition probabilities
        self.all_states = []
        self.states_mapping = []
        self.state_values = []
        self.policy = []
        self.dynamic_programming()

        # Forward iteration to simulate one/several game
        self.forward_iteration()

    def define_grid(self):
        self.game_grid.append('###|######')
        self.game_grid.append('#  |     #')
        self.game_grid.append('#  |  |__#')
        self.game_grid.append('#  |  |  #')
        self.game_grid.append('# ______ #')
        self.game_grid.append('#     |  #')
        self.game_grid.append('###|######')

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

        '''Custom wall cells: depend on map
            SET MANUALLY!!
        '''
        self.walls[0:3, 1, 2] = 1
        self.walls[0:3, 2, 0] = 1

        self.walls[self.GRID_SIZE_ROW-1, 1:5,1] = 1
        self.walls[self.GRID_SIZE_ROW-2,1:5,3] = 1

        self.walls[self.GRID_SIZE_ROW-1,3,2] = 1
        self.walls[self.GRID_SIZE_ROW-1,4,0] = 1

        self.walls[1:3, 3,2] = 1
        self.walls[1:3, 4,0] = 1

        self.walls[1,4:self.GRID_SIZE_COL,3] = 1
        self.walls[2, 4:self.GRID_SIZE_COL,1] = 1

        #self.walls = np.transpose(self.walls, (1,0,2))

    def find_next_location(self, current_location, action, agent):
        '''
        returns location (x_hat, y_hat) after applying input action to current_location
        '''
        x = current_location[0]
        y = current_location[1]

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

        if agent == 'minotaur':
            if next_location[0] < 0:      # it moved in -ve along ROW axis
                next_location = (self.GRID_SIZE_ROW-1, next_location[1])
            if next_location[0] >= self.GRID_SIZE_ROW:  #It exceeded the Max ROW
                next_location = (0, next_location[1])
            if next_location[1] < 0:   # agent moved -ve in COL axis
                next_location = (next_location[0],self.GRID_SIZE_COL-1)
            if next_location[1] >= self.GRID_SIZE_COL:  #It exceeded Max COL
                next_location = (next_location[0],0)

        return next_location

    def check_wall_constraint(self, forbidden_actions):
        # NOTE: walls[0,-1] actually means walls[0,5]!!!
        if self.walls[self.p[0], self.p[1], 0] == 1:  # player's LEFT side is blocked
            forbidden_actions[0] = 1
        if self.walls[self.p[0], self.p[1], 1] == 1:  # player's TOP side is blocked
            forbidden_actions[1] = 1
        if self.walls[self.p[0], self.p[1], 2] == 1:  # player's RIGHT side is blocked
            forbidden_actions[2] = 1
        if self.walls[self.p[0], self.p[1], 3] == 1:  # player's BOTTOM side is blocked
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

       #### CHECK only MOVABLE ACTIONS!
        # Generate all next possible locations for player and minotaur -- save the player's movement too!
        next_locations_player = [self.find_next_location(self.p, mov, 'player') for mov in (self.actions_p)]
        next_locations_minotaur = [self.find_next_location(self.m, mov, 'minotaur')  for mov in self.actions_m]

        # Container for storing forbidden actions for player: 1 means forbidden, free otherwise
        forbidden_actions_player = np.zeros((self.n_actions_p), dtype=np.int)   # dont take into account dead and win state

        # check wall restrictions from current state
        forbidden_actions_player = self.check_wall_constraint(forbidden_actions_player)

        self.p_sHat = np.zeros((self.n_actions_p, self.n_actions_m))  # initialize p_sHat

        for a_p in forbidden_actions_player:
            if forbidden_actions_player[a_p] == 0:
                for a_m in self.actions_m:
                    if np.all(next_locations_player[a_p] == next_locations_minotaur[a_m]):
                        forbidden_actions_player[a_p] = 1


        n_possible_actions_player = np.count_nonzero(
            forbidden_actions_player == 0)  # number of possible safe actions for player

        probability_parameter = 1.0 / ( n_possible_actions_player * self.n_actions_m )  # since Minotaur can always have all actions

        self.acceptable_actions_player = [value[0] for value in np.argwhere(forbidden_actions_player == 0).tolist()]

        # update transition probability matrix for current state
        for j in range(len(self.acceptable_actions_player)):
            self.p_sHat[self.acceptable_actions_player[j], :] = probability_parameter  # otherwise probability is zero

        if (round(np.sum(self.p_sHat), 2) != 1):
            print('p_sHat NOT summing to 1 !!!')

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

        #if player is in (4,4) move to winning state!
        if np.all(next_player_location == np.array([4,4]) ):
            return self.win

        # if player is in the same position as the minotaur then move to dead state!
        if np.all(next_player_location == next_minotaur_location):
            return self.dead

        return (next_player_location, next_minotaur_location)

    def dynamic_programming(self):

        self.define_grid()

        # Generate all possible states (player_location, minotaur_location)
        all_positions = [positions_pair for positions_pair in itertools.product(range(self.GRID_SIZE_ROW), range(self.GRID_SIZE_COL))]

        self.all_states = [states_pair for states_pair in itertools.product(all_positions, all_positions) if states_pair not in self.killing_states and states_pair not in self.winning_states]  # all positions for player/minotaur]
        self.all_states.append(self.dead)  # append dead state
        self.all_states.append(self.win)  # win state

        # assign an index to every state - useful for accessing the state value of other states
        self.states_mapping = dict(zip(list(self.all_states), range(0, self.NUM_STATES)))

        ## Change all_positions and self.all_states to numpy array
        self.all_states = (np.array(self.all_states)).reshape((self.NUM_STATES, 4))
        # all_actions = np.array(all_actions)

        self.state_values = np.zeros((self.NUM_STATES, self.T))  # keep the best state values computed in each iteration
        self.policy = np.zeros((self.NUM_STATES, self.T))  # store the best sequence of actions for the player

        #####base case of dynamic programming - compute state value at timestep T
        # WIN state
        win_state_indx = self.states_mapping[self.win]
        self.state_values[win_state_indx , self.T-1] = 1

        # in all other states, terminal reward is zero
        for t in range(self.T - 2, -1, -1):
            print('----------------------------------')
            print('Time: ' + str(t + 1))
            for i in range(self.NUM_STATES):
                # print('State: ' + str(self.all_states[i,:]))

                # change current state
                self.p = self.all_states[i, 0:2]
                self.m = self.all_states[i, 2:4]

                if np.all(((self.p[0], self.p[1]), (self.m[0], self.m[1])) == self.win):
                    self.state_values[i, t] = 100   #we dont care about the value since its a terminal state?
                    self.policy[i,t] = self.win_mov

                elif np.all(((self.p[0], self.p[1]), (self.m[0], self.m[1])) == self.dead):
                    self.state_values[i, t] = -100  #we dont care about the value since its a terminal state?
                    self.policy[i, t] = self.dead_mov
                else:

                    # generate state transition matrix self.p_sHat for state
                    self.calc_transition_prob()

                    # Generate all possible actions in pairs (player action, minotaur action)
                    all_acceptable_actions = [pair for pair in
                                              itertools.product(self.acceptable_actions_player, self.actions_m)]

                    action_returns = []
                    for action_p, action_m in all_acceptable_actions:

                        ######perform one step of self.policy evaluation
                        # apply action to state and get next_state
                        next_state = self.update_state((action_p, action_m))

                        # find reward for current state: if state is WIN then reward is 1, 0 otherwise
                        if np.all(self.p == self.win):
                            self.reward = 1

                        # define the probability of performing that action
                        transition_prob = self.p_sHat[action_p, action_m]

                        next_state_indx = self.states_mapping[next_state]
                        expected_reward = transition_prob * (self.reward + self.state_values[next_state_indx, t + 1])
                        action_returns.append(expected_reward)

                    new_state_value = np.max(action_returns)

                    # update state value
                    self.state_values[i, t] = new_state_value

                    # keep player's action that lead to the best state value
                    self.policy[i, t] = all_acceptable_actions[np.argmax(action_returns)][0]


        print('got here')


    def forward_iteration(self):

        '''PLOT RESULTS'''
        # x axis of plot: Time 't'
        #xx = np.linspace(1, 15, 15)
        # y axis of plot: Maximal probability of exiting the maze at time 't'
        #yy = np.amax(self.state_values, axis=0)
        #plt.plot(xx, yy)
        #plt.savefig('problem1b_maxprob.png')

        num_wins = 0
        num_games = 10000

        # distribution of win and dead for plotting histograms
        distribution_win = np.zeros(self.T)
        distribution_dead = np.zeros(self.T)

        for i in range(num_games):
            policy_list = []
            state_list = []
            current_state = ((0,0),(4,4))
            state_list.append(current_state)
            self.p = np.array(current_state[0])
            self.m = np.array(current_state[1])

            for t in range(self.T-1):
                # find state mapping for current state
                current_state_idx = self.states_mapping[current_state]
                # find self.policy for player movement
                player_mov = self.policy[current_state_idx,t]
                # generate random action for minotaur
                minotaur_mov = np.random.randint(0,5)
                # find next state
                next_state = self.update_state(( player_mov, minotaur_mov))
                state_list.append(next_state)
                policy_list.append(player_mov)

                if np.all(next_state == self.win):
                    print("FREEDOM!")
                    num_wins += 1
                    # note time when player won
                    distribution_win[t] += 1
                    break

                if np.all(next_state == self.dead):
                    print("DEAD!")
                    # note time when player died
                    distribution_dead[t] += 1
                    break

                current_state = next_state
                self.p = np.array(current_state[0])
                self.m = np.array(current_state[1])


            #print(state_list)
        # Plot histogram of player's performance
        xx = np.linspace(0, self.T-1, self.T)
        distribution_win[:] = [x / num_games for x in distribution_win]
        distribution_dead[:] = [x / num_games for x in distribution_dead]
        plt.hist(np.array(distribution_win), bins=xx)
        plt.hist(np.array(distribution_dead), bins=xx) 
        plt.title("Max. probability of exiting maze v/s time")
        plt.xlabel('time')
        plt.ylabel('Probability')
        plt.legend(['Player won', 'Player died'])
        plt.show()
        print('Total number of wins:', num_wins, 'out of %d games' %num_games)


def test():
    MDP()


test()
