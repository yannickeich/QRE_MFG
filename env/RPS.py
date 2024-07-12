import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class RPS(FastMARLEnv):
    """
    Models the Rock Paper Scissor game.
    """

    def __init__(self, time_steps: int = 5,mu_0 = np.array([1.,0.,0.,0.]),
                 num_agents: int = 100, **kwargs):

        # Initial state and a state for rock, paper, scissors each.
        observation_space = Discrete(4)
        action_space = Discrete(3)

        mu_0 = np.array([1.,0.,0.,0.])

        super().__init__(observation_space, action_space,
                         time_steps, mu_0, num_agents=num_agents, **kwargs)

    def reset(self):
        self.t = 0
        self.xs = self.sample_initial_states()

        return self.get_observation()

    def next_states(self, t, xs, us):

        raise NotImplementedError

    def reward(self, t, xs, us):

        raise NotImplementedError

    def get_P(self, t, mu):


        P = np.zeros((self.action_space.n, self.observation_space.n, self.observation_space.n))
        if mu.ndim == 2:
            P = np.zeros((mu.shape[0],self.action_space.n, self.observation_space.n, self.observation_space.n))

        # # inital state
        # P[0, 0, 1] = 1.0
        # P[1, 0, 2] = 1.0
        # P[2, 0, 3] = 1.0
        #
        # ### in Rock state it is safe to choose the next state
        # P[0, 1, 1] = 1.0
        # P[1, 1, 2] = 1.0
        # P[2, 1, 3] = 1.0
        #
        # ### in Paper state it is 50 50
        # # try rock
        # P[0, 2, 1] = 0.5
        # P[0, 2, 2] = 0.5
        # # try scissor
        # P[2, 2, 2] = 0.5
        # P[2, 2, 3] = 0.5
        # # try paper
        # P[1, 2, 2] = 0.5
        # P[1,2,1] = 0.25
        # P[1,2,3] = 0.25
        #
        # ### in Scissor state is hard to change the state
        #
        # P[2, 3, 3] = 1.0
        #
        # P[0, 3, 3] = 0.8
        # P[0, 3, 1] = 0.2
        #
        # P[1, 3, 3] = 0.8
        # P[1, 3, 2] = 0.2

        # inital state
        P[...,0, 0, 1] = 1.0
        P[...,1, 0, 2] = 1.0
        P[...,2, 0, 3] = 1.0

        ##State Rock:
        #Stay in rock
        P[...,0, 1, 1] = 1.0
        #Move to paper
        P[...,1,1,2] = 1 - mu[...,2]
        P[...,1,1,1] = mu[...,2]
        # Move to Scissor
        P[...,2,1,3] = 1.0 - mu[...,3]
        P[...,2,1,1] = mu[...,3]

        ##State Paper:
        # Stay in Paper
        P[...,1, 2, 2] = 1.0
        # Move to rock
        P[...,0, 2, 2] = mu[...,1]
        P[...,0, 2, 1] = 1-mu[...,1]
        # Move to Scissor
        P[...,2, 2, 3] = 1.0 - mu[...,3]
        P[...,2, 2, 2] = mu[...,3]


        ##State Scissor:
        # Stay in Scissor
        P[...,2, 3, 3] = 1.0
        # Move to paper
        P[...,1, 3, 2] = 1 - mu[...,2]
        P[...,1, 3, 3] = mu[...,2]
        # Move to Rock..
        P[...,0, 3, 1] = 1.0 - mu[...,1]
        P[...,0, 3, 3] = mu[...,1]

        return P

    def get_R(self, t, mu):
        R = np.zeros((*mu.shape,self.action_space.n))

        # Rock loses to Paper and wins against scissor
        R[...,1,:] = - 10 * mu[...,2] + mu[...,3]
        # Paper loses to Scissor
        R[...,2,:] = -10* mu[...,3] + 10 * mu[...,1]
        # Scissor loses to Rock
        R[...,3,:] = - mu[...,1] + 10* mu[...,2]

        return R

    #TODO:  For now, dont use final rewards, because they are not included in lookahead q functions
    # def final_R(self,mu):
    #     R = np.zeros(self.observation_space.n)
    #
    #     return R