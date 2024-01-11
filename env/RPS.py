import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class RPS(FastMARLEnv):
    """
    Models the Rock Paper Scissor game.
    """

    def __init__(self, time_steps: int = 100,
                 num_agents: int = 100, **kwargs):

        # Initial state and a state for rock, paper, scissors each.
        observation_space = Discrete(3)
        action_space = Discrete(3)

        mu_0 = np.array([1.,1.,1.])/3

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

        # Choose Rock
        # P[0, 0, 1] = 1.0
        # #already in end position -> stay
        P[0, 0, 0] = 1.0
        #P[0, 0, 1] = 0.1
        #P[0, 0, 2] = 0.1

        P[0,1,0] = 1.0
        #P[0, 1, 1] = 0.2

        P[0,2,0] = 1.0
        #P[0, 2, 2] = 0.2

        # Choose Paper
        # P[1, 0, 2] = 1.0
        # # already in end position -> stay
        P[1, 0, 1] = 0.8
        P[1, 0, 0] = 0.2

        P[1, 1, 1] = 0.8
        P[1, 1, 0] = 0.1
        P[1, 1, 2] = 0.1

        P[1, 2, 1] = 0.8
        P[1, 2, 2] = 0.2

        # Choose Scissor
        # P[2, 0, 3] = 1.0
        # # already in end position -> stay
        P[2, 0, 2] = 0.5
        P[2, 0, 0] = 0.5

        P[2, 1, 2] = 0.5
        P[2, 1, 1] = 0.5

        P[2, 2, 2] = 0.5
        P[2, 2, 0] = 0.25
        P[2, 2, 1] = 0.25

        return P

    def get_R(self, t, mu):
        R = np.zeros((self.observation_space.n,self.action_space.n))

        # Rock loses to Paper and wins against scissor
        R[0] = - mu[1] + mu[2]
        # Paper loses to Scissor
        R[1] = - mu[2] + 2*mu[0]
        # Scissor loses to Rock
        R[2] = - mu[0] + 3*mu[1]

        return R

    def final_R(self,mu):
        R = np.zeros(self.observation_space.n)

        # Rock loses to Paper and wins against scissor
        R[0] = - mu[1] + mu[2]
        # Paper loses to Scissor
        R[1] = - mu[2] + 2 * mu[0]
        # Scissor loses to Rock
        R[2] = - mu[0] + 3 * mu[1]

        return R