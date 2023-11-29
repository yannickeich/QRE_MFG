import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class RPS(FastMARLEnv):
    """
    Models the Rock Paper Scissor game.
    """

    def __init__(self, time_steps: int = 1,
                 num_agents: int = 100, **kwargs):

        # Initial state and a state for rock, paper, scissors each.
        observation_space = Discrete(4)
        action_space = Discrete(3)

        mu_0 = np.array([1.0,0.0,0.0,0.0])

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
        P[0, 0, 1] = 1.0
        #already in end position -> stay
        P[0,1,1] = 1.0
        P[0,2,2] = 1.0
        P[0,3,3] = 1.0

        # Choose Paper
        P[1, 0, 2] = 1.0
        # already in end position -> stay
        P[1, 1, 1] = 1.0
        P[1, 2, 2] = 1.0
        P[1, 3, 3] = 1.0

        # Choose Scissor
        P[2, 0, 3] = 1.0
        # already in end position -> stay
        P[2, 1, 1] = 1.0
        P[2, 2, 2] = 1.0
        P[2, 3, 3] = 1.0

        return P

    def get_R(self, t, mu):

      raise NotImplementedError

    def final_R(self,mu):
        R = np.zeros(self.observation_space.n)

        # Rock loses to Paper
        R[1] -= mu[2]
        # Paper loses to Scissor
        R[2] -= mu[3]
        # Scissor loses to Rock
        R[3] -= mu[1]

        return R