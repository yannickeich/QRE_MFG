import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class LR(FastMARLEnv):
    """
    Models the Left Right game.
    """

    def __init__(self, time_steps: int = 1,
                 cost_action: float = 0.5,
                 num_agents: int = 100, **kwargs):

        self.cost_action = cost_action

        observation_space = Discrete(3)
        action_space = Discrete(2)

        mu_0 = np.array([0.0,1.0,0.0])

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

        #Go left
        P[0, 1, 0] = 1.0
        P[0, 0, 0] = 1.0
        P[0, 2, 2] = 1.0
        #Go right
        P[1, 1, 2] = 1.0
        P[1, 0, 0] = 1.0
        P[1, 2, 2] = 1.0
        return P

    def get_R(self, t, mu):
        R = np.zeros((self.observation_space.n, self.action_space.n))

        R[0, :] -= 2*mu[0]
        R[2, :] -= mu[2]

        return R

    def final_R(self,mu):
        R = np.zeros(self.observation_space.n)

        R[0] -= 2*mu[0]
        R[2] -= mu[2]

        return R