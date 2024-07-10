import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class A3_MDP(FastMARLEnv):
    """
    Models a deterministic MDP
    """

    def __init__(self, time_steps: int = 9, mu_0 = np.array([0.2,0.2,0.2,0.2,0.2]),
                 num_agents: int = 100, **kwargs):

        # Initial state and a state for rock, paper, scissors each.
        observation_space = Discrete(5)
        action_space = Discrete(3)



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

        # Action 0: Go to state 1
        P[0, 0, 0] = 1.0
        P[0, 1, 0] = 1.0
        P[0, 2, 0] = 1.0
        P[0, 3, 0] = 1.0
        P[0, 4, 0] = 1.0


        #Action 1: Go forward
        P[1, 0, 1] = 1.0
        P[1, 1, 2] = 1.0
        P[1, 2, 3] = 1.0
        P[1, 3, 4] = 1.0
        P[1, 4, 0] = 1.0

        #Action 2 risk jump
        P[2, 0, 2] = 0.6
        P[2, 0, 0] = 0.4

        P[2, 1, 3] = 0.6
        P[2, 1, 0] = 0.4

        P[2, 2, 4] = 0.6
        P[2, 2, 0] = 0.4

        P[2, 3, 4] = 0.6
        P[2, 3, 0] = 0.4

        P[2, 4, 0] = 1.0
        return P

    def get_R(self, t, mu):
        R = np.zeros((self.observation_space.n,self.action_space.n))


        R[0] = 1
        R[1] = 0
        R[2] = 0
        R[3] = 0
        R[4] = 100 *(1-mu[4])

        return R

    def final_R(self,mu):
        R = np.zeros(self.observation_space.n)

        R[0] = 1
        R[1] = 0
        R[2] = 0
        R[3] = 0
        R[4] = 100 *(1-mu[4])

        return R