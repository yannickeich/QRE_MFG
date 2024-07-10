import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class riskRPS(FastMARLEnv):
    """
    Models the Rock Paper Scissor game.
    """

    def __init__(self, time_steps: int = 2, mu_0 = np.array([1.,0.,0.,0.,0.]),
                 num_agents: int = 100, **kwargs):

        # Initial state and a state for rock, paper, scissors each.
        observation_space = Discrete(5)
        action_space = Discrete(3)

        mu_0 = np.array([1.,0.,0.,0.,0.])

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
        # Choose Paper
        P[1, 0, 2] = 1.0
        # Choose Scissor
        P[2, 0, 3] = 1.0

        #Already decided -> Paper and Scissor stay, Rock has 50/50 to move in lose state
        #Not dependent on action
        P[:, 1, 1] = 0.5
        P[:, 1, 4] = 0.5
        P[:, 2, 2] = 1.0
        P[:, 3, 3] = 1.0



        return P

    def get_R(self, t, mu):
        R = np.zeros((self.observation_space.n,self.action_space.n))

        # only gets a final reward
        return R

    def final_R(self,mu):
        R = np.zeros(self.observation_space.n)

        # Rock loses to Paper and wins against scissor
        R[1] = - mu[2] + 3*mu[3]
        # Paper loses to Scissor and wins against rock
        R[2] = - mu[3] + (mu[1] + mu[4])
        # Scissor loses to Rock and wins against paper
        R[3] = - (mu[1]+mu[4]) + mu[2]

        #Lose state rock: loses to Paper but does not win
        R[4] = -mu[2]

        return R