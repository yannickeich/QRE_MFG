import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv


class SIS(FastMARLEnv):
    """
    Models the SIS game.
    """

    def __init__(self, infection_rate: float = 0.81, recovery_rate: float = 0.3, time_steps: int = 50,
                 initial_infection_prob: float = 0.1, cost_infection: float = 1, cost_action: float = 0.5,
                 num_agents: int = 100, **kwargs):
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infection_prob = initial_infection_prob
        self.cost_infection = cost_infection
        self.cost_action = cost_action

        observation_space = Discrete(2)
        action_space = Discrete(2)

        mu_0 = np.array([1 - initial_infection_prob, initial_infection_prob])

        super().__init__(observation_space, action_space,
                         time_steps, mu_0, num_agents=num_agents, **kwargs)

    def reset(self):
        self.t = 0
        self.xs = self.sample_initial_states()

        return self.get_observation()

    def next_states(self, t, xs, us):
        mu_infected = np.mean(xs)
        recoveries = np.random.rand(self.num_agents) < self.recovery_rate
        infections = np.random.rand(self.num_agents) < self.infection_rate * mu_infected
        new_xs = xs * (1 - recoveries) \
                 + (1 - xs) * (1 - us) * infections \
                 + (1 - xs) * us * 0

        return new_xs

    def reward(self, t, xs, us):
        rewards = - self.cost_infection * xs - self.cost_action * us
        return rewards

    def get_P(self, t, mu):
        P = np.zeros((self.action_space.n, self.observation_space.n, self.observation_space.n))

        P[:, 1, 0] = self.recovery_rate
        P[:, 1, 1] = 1 - P[:, 1, 0]
        P[0, 0, 1] = self.infection_rate * mu[1]
        P[0, 0, 0] = 1 - P[0, 0, 1]
        P[1, 0, 0] = 1

        return P

    def get_R(self, t, mu):
        R = np.zeros((self.observation_space.n, self.action_space.n))

        R[1, :] -= self.cost_infection
        R[:, 1] -= self.cost_action

        return R
