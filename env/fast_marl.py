from abc import ABC, abstractmethod

import numpy as np


class FastMARLEnv(ABC):
    """
    Models a mean-field MARL problem in discrete time.
    """

    def __init__(self, observation_space, action_space, time_steps, mu_0, num_agents=100, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.time_steps = time_steps
        self.mu_0 = mu_0
        self.num_agents = num_agents

        super().__init__()

        self.xs = None
        self.t = None

    def reset(self):
        self.t = 0
        self.xs = self.sample_initial_states()
        return self.get_observation()

    def get_observation(self):
        return self.xs

    def sample_initial_states(self):
        return np.random.choice(range(len(self.mu_0)), size=(self.num_agents,), p=self.mu_0)

    def step(self, actions):
        next_xs = self.next_states(self.t, self.xs, actions)
        rewards = self.reward(self.t, self.xs, actions)

        self.t += 1
        self.xs = next_xs

        return self.get_observation(), rewards, self.t >= self.time_steps - 1, {}

    """
    Note that for fast execution, we vectorize and use the states and actions of all agents directly. 
     The implementing class makes sure that the next states and reward function follow the MFC model assumptions. """
    @abstractmethod
    def next_states(self, t, xs, us):
        pass  # sample new states for all agents

    @abstractmethod
    def reward(self, t, xs, us):
        pass  # sample reward defined on the state-action mean-field

    """ For dynamic programming solutions """
    @abstractmethod
    def get_P(self, t, mu):
        pass  # Return transition matrices U x X x X for given mf

    @abstractmethod
    def get_R(self, t, mu):
        pass  # Return array X x U of expected rewards for given mf
