import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv

np.random.seed(1002)
class RandomMFG(FastMARLEnv):
    """
    Models the Left Right game.
    """

    def __init__(self, num_states: int = 10, num_actions: int = 3,time_steps: int = 10,
                 num_agents: int = 100, **kwargs):

        self.num_states = num_states
        self.num_actions = num_actions


        #Create P:
        self.P = np.random.rand(num_actions,num_states,num_states)
        self.P /= self.P.sum(axis=-1, keepdims=True)

        #Create mu independent reward:
        self.R = np.random.rand(num_states,num_actions)

        observation_space = Discrete(num_states)
        action_space = Discrete(num_actions)

        mu_0 = np.ones(num_states)/num_states

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

        return self.P

    def get_R(self, t, mu):
        R = self.R.copy()
        R += -np.log(mu)[:,None]
        return R

    def final_R(self,mu):
        R = np.zeros(self.observation_space.n)
        return R