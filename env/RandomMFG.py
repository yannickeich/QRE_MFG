import numpy as np
from gym.spaces import Discrete
from env.fast_marl import FastMARLEnv

np.random.seed(95510066)
class RandomMFG(FastMARLEnv):
    """
    Models the Left Right game.
    """

    def __init__(self, num_states: int = 100, num_actions: int = 10,time_steps: int = 10,mu_0=None,
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
        if mu_0 is None:
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

        if len(mu.shape)==1:
            return self.P
        # if mu is an ensemble return an extra dimension
        elif len(mu.shape)==2:
            return self.P[None]
        else:
            raise ValueError

    def get_R(self, t, mu):
        R = self.R.copy()
        if mu.ndim ==1:
            R -= np.log(mu)[...,None]
        elif mu.ndim == 2:
            R= R[None]- np.log(mu)[..., None]
        else:
            raise ValueError
        return R

    # TODO:  For now, dont use final rewards, because they are not included in lookahead q functions
    # def final_R(self,mu):
    #     R = np.zeros(self.observation_space.n)
    #     R -= np.log(mu)
    #     return R