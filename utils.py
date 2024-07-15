import numpy as np


def get_action_probs_from_Qs(Qs):
    """ For Q tables in N x ... x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    return b.reshape(Qs.shape)


def get_new_action_probs_from_Qs(num_averages_yet, old_probs, Qs):
    """ For Q tables in N x ... x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    new_probs = b.reshape(Qs.shape)
    return (old_probs * num_averages_yet + new_probs) / (num_averages_yet + 1)


def find_best_lookahead_response(env, mus, lookahead):
    #TODO for now implemented without final reward!
    Qs = []

    for t in range(env.time_steps):
        V_t_next = np.zeros((env.observation_space.n,))
        for tau in range(lookahead).__reversed__():
            if t + tau < env.time_steps:
                P_t_tau = env.get_P(t+tau, mus[t+tau])
                Q_t_tau = env.get_R(t+tau, mus[t+tau]) + np.einsum('ijk,k->ji', P_t_tau, V_t_next)
                V_t_next = np.max(Q_t_tau,axis = -1)
        Qs.append(Q_t_tau)
    out_Qs = np.array(Qs)
    return out_Qs

def find_best_response(env, mus):
    Qs = []
    V_t_next = np.zeros((env.observation_space.n, ))
    V_t_next += env.final_R(mus[-1])
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_P(t, mus[t])
        Q_t = env.get_R(t, mus[t]) + np.einsum('ijk,k->ji', P_t, V_t_next)
        V_t_next = np.max(Q_t, axis=-1)
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return out_Qs

def find_best_response_p(env, mus):
    """
    Parallelized version of find_best_response
    """
    n_mfgs, time_steps, n_obs, n_actions = mus.shape[0], env.time_steps, env.observation_space.n, env.action_space.n
    Qs = np.zeros((n_mfgs,time_steps,n_obs,n_actions))
    # V_t_next = np.zeros((n_mfgs,env.observation_space.n ))
    # P_t = np.zeros((n_mfgs, n_actions, n_obs, n_obs))
    # R_t = np.zeros((n_mfgs, n_obs, n_actions))
    # for i in range(n_mfgs):
    #     V_t_next[i] += env.final_R(mus[i,-1])
    V_t_next = env.final_R(mus[:,-1])
    for t in range(env.time_steps).__reversed__():
        # for i in range(n_mfgs):
        #     P_t[i] = env.get_P(t, mus[i,t])
        #     R_t[i] = env.get_R(t, mus[i,t])
        P_t = env.get_P(t, mus[:,t])
        R_t = env.get_R(t, mus[:,t])
        Q_t =R_t + np.einsum('hijk,hk->hji', P_t, V_t_next)
        V_t_next = np.max(Q_t, axis=-1)
        Qs[:,t]=Q_t

    return Qs

def find_soft_lookahead_response(env, mus,lookahead, temperature=1.0):
    Qs = []
    for t in range(env.time_steps):
        V_t_next = np.zeros((env.observation_space.n,))
        for tau in range(lookahead).__reversed__():
            if t + tau < env.time_steps:
                P_t = env.get_P(t+tau, mus[t+tau])
                Q_t = env.get_R(t+tau, mus[t+tau]) + np.einsum('ijk,k->ji', P_t, V_t_next)
                V_t_next = Q_t.max(-1) + temperature * np.log(np.exp((Q_t-Q_t.max(-1)[...,None])/temperature).sum(-1))
        Qs.append(Q_t)
    out_Qs = np.array(Qs)
    return out_Qs

def find_soft_response(env, mus,temperature=1.0):
    Qs = []
    V_t_next = np.zeros((env.observation_space.n, ))
    V_t_next += env.final_R(mus[-1])
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_P(t, mus[t])
        Q_t = env.get_R(t, mus[t]) + np.einsum('ijk,k->ji', P_t, V_t_next)
        V_t_next = Q_t.max(-1) + temperature * np.log(np.exp((Q_t-Q_t.max(-1)[...,None])/temperature).sum(-1))
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return out_Qs

def find_soft_response_p(env, mus,temperature=1.0):
    """
    Parallel version of find_soft_response
    """
    n_mfgs, time_steps, n_obs, n_actions = mus.shape[0], env.time_steps, env.observation_space.n, env.action_space.n
    Qs = np.zeros((n_mfgs,time_steps,n_obs,n_actions))
    # V_t_next = np.zeros((n_mfgs,env.observation_space.n ))
    # P_t = np.zeros((n_mfgs, n_actions, n_obs, n_obs))
    # R_t = np.zeros((n_mfgs, n_obs, n_actions))
    # for i in range(n_mfgs):
    #     V_t_next[i] += env.final_R(mus[i,-1])
    V_t_next = env.final_R(mus[:, -1])
    for t in range(env.time_steps).__reversed__():
        # for i in range(n_mfgs):
        #     P_t[i] = env.get_P(t, mus[i, t])
        #     R_t[i] = env.get_R(t, mus[i, t])
        P_t = env.get_P(t, mus[:, t])
        R_t = env.get_R(t, mus[:, t])
        Q_t =R_t + np.einsum('hijk,hk->hji', P_t, V_t_next)
        V_t_next = Q_t.max(-1) + temperature * np.log(np.exp((Q_t-Q_t.max(-1)[...,None])/temperature).sum(-1))
        Qs[:,t] = Q_t

    return Qs

def get_curr_mf(env, action_probs):
    mus = []
    curr_mf = env.mu_0
    mus.append(curr_mf)
    for t in range(env.time_steps):
        P_t = env.get_P(t, mus[t])
        xu = np.expand_dims(curr_mf, axis=(1,)) * action_probs[t]
        curr_mf = np.einsum('ijk,ji->k', P_t, xu)
        mus.append(curr_mf)

    return np.array(mus)

def get_curr_mf_p(env,mus_0, action_probs):
    """
    Parallel version of get_curr_mf.
    Parallel over the first dimension
    """
    #TODO get rid of for loop for calling P_t. Needs to be changed in the environments.
    n_mfgs, time_steps, n_obs = mus_0.shape[0], env.time_steps, env.observation_space.n
    mus = np.zeros((n_mfgs,time_steps+1,n_obs))
    curr_mf = mus_0
    mus[:,0,:] = mus_0
    for t in range(time_steps):
        # P_t = []
        # for i in range(n_mfgs):
        #     P_t.append(env.get_P(t, mus[i,t]))
        # P_t = np.array(P_t)
        P_t = env.get_P(t,mus[:,t])
        xu = curr_mf[...,None] * action_probs[:,t]
        curr_mf = np.einsum('hijk,hji->hk', P_t, xu)
        mus[:, t+1, :] = curr_mf

    return mus




def eval_curr_reward_lookahead(env, action_probs, mus,lookahead):
    Qs = []
    for t in range(env.time_steps):
        V_t_next = np.zeros((env.observation_space.n, ))
        for tau in range(lookahead).__reversed__():
            if t + tau < env.time_steps:
                P_t = env.get_P(t+tau, mus[t+tau])
                Q_t = env.get_R(t+tau, mus[t+tau]) \
                      + np.einsum('ijk,k->ji', P_t, V_t_next)
                V_t_next = np.sum(action_probs[t+tau] * Q_t, axis=-1)

        Qs.append(Q_t)

    out_Qs = np.array(Qs)
    return out_Qs


def eval_curr_reward(env, action_probs, mus):
    Qs = []
    V_t_next = np.zeros((env.observation_space.n, ))
    V_t_next += env.final_R(mus[-1])
    for t in range(env.time_steps).__reversed__():
        P_t = env.get_P(t, mus[t])
        Q_t = env.get_R(t, mus[t]) \
              + np.einsum('ijk,k->ji', P_t, V_t_next)
        V_t_next = np.sum(action_probs[t] * Q_t, axis=-1)
        Qs.append(Q_t)

    Qs.reverse()
    out_Qs = np.array(Qs)
    return V_t_next, out_Qs

def eval_curr_reward_p(env, action_probs, mus):
    """
    Parallel version of eval_curr_reward
    """
    n_mfgs, time_steps, n_obs, n_actions = mus.shape[0], env.time_steps, env.observation_space.n, env.action_space.n
    Qs = np.zeros((n_mfgs,time_steps,n_obs,n_actions))
    #V_t_next = np.zeros((n_mfgs,env.observation_space.n ))
    # P_t = np.zeros((n_mfgs, n_actions, n_obs, n_obs))
    # R_t = np.zeros((n_mfgs, n_obs, n_actions))
    # for i in range(n_mfgs):
    #     V_t_next[i] += env.final_R(mus[i,-1])
    V_t_next = env.final_R(mus[:, -1])
    for t in range(env.time_steps).__reversed__():
        # for i in range(n_mfgs):
        #     P_t[i] = env.get_P(t, mus[i,t])
        #     R_t[i] = env.get_R(t, mus[i,t])
        P_t =  env.get_P(t, mus[:, t])
        R_t = env.get_R(t, mus[:, t])
        Q_t = R_t  + np.einsum('hijk,hk->hji', P_t, V_t_next)
        V_t_next = np.sum(action_probs[:, t] * Q_t, axis=-1)
        Qs[:,t] = Q_t

    return V_t_next, Qs


def get_softmax_action_probs_from_Qs(Qs, temperature=1.0):
    """ For Q tables in N x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, """
    a = Qs.reshape((-1, Qs.shape[-1]))
    a = a - a.max(1, keepdims=True)
    b = np.exp(a / temperature)
    b = b / (np.sum(b, axis=1, keepdims=True))
    return b.reshape(Qs.shape)


def get_softmax_new_action_probs_from_Qs(num_averages_yet, old_probs, Qs, temperature=1.0):
    """ For Q tables in N x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    a = a - a.max(1, keepdims=True)
    b = np.exp(a / temperature)
    b = b / (np.sum(b, axis=1, keepdims=True))
    new_probs = b.reshape(Qs.shape)
    return (old_probs * 0.9 + 0.1 * new_probs) / (1)

#
# def value_based_forward(env, V_br):
#     mus = []
#     curr_mf = env.mu_0
#     mus.append(curr_mf)
#     for t in range(env.time_steps):
#         P_t = env.get_P(t, mus[t])
#         Q_t = env.get_R(t, mus[t])
#
#         Q_t += np.einsum('ijk,k->ji', P_t, V_br[t+1])
#         a = Q_t.reshape((-1, Q_t.shape[-1]))
#         action_probs = np.zeros_like(a)
#         action_probs[np.arange(len(a)), a.argmax(1)] = 1
#         #action_probs = action_probs.reshape(Q_t.shape).mean(0)
#         xu = np.expand_dims(curr_mf, axis=(1,)) * action_probs
#         curr_mf = np.einsum('ijk,ji->k', P_t, xu)
#         mus.append(curr_mf)
#
#     return np.array(mus)
#
# def value_based_softmax_forward(env, V_br,temperature):
#     mus = []
#     curr_mf = env.mu_0
#     mus.append(curr_mf)
#     for t in range(env.time_steps):
#         P_t = env.get_P(t, mus[t])
#         Q_t = env.get_R(t, mus[t])
#
#         Q_t += np.einsum('ijk,k->ji', P_t, V_br[t+1])
#
#         a = Q_t.reshape((-1, Q_t.shape[-1]))
#         a = a - a.max(1, keepdims=True)
#         b = np.exp(a / temperature)
#         b = b / (np.sum(b, axis=1, keepdims=True))
#         #b=b.reshape(Q_t.shape).mean(0)
#         action_probs = b
#
#         xu = np.expand_dims(curr_mf, axis=(1,)) * action_probs
#         curr_mf = np.einsum('ijk,ji->k', P_t, xu)
#         mus.append(curr_mf)
#
#     return np.array(mus)