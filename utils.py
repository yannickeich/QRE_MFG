import numpy as np


def get_action_probs_from_Qs(Qs):
    """ For Q tables in N x ... x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    return b.reshape(Qs.shape).mean(0)


def get_new_action_probs_from_Qs(num_averages_yet, old_probs, Qs):
    """ For Q tables in N x ... x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    new_probs = b.reshape(Qs.shape).mean(0)
    return (old_probs * num_averages_yet + new_probs) / (num_averages_yet + 1)


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


def get_softmax_action_probs_from_Qs(Qs, temperature=1.0):
    """ For Q tables in N x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    a = a - a.max(1, keepdims=True)
    b = np.exp(a / temperature)
    b = b / (np.sum(b, axis=1, keepdims=True))
    return b.reshape(Qs.shape).mean(0)


def get_softmax_new_action_probs_from_Qs(num_averages_yet, old_probs, Qs, temperature=1.0):
    """ For Q tables in N x X x U where N the number of Q tables, compute the action probs X x U,
     i.e. max over last argument, and averaged over first argument """
    a = Qs.reshape((-1, Qs.shape[-1]))
    a = a - a.max(1, keepdims=True)
    b = np.exp(a / temperature)
    b = b / (np.sum(b, axis=1, keepdims=True))
    new_probs = b.reshape(Qs.shape).mean(0)
    return (old_probs * num_averages_yet + new_probs) / (num_averages_yet + 1)


def value_based_forward(env, V_br):
    mus = []
    curr_mf = env.mu_0
    mus.append(curr_mf)
    for t in range(env.time_steps):
        P_t = env.get_P(t, mus[t])
        Q_t = env.get_R(t, mus[t])
        if t < env.time_steps-1:
            Q_t += np.einsum('ijk,k->ji', P_t, V_br[t+1])
        a = Q_t.reshape((-1, Q_t.shape[-1]))
        action_probs = np.zeros_like(a)
        action_probs[np.arange(len(a)), a.argmax(1)] = 1
        #action_probs = action_probs.reshape(Q_t.shape).mean(0)
        xu = np.expand_dims(curr_mf, axis=(1,)) * action_probs
        curr_mf = np.einsum('ijk,ji->k', P_t, xu)
        mus.append(curr_mf)

    return np.array(mus)

def value_based_softmax_forward(env, V_br,temperature):
    mus = []
    curr_mf = env.mu_0
    mus.append(curr_mf)
    for t in range(env.time_steps):
        P_t = env.get_P(t, mus[t])
        Q_t = env.get_R(t, mus[t])
        if t < env.time_steps-1:
            Q_t += np.einsum('ijk,k->ji', P_t, V_br[t+1])

        a = Q_t.reshape((-1, Q_t.shape[-1]))
        a = a - a.max(1, keepdims=True)
        b = np.exp(a / temperature)
        b = b / (np.sum(b, axis=1, keepdims=True))
        #b=b.reshape(Q_t.shape).mean(0)
        action_probs = b

        xu = np.expand_dims(curr_mf, axis=(1,)) * action_probs
        curr_mf = np.einsum('ijk,ji->k', P_t, xu)
        mus.append(curr_mf)

    return np.array(mus)