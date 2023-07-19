import itertools

import numpy as np

from matplotlib import pyplot as plt, pylab as pl, cycler

import args_parser
from env.fast_marl import FastMARLEnv
from utils import get_curr_mf


def run_once(env, action_probs):
    """ Plot mean field of states (for debugging) """
    xs = env.reset()
    mf_state = [np.mean(xs == i) for i in range(env.observation_space.n)]
    mf_states = [mf_state]
    xss = [xs]
    uss = []
    val = 0
    for t in range(env.time_steps):
        cum_ps = np.cumsum(action_probs[t, :,][xs], axis=-1)
        actions = np.zeros((env.num_agents,), dtype=int)
        uniform_samples = np.random.uniform(0, 1, size=env.num_agents)
        for idx in range(env.observation_space.n):
            actions += idx * np.logical_and(uniform_samples >= (cum_ps[:, idx - 1] if idx - 1 >= 0 else 0.0),
                                                   uniform_samples < cum_ps[:, idx])

        xs, rewards, done, info = env.step(actions)
        val += rewards

        xss.append(xs)
        uss.append(actions)

        mf_state = [np.mean(xs == i) for i in range(env.observation_space.n)]
        mf_states.append(mf_state)

    return val, np.array(mf_states), xss


if __name__ == '__main__':
    config = args_parser.parse_config()
    env: FastMARLEnv = config['game'](**config, num_agents=300)

    action_probs = np.load(config['exp_dir'] + f"action_probs.npy")
    best_response = np.load(config['exp_dir'] + f"best_response.npy")

    plt.figure()
    plt.ylabel("MF $\mu$")
    plt.ylim([0, 1])
    clist = itertools.cycle(cycler(color='rbgcmyk'))
    linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))

    """ Plot limiting MFs """
    mus = get_curr_mf(env, action_probs)
    color = clist.__next__()['color']
    linestyle = linestyle_cycler.__next__()['linestyle']
    plt.plot(range(env.time_steps + 1), mus[:, 1], color=color, label=f"$\mu(I)$", linestyle=linestyle, linewidth=2)

    """ Plot empirical N-agent MFs """
    val, mf_states, xss = run_once(env, action_probs)
    color = clist.__next__()['color']
    linestyle = linestyle_cycler.__next__()['linestyle']
    plt.plot(range(env.time_steps + 1), mf_states[:, 1], color=color, label=f"$\hat \mu(I)$", linestyle=linestyle, linewidth=2)

    plt.legend()
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    from pathlib import Path
    Path(f"./figures/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./figures/%s_%s_%d_%d_%f_%d.png" % (config['game'].__name__, config['variant'], config['fp_iterations'],
                  config['inf'], config['temperature'], config['softmax']), bbox_inches='tight', transparent=True, pad_inches=0)
