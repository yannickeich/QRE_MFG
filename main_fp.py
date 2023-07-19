import numpy as np

import args_parser
from env.fast_marl import FastMARLEnv
from utils import get_softmax_action_probs_from_Qs, get_action_probs_from_Qs, get_curr_mf, find_best_response, \
    eval_curr_reward, get_softmax_new_action_probs_from_Qs, get_new_action_probs_from_Qs

if __name__ == '__main__':
    config = args_parser.parse_config()
    env: FastMARLEnv = config['game'](**config)

    Q_0 = [np.zeros((env.time_steps, env.observation_space.n, env.action_space.n))]

    if config['softmax']:
        action_probs = get_softmax_action_probs_from_Qs(np.array(Q_0), temperature=config['temperature'])
    else:
        action_probs = get_action_probs_from_Qs(np.array(Q_0))

    if config['variant'] == "omd":
        mus = get_curr_mf(env, action_probs)
        y = 0 * eval_curr_reward(env, action_probs, mus)[1]

    """ Compute the MFG fixed point for all high degree agents """
    with open(config['exp_dir'] + f"stdout", "w", buffering=1) as fo:
        for iteration in range(config['fp_iterations']):
            mus = get_curr_mf(env, action_probs)

            if config['variant'] == "omd":
                Q_pi = eval_curr_reward(env, action_probs, mus)[1]
            if config['variant'] == "off-omd":
                Q_pi = eval_curr_reward(env, action_probs, mus)[1]
            Q_br = find_best_response(env, mus)

            """ Evaluate current best response against current average policy """
            v_1 = np.vdot(env.mu_0, Q_br.max(axis=-1)[0])
            v_curr_1 = np.vdot(env.mu_0, eval_curr_reward(env, action_probs, mus)[0])

            print(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write('\n')

            if config['variant'] == "fpi":
                if config['softmax']:
                    action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])
                else:
                    action_probs = get_action_probs_from_Qs(np.array([Q_br]))
            elif config['variant'] == "fp":
                if config['softmax']:
                    action_probs = get_softmax_new_action_probs_from_Qs(iteration + 1, action_probs, np.array([Q_br]), temperature=config['temperature'])
                else:
                    action_probs = get_new_action_probs_from_Qs(iteration + 1, action_probs, np.array([Q_br]))
            elif config['variant'] == "omd":
                y += config['temperature'] * Q_pi
                action_probs = get_softmax_action_probs_from_Qs(np.array([y]), temperature=1)

            np.save(config['exp_dir'] + f"action_probs.npy", action_probs)
            np.save(config['exp_dir'] + f"best_response.npy", Q_br)
