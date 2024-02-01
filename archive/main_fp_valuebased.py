import numpy as np

import args_parser
from env.fast_marl import FastMARLEnv
from utils import get_softmax_action_probs_from_Qs, get_action_probs_from_Qs, get_curr_mf, find_best_response, \
    eval_curr_reward, get_softmax_new_action_probs_from_Qs, get_new_action_probs_from_Qs, value_based_forward, value_based_softmax_forward

if __name__ == '__main__':
    config = args_parser.parse_config()
    env: FastMARLEnv = config['game'](**config)

    Q_0 = [np.zeros((env.time_steps, env.observation_space.n, env.action_space.n))]
    V_0 = [np.zeros((env.time_steps,env.observation_space.n))]
    Q_br = Q_0[0].copy()
    if config['softmax']:
        action_probs = get_softmax_action_probs_from_Qs(np.array(Q_0), temperature=config['temperature'])
    else:
        action_probs = get_action_probs_from_Qs(np.array(Q_0))

    if config['variant'] == "omd":
        mus = get_curr_mf(env, action_probs)
        y = 0 * eval_curr_reward(env, action_probs, mus)[1]
    #intial mu
    mus = get_curr_mf(env, action_probs)
    """ Compute the MFG fixed point for all high degree agents """
    with open(config['exp_dir'] + f"stdout", "w", buffering=1) as fo:
        for iteration in range(config['fp_iterations']):
            if config['softmax']:
                mus = value_based_softmax_forward(env, np.concatenate((Q_br.max(axis=-1),env.final_R(mus[-1])[None])),temperature=config['temperature'])
            else:
                mus = value_based_forward(env, np.concatenate((Q_br.max(axis=-1),env.final_R(mus[-1])[None])))



            if config['variant'] == "omd":
                Q_pi = eval_curr_reward(env, action_probs, mus)[1]

            # # This is what the policy based on Q would do
            mus_test = get_curr_mf(env, action_probs)
            Q_br_test = find_best_response(env, mus_test)
            """ Evaluate current best response against current average policy """
            v_1 = np.vdot(env.mu_0, Q_br_test.max(axis=-1)[0])
            v_curr_1 = np.vdot(env.mu_0, eval_curr_reward(env, action_probs, mus_test)[0])

            print(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write('\n')

            Q_br = find_best_response(env, mus)

            if config['variant'] == "fpi":
                if config['softmax']:
                    action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])
                else:
                    action_probs = get_action_probs_from_Qs(np.array([Q_br]))
            elif config['variant'] == "fp":
                if config['softmax']:
                    action_probs = get_softmax_new_action_probs_from_Qs(iteration + 1, action_probs, np.array([Q_br]),
                                                                        temperature=config['temperature'])
                else:
                    action_probs = get_new_action_probs_from_Qs(iteration + 1, action_probs, np.array([Q_br]))
            elif config['variant'] == "omd":
                y += Q_pi
                action_probs = get_softmax_action_probs_from_Qs(np.array([y]), temperature=config['temperature'])

            np.save(config['exp_dir'] + f"action_probs.npy", action_probs)
            np.save(config['exp_dir'] + f"best_response.npy", Q_br)