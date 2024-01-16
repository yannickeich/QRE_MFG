import numpy as np

import args_parser
from env.fast_marl import FastMARLEnv
from utils import get_softmax_action_probs_from_Qs, get_action_probs_from_Qs, get_curr_mf, find_best_response, \
    eval_curr_reward,find_soft_response, get_softmax_new_action_probs_from_Qs, get_new_action_probs_from_Qs

if __name__ == '__main__':
    config = args_parser.parse_config()
    env: FastMARLEnv = config['game'](**config)

    Q_0 = [np.zeros((env.time_steps, env.observation_space.n, env.action_space.n))]


    action_probs = get_action_probs_from_Qs(np.array(Q_0))
    mus_avg = get_curr_mf(env, action_probs)

    y = np.zeros((env.time_steps, env.observation_space.n, env.action_space.n))

    """ Compute the MFG fixed point for all high degree agents """
    with open(config['exp_dir'] + f"stdout", "w", buffering=1) as fo:
        for iteration in range(config['fp_iterations']):
            mus = get_curr_mf(env, action_probs)

            """ Evaluate current policy """
            V_pi, Q_pi = eval_curr_reward(env, action_probs, mus)

            """ Evaluate current best response against current average policy """
            """ Exploitability """
            Q_br = find_best_response(env, mus)
            v_1 = np.vdot(env.mu_0, Q_br.max(axis=-1)[0])
            v_curr_1 = np.vdot(env.mu_0, V_pi)


            print(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write('\n')

            """ Compare Policies """
            """Boltzmann L1-Distance """
            BE_action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])
            print(f"{config['exp_dir']} {iteration}: l1_distance: {np.abs(BE_action_probs - action_probs).sum(-1).sum(-1).max()}")
            fo.write(f"{config['exp_dir']} {iteration}: l1_distance: {np.abs(BE_action_probs - action_probs).sum(-1).sum(-1).max()}")
            fo.write('\n')
            """QRE L1-Distance"""
            QRE_action_probs = get_softmax_action_probs_from_Qs(np.array([Q_pi]), temperature=config['temperature'])
            print(f"{config['exp_dir']} {iteration}: l1_distance: {np.abs(QRE_action_probs - action_probs).sum(-1).sum(-1).max()}")
            """Relative Entropy L1-Distance"""
            Q_sr = find_soft_response(env, mus, temperature=config['temperature'])
            RE_action_probs = get_softmax_action_probs_from_Qs(np.array([Q_sr]), temperature=config['temperature'])
            print(f"{config['exp_dir']} {iteration}: l1_distance: {np.abs(RE_action_probs - action_probs).sum(-1).sum(-1).max()}")

            if config['variant'] == "BE_fpi":
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])
            elif config['variant'] == "QRE_fpi":
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_pi]), temperature=config['temperature'])
            elif config['variant'] == "RE_fpi":
                Q_sr = find_soft_response(env,mus,temperature=config['temperature'])
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_sr]), temperature=config['temperature'])
            elif config['variant'] == "NE_fpi":
                action_probs = get_action_probs_from_Qs(np.array([Q_br]))
            elif config['variant'] == "BE_fp":
                #mus_avg  = (iteration * mus_avg + mus)/(iteration+1)
                mus_avg = 0.95 * mus_avg + 0.05 * mus
                Q_br = find_best_response(env, mus_avg)
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])
            elif config['variant'] == "QRE_fp":
                #mus_avg  = (iteration * mus_avg + mus)/(iteration+1)
                mus_avg = 0.95 * mus_avg + 0.05 * mus
                V_pi, Q_pi = eval_curr_reward(env, action_probs, mus_avg)
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_pi]), temperature=config['temperature'])
            elif config['variant'] == "RE_fp":
                #mus_avg  = (iteration * mus_avg + mus)/(iteration+1)
                mus_avg = 0.95 * mus_avg + 0.05 * mus
                Q_sr = find_soft_response(env, mus_avg,temperature=config['temperature'])
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_sr]), temperature=config['temperature'])
            elif config['variant'] == "NE_fp":
                mus_avg = 0.9 * mus_avg + 0.1 * mus
                Q_br = find_best_response(env, mus_avg)
                action_probs = get_action_probs_from_Qs(np.array([Q_br]))
            elif config['variant'] == "BE_omd":
                #y = (iteration * y + Q_br)/(iteration+1)
                y = 0.9 * y + 0.1 *Q_br
                action_probs = get_softmax_action_probs_from_Qs(np.array([y]), temperature=config['temperature'])
            elif config['variant'] == "QRE_omd":
                #y = (iteration * y + Q_br)/(iteration+1)
                y = 0.9 * y + 0.1 * Q_pi
                action_probs = get_softmax_action_probs_from_Qs(np.array([y]), temperature=config['temperature'])
            elif config['variant'] == "NE_omd":
                y += Q_pi
                action_probs = get_softmax_action_probs_from_Qs(np.array([y]), temperature=config['temperature'])
            else:
                raise NotImplementedError

            np.save(config['exp_dir'] + f"action_probs.npy", action_probs)
            np.save(config['exp_dir'] + f"best_response.npy", Q_br)
            np.save(config['exp_dir'] + f"mean_field.npy", mus)
