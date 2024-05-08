import numpy as np

import args_parser
from env.fast_marl import FastMARLEnv
from utils import get_softmax_action_probs_from_Qs, get_action_probs_from_Qs, get_curr_mf, find_best_response, \
    eval_curr_reward,find_soft_response, get_softmax_new_action_probs_from_Qs, get_new_action_probs_from_Qs, find_best_lookahead_response, eval_curr_reward_lookahead, find_soft_lookahead_response

if __name__ == '__main__':
    config = args_parser.parse_config()
    env: FastMARLEnv = config['game'](**config)
    tau = config['tau']
    #Initial
    Q_0 = [np.zeros((env.time_steps, env.observation_space.n, env.action_space.n))]
    action_probs = get_action_probs_from_Qs(np.array(Q_0))

    #For FP
    sum_action_probs = np.zeros_like(action_probs)
    mus_avg = get_curr_mf(env, action_probs)

    #For OMD
    y = np.zeros((env.time_steps, env.observation_space.n, env.action_space.n))

    beta = 0.95

    """ Compute the MFG fixed point for all high degree agents """
    with open(config['exp_dir'] + f"stdout", "w", buffering=1) as fo:
        for iteration in range(config['fp_iterations']):
            mus = get_curr_mf(env, action_probs)

            """Evaluation"""
            # IF FP: we have to compare the average policy against the best response to the meanfield induced by the average policy
            if config['method'] == "FP":
                sum_action_probs = (iteration * sum_action_probs + action_probs * mus[:-1][..., None])/(iteration+1)
                action_probs_avg = sum_action_probs/sum_action_probs.sum(-1)[...,None]
                action_probs_avg[np.isnan(action_probs_avg)] = 1 / env.action_space.n
                action_probs_compare = action_probs_avg.copy()
                mu_compare = get_curr_mf(env, action_probs_compare)

            else:
                action_probs_compare = action_probs.copy()
                mu_compare = mus.copy()

            """ Evaluate current policy """
            V_pi, Q_pi = eval_curr_reward(env, action_probs_compare, mu_compare)
            Q_pi_lookahead = eval_curr_reward_lookahead(env,action_probs,mu_compare,lookahead=5)
            """ Evaluate current best response against current average policy """
            #Q_br_lookahead = find_best_lookahaead_response(env,mu_compare,lookahead = 5)
            Q_br = find_best_response(env, mu_compare)
            v_1 = np.vdot(env.mu_0, Q_br.max(axis=-1)[0])
            v_curr_1 = np.vdot(env.mu_0, V_pi)

            """ Exploitability """
            print(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write(f"{config['exp_dir']} {iteration}: expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")
            fo.write('\n')

            """ Compare Policies """
            """Boltzmann L1-Distance """
            BE_action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])
            print(f"{config['exp_dir']} {iteration}: BE_l1_distance: {np.abs(BE_action_probs - action_probs_compare).sum(-1).sum(-1).max()}")
            fo.write(f"{config['exp_dir']} {iteration}: BE_l1_distance: {np.abs(BE_action_probs - action_probs_compare).sum(-1).sum(-1).max()}")
            fo.write('\n')

            """QRE L1-Distance"""
            QRE_action_probs = get_softmax_action_probs_from_Qs(np.array([Q_pi]), temperature=config['temperature'])
            print(f"{config['exp_dir']} {iteration}: QRE_l1_distance: {np.abs(QRE_action_probs - action_probs_compare).sum(-1).sum(-1).max()}")
            fo.write(f"{config['exp_dir']} {iteration}: QRE_l1_distance: {np.abs(QRE_action_probs - action_probs_compare).sum(-1).sum(-1).max()}")
            fo.write('\n')

            """Relative Entropy L1-Distance"""
            Q_sr = find_soft_response(env, mus, temperature=config['temperature'])
            RE_action_probs = get_softmax_action_probs_from_Qs(np.array([Q_sr]), temperature=config['temperature'])
            print(f"{config['exp_dir']} {iteration}: RE_l1_distance: {np.abs(RE_action_probs - action_probs_compare).sum(-1).sum(-1).max()}")
            fo.write(f"{config['exp_dir']} {iteration}: RE_l1_distance: {np.abs(RE_action_probs - action_probs_compare).sum(-1).sum(-1).max()}")
            fo.write("\n")

            ### Average mean_field for FP methods
            if config['method']=='FP':
                mus_avg = (iteration * mus_avg + mus) / (iteration + 1)
            elif config['method']=='expFP':
                mus_avg = beta * mus_avg + (1 - beta) * mus

            #Nash Equilibrium
            if config['variant'] == 'NE':
                #If FP method compute best response wtr mus_avg
                if (config['method'] == 'FP') | (config['method'] == 'expFP'):
                    if config['lookahead']:
                        Q_br = find_best_lookahead_response(env, mus_avg,tau)
                    else:
                        Q_br = find_best_response(env, mus_avg)
                else:
                    if config['lookahead']:
                        Q_br = find_best_lookahead_response(env, mus, tau)
                action_probs = get_action_probs_from_Qs(np.array([Q_br]))

            elif config['variant'] == 'QRE':
                #If FP method compute Q_pi wtr mus_avg
                if (config['method'] == 'FP') | (config['method'] == 'expFP'):
                    if config['lookahead']:
                        V_pi, Q_pi = eval_curr_reward_lookahead(env, action_probs, mus_avg,tau)
                    else:
                        V_pi, Q_pi = eval_curr_reward(env, action_probs, mus_avg)
                else:
                    if config['lookahead']:
                        V_pi, Q_pi = eval_curr_reward_lookahead(env, action_probs, mus, tau)
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_pi]), temperature=config['temperature'])

            elif config['variant'] == 'BE':
                #If FP method compute best response wtr mus_avg
                if (config['method'] == 'FP') | (config['method'] == 'expFP'):
                    if config['lookahead']:
                        Q_br = find_best_lookahead_response(env, mus_avg,tau)
                    else:
                        Q_br = find_best_response(env, mus_avg)
                else:
                    if config['lookahead']:
                        Q_br = find_best_lookahead_response(env,mus,tau)
                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_br]), temperature=config['temperature'])

            elif config['variant'] == "RE_fp":
                #If FP method compute soft response wtr mus_avg
                if (config['method'] == 'FP') | (config['method'] == 'expFP'):
                    if config['lookahead']:
                        Q_sr = find_soft_lookahead_response(env, mus_avg,tau, temperature=config['temperature'])
                    else:
                        Q_sr = find_soft_response(env,mus_avg,temperature=config['temperature'])
                else:
                    if config['lookahead']:
                        Q_sr = find_soft_lookahead_response(env, mus,tau, temperature=config['temperature'])
                    else:
                        Q_sr = find_soft_response(env, mus, temperature=config['temperature'])

                action_probs = get_softmax_action_probs_from_Qs(np.array([Q_sr]), temperature=config['temperature'])

            else:
                raise NotImplementedError

            np.save(config['exp_dir'] + f"action_probs.npy", action_probs)
            np.save(config['exp_dir'] + f"best_response.npy", Q_br)
            np.save(config['exp_dir'] + f"mean_field.npy", mus)
