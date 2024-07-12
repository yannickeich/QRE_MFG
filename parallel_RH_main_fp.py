import subprocess
import numpy as np
import args_parser
from env.fast_marl import FastMARLEnv
from utils import (get_softmax_action_probs_from_Qs, get_action_probs_from_Qs, get_curr_mf, find_best_response, \
    eval_curr_reward,find_soft_response, get_softmax_new_action_probs_from_Qs, get_new_action_probs_from_Qs)
from utils import get_curr_mf_p, eval_curr_reward_p, find_best_response_p, find_soft_response_p

if __name__ == '__main__':
    config = args_parser.parse_config(mf_method="pRH")
    tau = config['tau']
    env: FastMARLEnv = config['game'](**config)

    #Get initial condition and horizon from the environment
    mu_0 = env.mu_0
    total_horizon = env.time_steps


    number_mfgs = total_horizon-tau+1
    #Set the horizon of the environment to the receding horizon lookahead
    env.time_steps = tau
    #Add another dimension for parallel computing. Index p stands for parallel variable
    mu_p = np.zeros((number_mfgs,tau+1,env.observation_space.n))
    mu_p[:,0,:] = mu_0
    #env.mu_0 = mu_final[i]
    #env.time_steps = tau
    Q_0_p = np.zeros((number_mfgs,tau, env.observation_space.n, env.action_space.n))
    action_probs_p= get_action_probs_from_Qs(Q_0_p)

    # For FP
    sum_action_probs_p = np.zeros_like(action_probs_p)

    # mus_avg_p = []
    # for i in range(number_mfgs):
    #     mus_avg_p.append(get_curr_mf(env, action_probs_p[i]))

    #parallel version instead of for loop
    initial_mu = mu_p[:, 0, :]
    mus_avg_p = get_curr_mf_p(env,initial_mu,action_probs_p)


    beta = 0.95
    with open(config['exp_dir'] + f"stdout", "w", buffering=1) as fo:
        for iteration in range(config['fp_iterations']):
                if config['method'] == 'pFP':
                    #FP method where the policy gets averaged.
                    sum_action_probs_p  =  sum_action_probs_p * beta + (1-beta)*action_probs_p
                    action_probs_p = sum_action_probs_p/sum_action_probs_p.sum(-1)[...,None]
                    action_probs_p[np.isnan(action_probs_p)] = 1 / env.action_space.n

                # mus_p = []
                # for i in range(number_mfgs):
                #     mus_p.append(get_curr_mf(env, action_probs_p[i]))
                mus_p = get_curr_mf_p(env,initial_mu,action_probs_p)

                """Evaluation"""
                # IF FP: we have to compare the average policy against the best response to the meanfield induced by the average policy
                # the average policy, that leads to the average mean field, can only be computed, when the dynamics do not depend on the mean field
                if config['method'] == "FP":
                    sum_action_probs_p = sum_action_probs_p + action_probs_p * np.array(mus_p)[:,:-1][..., None]
                    action_probs_avg_p = sum_action_probs_p/sum_action_probs_p.sum(-1)[...,None]
                    action_probs_avg_p[np.isnan(action_probs_avg_p)] = 1 / env.action_space.n

                    action_probs_compare_p = action_probs_avg_p.copy()
                    # mu_compare_p = []
                    # for i in range(total_horizon-tau+1):
                    #     mu_compare_p.append(get_curr_mf(env, action_probs_compare_p[i]))
                    mu_compare_p  = get_curr_mf_p(env,initial_mu,action_probs_compare_p)

                elif config['method'] == "expFPv2":
                    if iteration ==0:
                        sum_action_probs_p = action_probs_p * np.array(mus_p)[:-1][..., None]
                    else:
                        sum_action_probs_p = sum_action_probs_p * beta + (1-beta)*action_probs_p * np.array(mus_p)[:-1][..., None]
                    action_probs_avg_p = sum_action_probs_p / sum_action_probs_p.sum(-1)[..., None]
                    action_probs_avg_p[np.isnan(action_probs_avg_p)] = 1 / env.action_space.n
                    action_probs_compare_p = action_probs_avg_p.copy()
                    # mu_compare_p = []
                    # for i in range(total_horizon-tau+1):
                    #     mu_compare_p.append(get_curr_mf(env, action_probs_compare_p[0]))
                    mu_compare_p = get_curr_mf_p(env,initial_mu,action_probs_compare_p)

                else:
                    action_probs_compare_p = action_probs_p.copy()
                    mu_compare_p = mus_p.copy()

                #Compute Value functions to get the response policies with respect to the induced meanfield
                V_pi_p, Q_pi_p = eval_curr_reward_p(env,action_probs_compare_p,mu_compare_p)
                Q_br_p= find_best_response_p(env, mu_compare_p)
                Q_sr_p = find_soft_response_p(env,mu_compare_p,temperature=config['temperature'])
                v_1_p = (initial_mu * Q_br_p.max(axis=-1)[:,0]).sum(-1)
                v_curr_1_p = (initial_mu*V_pi_p).sum(-1)

                # V_pi_p=[]
                # Q_pi_p=[]
                # Q_br_p=[]
                # Q_sr_p = []
                # v_1_p=[]
                # v_curr_1_p=[]
                # for i in range(number_mfgs):
                #     """ Evaluate current policy """
                #     V_pi, Q_pi = eval_curr_reward(env, action_probs_compare_p[i], mu_compare_p[i])
                #     V_pi_p.append(V_pi)
                #     Q_pi_p.append(Q_pi)
                #     """ Evaluate current best response against current average policy """
                #     Q_br = find_best_response(env, mu_compare_p[i])
                #     Q_br_p.append(Q_br)
                #     v_1 = np.vdot(env.mu_0, Q_br.max(axis=-1)[0])
                #     v_curr_1 = np.vdot(env.mu_0, V_pi)
                #     v_1_p.append(v_1)
                #     v_curr_1_p.append(v_curr_1)
                #     Q_sr_p.append(find_soft_response(env, mus_p[i], temperature=config['temperature']))

                BE_action_probs_p = get_softmax_action_probs_from_Qs(Q_br_p,
                                                                         temperature=config['temperature'])
                QRE_action_probs_p = get_softmax_action_probs_from_Qs(Q_pi_p,
                                                                      temperature=config['temperature'])
                RE_action_probs_p = get_softmax_action_probs_from_Qs(Q_sr_p,
                                                                     temperature=config['temperature'])

                if iteration % 10 ==0:
                    for i in range(number_mfgs):
                        """ Exploitability """
                        print(f"{config['exp_dir']} game {i} iteration {iteration}: expl: {v_1_p[i] - v_curr_1_p[i]}, ... br achieves {v_1_p[i]} vs. {v_curr_1_p[i]}")
                        fo.write(f"{config['exp_dir']} game{i} iteration {iteration}: expl: {v_1_p[i] - v_curr_1_p[i]}, ... br achieves {v_1_p[i]} vs. {v_curr_1_p[i]}")
                        fo.write('\n')

                        """Boltzmann L1-Distance """
                        print(f"{config['exp_dir']} game {i} iteration {iteration}: BE_l1_distance: {np.abs(BE_action_probs_p[i] - action_probs_compare_p[i]).sum(-1).sum(-1).max()}")
                        fo.write(f"{config['exp_dir']} game {i} iteration {iteration}: BE_l1_distance: {np.abs(BE_action_probs_p[i] - action_probs_compare_p[i]).sum(-1).sum(-1).max()}")
                        fo.write('\n')

                        """QRE L1-Distance"""
                        print(f"{config['exp_dir']} game {i} iteration {iteration}: QRE_l1_distance: {np.abs(QRE_action_probs_p[i] - action_probs_compare_p[i]).sum(-1).sum(-1).max()}")
                        fo.write(f"{config['exp_dir']} game {i} iteration {iteration}: QRE_l1_distance: {np.abs(QRE_action_probs_p[i] - action_probs_compare_p[i]).sum(-1).sum(-1).max()}")
                        fo.write('\n')

                        """Relative Entropy L1-Distance"""
                        print(f"{config['exp_dir']} game {i} iteration {iteration}: RE_l1_distance: {np.abs(RE_action_probs_p[i] - action_probs_compare_p[i]).sum(-1).sum(-1).max()}")
                        fo.write(f"{config['exp_dir']} game {i} iteration {iteration}: RE_l1_distance: {np.abs(RE_action_probs_p[i] - action_probs_compare_p[i]).sum(-1).sum(-1).max()}")
                        fo.write("\n")

                ### Average mean_field for FP methods
                if config['method']=='FP':
                    mus_avg_p = (iteration * np.array(mus_avg_p) + np.array(mus_p)) / (iteration + 1)
                elif (config['method']=='expFPv1')|(config['method']=='expFPv2'):
                    mus_avg_p = beta * np.array(mus_avg_p) + (1 - beta) * np.array(mus_p)

                #Nash Equilibrium
                if config['variant'] == 'NE':
                    #If FP method compute best response wtr mus_avg
                    if (config['method'] == 'FP') | (config['method']=='expFPv1')|(config['method']=='expFPv2'):
                            Q_br_p = find_best_response_p(env, mus_avg_p)
                    action_probs_p = get_action_probs_from_Qs(Q_br_p)

                elif config['variant'] == 'QRE':
                    #If FP method compute Q_pi wtr mus_avg
                    if (config['method'] == 'FP') | (config['method']=='expFPv1')|(config['method']=='expFPv2'):
                            V_pi_p, Q_pi_p = eval_curr_reward_p(env, action_probs_p, mus_avg_p)
                    action_probs_p = get_softmax_action_probs_from_Qs(Q_pi_p, temperature=config['temperature'])

                elif config['variant'] == 'BE':
                    #If FP method compute best response wtr mus_avg
                    if (config['method'] == 'FP') |(config['method']=='expFPv1')|(config['method']=='expFPv2'):
                            Q_br_p = find_best_response_p(env, mus_avg_p)
                    action_probs_p = get_softmax_action_probs_from_Qs(Q_br_p, temperature=config['temperature'])

                elif config['variant'] == "RE":
                    #If FP method compute soft response wtr mus_avg
                    if (config['method'] == 'FP') |(config['method']=='expFPv1')|(config['method']=='expFPv2'):
                            Q_sr_p = find_soft_response(env,mus_avg_p,temperature=config['temperature'])
                    action_probs_p = get_softmax_action_probs_from_Qs(Q_sr_p, temperature=config['temperature'])

                else:
                    raise NotImplementedError

                # Change the initial mu of each MFG to the second time step mu of the previous MF solution
                initial_mu[1:] = mu_compare_p[:-1, 1]


    action_probs_final = np.zeros((total_horizon, env.observation_space.n, env.action_space.n))
    mu_final = np.zeros((total_horizon+1, env.observation_space.n))
    mu_final[0] = mu_0
    # np.save(config['exp_dir'] + f"action_probs.npy", action_probs_compare)
    # np.save(config['exp_dir'] + f"best_response.npy", Q_br)
    # np.save(config['exp_dir'] + f"mean_field.npy", mu_compare)
