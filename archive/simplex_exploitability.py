import numpy as np
import fnmatch
import args_parser
import os
from env.RPS import RPS
from env.fast_marl import FastMARLEnv
from utils import get_softmax_action_probs_from_Qs, get_action_probs_from_Qs, get_curr_mf, find_best_response, \
    eval_curr_reward,find_soft_response, get_softmax_new_action_probs_from_Qs, get_new_action_probs_from_Qs


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

env = RPS()
game= 'RPS'
### Load results of RE with low temperature ->> Approx Nash equilibria
n_actions = 3
variants = ["RE_fp", "QRE_fp", "BE_fp"]
fp_iterations = 1000
# same configs as in experiment
temperature_list = np.exp(np.linspace(-0.5, 4.5, 50))

#Load action probs
config = args_parser.generate_config_from_kw(game=game, variant=variants[0],temperature=temperature_list[0],fp_iterations=fp_iterations)
files = find('action_probs.npy', config['exp_dir'])
nash_action_probs = np.load(files[0])




# Number of points along each axis
n_points = 50
# Generate coordinates within the 3D simplex
x = np.linspace(0, 1, n_points)
y = np.linspace(0, 1, n_points)
# Create a meshgrid of coordinates
xx, yy = np.meshgrid(x, y)
# Flatten the coordinates to obtain all combinations
xy_coordinates = np.vstack([xx.flatten(), yy.flatten()]).T
z_coordinate = 1-xy_coordinates.sum(1)
points = np.concatenate((xy_coordinates,z_coordinate[...,None]),axis = 1)
# Filter out points where z is negative
valid_points = points[points[:, 2] >= 0]

for action_probs in valid_points:
    #Exchange first action by grid points
    nash_action_probs[0] = action_probs
    ### Compute Exploitability

    mus = get_curr_mf(env, nash_action_probs)

    """ Evaluate current policy """
    V_pi, Q_pi = eval_curr_reward(env, nash_action_probs, mus)

    """ Evaluate current best response against current average policy """
    Q_br = find_best_response(env, mus)
    v_1 = np.vdot(env.mu_0, Q_br.max(axis=-1)[0])
    v_curr_1 = np.vdot(env.mu_0, V_pi)

    """ Exploitability """
    print(f"{config['exp_dir']} : expl: {v_1 - v_curr_1}, ... br achieves {v_1} vs. {v_curr_1}")

print('debug')
