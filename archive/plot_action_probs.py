import fnmatch
import itertools
import os
import string
import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler

import args_parser


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def plot():

    """ Plot figures """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 24,
        "font.sans-serif": ["Helvetica"],
    })

    i = 1
    skip_n = 1

    games = ['RPS',]
    n_actions = 3
    variants = ["BE_fp","RE_fp","QRE_fp"]
    # same configs as in experiment
    # np.linspace(0.5, 1.50, 11)
    temperature_list =np.linspace(0.2, 5, 20)
    fp_iterations = 1000
    #stationary = False

    for game in games:
        clist = itertools.cycle(cycler(color='rbgcmyk'))
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        subplot = plt.subplot(1, len(games), i)
        subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
                         (0, 0),
                         xytext=(10, +32),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')
        # subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        for variant in variants:
            plot_values = np.zeros((temperature_list.size,n_actions-1))
            i=0
            for temperature in temperature_list:
                config = args_parser.generate_config_from_kw(game=game, variant=variant,temperature=temperature,fp_iterations=fp_iterations)
                files = find('action_probs.npy', config['exp_dir'])
                action_probs = np.load(files[0])
                plot_values[i] = action_probs[0,0][:-1]
                i+=1
            color = clist.__next__()['color']
            linestyle = linestyle_cycler.__next__()['linestyle']
            subplot.plot(temperature_list, plot_values[::skip_n], linestyle, color=color,
                             label=variant, alpha=0.5, linewidth=2)

        lgd1 = plt.legend(loc="lower left")
        # # plt.title(game + " " + variant)
        plt.grid('on')
        plt.xlabel(r'Temperature $\alpha$', fontsize=22)
        plt.ylabel(r'action probs', fontsize=22)
        # plt.xlim([0, len(plot_vals)-1])
        # plt.xscale('symlog')
        # # plt.yscale('symlog')

    """ Finalize plot """
    plt.gcf().set_size_inches(12, 6.5)
    plt.tight_layout(w_pad=0.0)
    # plt.savefig(f'./figures/exploitability.pdf', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
    # plt.savefig(f'./figures/exploitability.png', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
