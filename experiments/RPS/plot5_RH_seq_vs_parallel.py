import fnmatch
import itertools
import os
import plot_configurations
import string
from env.fast_marl import FastMARLEnv
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from env.RPS import RPS
import args_parser


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def plot():

    skip_n=1

    # Same settings as in experiment
    mf_methods = ["RH","pRH"]
    games = ['RPS',]
    variants = ["QRE"]
    methods = ["pFP"]
    temperature = 0.5
    iterations = 10000
    #tau = np.range(2,10)
    tau = 5
    time_steps = 10

    #Default settings:
    lookahead = False


    n_mfgs = time_steps-tau+1

    for n_game in range(n_mfgs):
        for game in games:
            clist = itertools.cycle(cycler('color',['blue','red']))
            linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--']))
            fig,ax2 = plt.subplots(1,1,sharex=True)
            fig.subplots_adjust(hspace=0.01)
            # ax1.annotate('(' + string.ascii_lowercase[0] + ')',
            #                  (0, 0),
            #                  xytext=(10, +32),
            #                  xycoords='axes fraction',
            #                  textcoords='offset points',
            #                  fontweight='bold',
            #                  color='black',
            #                  alpha=0.7,
            #                  backgroundcolor='white',
            #                  ha='left', va='top')

            ax2.annotate('(' + str(n_game) + ')',
                             (0, 0),
                             xytext=(10, +32),
                             xycoords='axes fraction',
                             textcoords='offset points',
                             fontweight='bold',
                             color='black',
                             alpha=0.7,
                             backgroundcolor='white',
                             ha='left', va='top')

            for method in methods:
                for variant in variants:
                    for mf_method in mf_methods:
                        config = args_parser.generate_config_from_kw(game=game,method=method, variant=variant,temperature=temperature, softmax=True,fp_iterations=iterations,lookahead=lookahead,tau=tau,mf_method=mf_method)
                        files = find('stdout', config['exp_dir'])


                        plot_vals = []
                        with open(max(files, key=os.path.getctime), 'r') as fi:
                            fi_lines = fi.readlines()
                            for line in fi_lines[:]:
                                fields = line.split(" ")
                                #Check, which mean field game
                                if fields[2] == str(n_game):

                                    for i, field in enumerate(fields):
                                        if field == 'QRE_l1_distance:':
                                            # Save number without comma
                                            plot_vals.append(float(fields[i+1][:-1]))

                        color = clist.__next__()['color']
                        linestyle = linestyle_cycler.__next__()['linestyle']
                        ax2.loglog(range(len(plot_vals))[::skip_n], plot_vals[::skip_n], linestyle, color=color,
                                 label=variant)



            #ax1.grid('on')
            ax2.grid('on')
            plt.xlabel(r'Iterations $k$')

            #ax1.set_ylabel(r'$\Delta J(\pi)$')
            ax2.set_ylabel(r'$\Delta \mathrm{Q}^\pi{RE}(\pi)$')


            plt.xlim([0, len(plot_vals)-1])

            #ax2.set_yscale('symlog')
            #ax1.set_yscale('symlog')
            #ax1.set_xscale('symlog')

        """ Finalize plot """
        plt.gcf().set_size_inches(3.25, 2.8)
        plt.tight_layout(w_pad=0.0,h_pad=0.2)
        plt.savefig(f'./figures/exp5'+str(n_game)+'.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
        #plt.savefig(f'./figures/exploitability_without_legend.pdf', bbox_inches = 'tight', transparent = True, pad_inches = 0)

        #plt.savefig(f'./figures/exploitability.png', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()


if __name__ == '__main__':
    plot()
