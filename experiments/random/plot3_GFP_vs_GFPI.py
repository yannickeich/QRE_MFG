import fnmatch
import itertools
import os
import plot_configurations
import string

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

    skip_n=1

    # Same settings as in experiment
    games = ['random',]
    variants = ["QRE"]
    methods = ["FPI","pFP"]
    temperatures = [0.01,1.0]
    iterations = 10000

    #Default settings:
    lookahead = False
    tau = 5

    for game in games:
        clist = itertools.cycle(cycler('color',['purple','blue','red','green']))
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-','--']))
        fig,(ax1) = plt.subplots(1,1,sharex=True)
        fig.subplots_adjust(hspace=0.01)
        ax1.annotate('(' + string.ascii_lowercase[0] + ')',
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
            color = clist.__next__()['color']
            for variant in variants:
                for temperature in temperatures:
                    linestyle = linestyle_cycler.__next__()['linestyle']
                    config = args_parser.generate_config_from_kw(game=game,method=method, variant=variant,temperature=temperature, softmax=True,fp_iterations=iterations,lookahead=lookahead,tau=tau)
                    files = find('stdout', config['exp_dir'])


                    plot_vals = []
                    with open(max(files, key=os.path.getctime), 'r') as fi:
                        fi_lines = fi.readlines()
                        for line in fi_lines[:]:
                            fields = line.split(" ")
                            for i, field in enumerate(fields):
                                if field == 'QRE_l1_distance:':
                                    # Save number without comma
                                    plot_vals.append(float(fields[i+1][:-1]))

                    ax1.loglog(range(len(plot_vals))[::skip_n], plot_vals[::skip_n], linestyle, color=color,
                             label=variant)



        ax1.grid('on')
        plt.xlabel(r'Iterations $k$')

        ax1.set_ylabel(r'$\Delta \mathrm{QRE}(\pi)$')


        plt.xlim([0, len(plot_vals)-1])

        #ax2.set_yscale('symlog')
        #ax1.set_yscale('symlog')
        ax1.set_xscale('symlog')

    """ Finalize plot """
    plt.gcf().set_size_inches(3.25, 1.5)
    plt.tight_layout(w_pad=0.0,h_pad=0.2)
    plt.savefig(f'./figures/exp3.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    #plt.savefig(f'./figures/exploitability_without_legend.pdf', bbox_inches = 'tight', transparent = True, pad_inches = 0)

    #plt.savefig(f'./figures/exploitability.png', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
