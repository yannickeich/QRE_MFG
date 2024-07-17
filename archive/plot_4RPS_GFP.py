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


    i = 1
    skip_n = 1

    # Same settings as in experiment
    games = ['RPS',]
    variants = ['NE_fpi',"QRE_fp","expQRE_fp","RE_fp","expRE_fp","BE_fp","expBE_fp","NE_fp","expNE_fp"]
    stationary = False
    temperature = 1.0
    iterations = 10000

    for game in games:
        # clist = itertools.cycle(cycler('color',['purple','blue','red','green']))
        # linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
        fig.subplots_adjust(hspace=0.25)
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

        ax2.annotate('(' + string.ascii_lowercase[1] + ')',
                         (0, 0),
                         xytext=(10, +32),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')
        ax3.annotate('(' + string.ascii_lowercase[2] + ')',
                         (0, 0),
                         xytext=(10, +32),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')

        ax4.annotate('(' + string.ascii_lowercase[3] + ')',
                         (0, 0),
                         xytext=(10, +32),
                         xycoords='axes fraction',
                         textcoords='offset points',
                         fontweight='bold',
                         color='black',
                         alpha=0.7,
                         backgroundcolor='white',
                         ha='left', va='top')
        for variant in variants:
            plot_vals = []
            if variant[0]=='e':
                linestyle = '-'
            else:
                linestyle='--'

            config = args_parser.generate_config_from_kw(game=game, variant=variant,temperature=temperature, softmax=True,fp_iterations=iterations)
            files = find('stdout', config['exp_dir'])

            if (variant == "NE_fp")|(variant == "expNE_fp"):

                with open(max(files, key=os.path.getctime), 'r') as fi:
                    fi_lines = fi.readlines()
                    for line in fi_lines[:]:
                        fields = line.split(" ")
                        if fields[2] == 'expl:':
                            # Save number without comma
                            plot_vals.append(float(fields[3][:-1]))


                ax1.loglog(range(len(plot_vals))[::skip_n], plot_vals[::skip_n], linestyle, color='purple',
                             label=variant)
            if (variant == "QRE_fp")|(variant == "expQRE_fp"):
                plot_vals = []
                with open(max(files, key=os.path.getctime), 'r') as fi:
                    fi_lines = fi.readlines()
                    for line in fi_lines[:]:
                        fields = line.split(" ")
                        if fields[2] == 'QRE_l1_distance:':
                            # Save number without comma
                            plot_vals.append(float(fields[3][:-1]))

                ax2.loglog(range(len(plot_vals))[::skip_n], plot_vals[::skip_n], linestyle, color='blue',
                         label=variant)
            if (variant == "RE_fp")|(variant == "expRE_fp"):
                plot_vals = []
                with open(max(files, key=os.path.getctime), 'r') as fi:
                    fi_lines = fi.readlines()
                    for line in fi_lines[:]:
                        fields = line.split(" ")
                        if fields[2] == 'RE_l1_distance:':
                            # Save number without comma
                            plot_vals.append(float(fields[3][:-1]))

                ax3.loglog(range(len(plot_vals))[::skip_n], plot_vals[::skip_n], linestyle, color='red',
                           label=variant)

            if  (variant == "BE_fp")|(variant == "expBE_fp"):
                plot_vals = []
                with open(max(files, key=os.path.getctime), 'r') as fi:
                    fi_lines = fi.readlines()
                    for line in fi_lines[:]:
                        fields = line.split(" ")
                        if fields[2] == 'BE_l1_distance:':
                            # Save number without comma
                            plot_vals.append(float(fields[3][:-1]))

                ax4.loglog(range(len(plot_vals))[::skip_n], plot_vals[::skip_n], linestyle, color='green',
                           label=variant)

        ax1.grid('on')
        ax2.grid('on')
        ax3.grid('on')
        ax4.grid('on')
        plt.xlabel(r'Iterations $k$')
        ax1.set_ylabel(r'$\Delta J(\pi)$')
        ax2.set_ylabel(r'$\Delta \mathrm{QRE}(\pi)$')
        ax3.set_ylabel(r'$\Delta \mathrm{RE}(\pi)$')
        ax4.set_ylabel(r'$\Delta \mathrm{BE}(\pi)$')

        plt.xlim([0, len(plot_vals)-1])

        #ax2.set_yscale('symlog')
        #ax1.set_yscale('symlog')
        #ax1.set_xscale('symlog')

    """ Finalize plot """
    plt.gcf().set_size_inches(6.5, 5)
    plt.tight_layout(w_pad=0.0)
    #plt.savefig(f'./figures/exp+QRE+RE_small.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig(f'./figures/RPS_GFP.pdf', bbox_inches = 'tight', transparent = True, pad_inches = 0.5)

    #plt.savefig(f'./figures/exploitability.png', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
