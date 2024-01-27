import fnmatch
import itertools
import os
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.collections import LineCollection

import args_parser

def projectSimplex(points):
    """
    Project probabilities on the 3-simplex to a 2D triangle

    N points are given as N x 3 array
    """
    # Convert points one at a time
    tripts = np.zeros((points.shape[0], 2))
    for idx in range(points.shape[0]):
        # Init to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * np.sqrt(3))
        # Vector 1 - bisect out of lower left vertex
        p1 = points[idx, 0]
        x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
        # Vector 2 - bisect out of lower right vertex
        p2 = points[idx, 1]
        x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)
        # Vector 3 - bisect out of top vertex
        p3 = points[idx, 2]
        y = y + (1.0 / np.sqrt(3) * p3)

        tripts[idx, :] = (x, y)

    return tripts


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

    games = ['A3_MDP',]
    n_actions = 3
    variants = ["BE_fp","RE_fp","QRE_fp"]

    # same configs as in experiment
    temperature_list  = [0.9,1.0,1.3,1.7,2.0,3.0,5.0,7.0,10.0,20.0,50.0,100.]
    fp_iterations = 1000
    #stationary = False

    for game in games:
        clist = itertools.cycle(cycler(color='rbgcmyk'))
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        marker_cycler = itertools.cycle(cycler('marker',['.','+','x']))
        subplot = plt.subplot(1, len(games), i)
        # subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
        #                  (0, 0),
        #                  xytext=(10, +32),
        #                  xycoords='axes fraction',
        #                  textcoords='offset points',
        #                  fontweight='bold',
        #                  color='black',
        #                  alpha=0.7,
        #                  backgroundcolor='white',
        #                  ha='left', va='top')
        # subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        i += 1

        # Draw the triangle
        simplex = matplotlib.lines.Line2D([0, 0.5, 1.0, 0], [0, np.sqrt(3 / 4), 0, 0])  # xcoords, ycoords
        subplot.add_line(simplex)
        subplot.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        subplot.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Leave some buffer around the triangle for vertex labels
        subplot.set_xlim(-0.2, np.sqrt(3 / 4) + 0.2)
        subplot.set_ylim(-0.2, 1.0)
        subplot.set_aspect('equal')

        norm = plt.Normalize(temperature_list[0],temperature_list[-1])




        for variant in variants:
            plot_values = np.zeros((len(temperature_list),n_actions))
            i=0

            for temperature in temperature_list:
                config = args_parser.generate_config_from_kw(game=game, variant=variant,temperature=temperature,fp_iterations=fp_iterations)
                files = find('action_probs.npy', config['exp_dir'])
                action_probs = np.load(files[0])
                plot_values[i] = action_probs[0,0]
                i+=1


            color = clist.__next__()['color']
            linestyle = linestyle_cycler.__next__()['linestyle']
            marker = marker_cycler.__next__()['marker']

            # Project points to 2d simplex and draw
            projected = projectSimplex(plot_values)

            #Scatter
            #subplot.scatter(projected[:, 0], projected[:, 1], c=np.log(temperature_list), cmap='viridis',marker=marker)

            ##Lines
            # segments = np.concatenate([projected[:-1][:, None, :], projected[1:][:, None, :]], axis=1)
            # lc = LineCollection(segments,cmap = 'magma')
            # lc.set_array(np.log(temperature_list))
            # lc.set_linestyle(linestyle)
            # lc.set_linewidth(2)
            # lc.set_label(variant)
            # line = subplot.add_collection(lc)
            subplot.plot(projected[:, 0], projected[:, 1],linestyle=linestyle,color=color,label= variant,alpha=0.5,linewidth=2)

        #plt.gcf().colorbar(line,ax=subplot)
        lgd1 = plt.legend(loc="lower right")
        # # plt.title(game + " " + variant)
        plt.grid('on')
        # plt.xlabel(r'Temperature $\alpha$', fontsize=22)
        # plt.ylabel(r'action probs', fontsize=22)
        # plt.xlim([0, len(plot_vals)-1])
        # plt.xscale('symlog')
        # # plt.yscale('symlog')

    """ Finalize plot """
    plt.gcf().set_size_inches(10, 8)
    #plt.tight_layout(w_pad=0.0)
    # plt.savefig(f'./figures/exploitability.pdf', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
    # plt.savefig(f'./figures/exploitability.png', bbox_extra_artists=(lgd1,), bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()



