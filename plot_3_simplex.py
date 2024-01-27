import fnmatch
import itertools
import os
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import args_parser

# def projectSimplex(points):
#     """
#     Project probabilities on the 3-simplex to a 2D triangle
#
#     N points are given as N x 3 array
#     """
#     # Convert points one at a time
#     tripts = np.zeros((points.shape[0], 2))
#     for idx in range(points.shape[0]):
#         # Init to triangle centroid
#         x = 1.0 / 2
#         y = 1.0 / (2 * np.sqrt(3))
#         # Vector 1 - bisect out of lower left vertex
#         p1 = points[idx, 0]
#         x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
#         y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
#         # Vector 2 - bisect out of lower right vertex
#         p2 = points[idx, 1]
#         x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
#         y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)
#         # Vector 3 - bisect out of top vertex
#         p3 = points[idx, 2]
#         y = y + (1.0 / np.sqrt(3) * p3)
#
#         tripts[idx, :] = (x, y)
#
#     return tripts


def projectSimplex_vector(points):
    """
    Project probabilities on the 3-simplex to a 2D triangle

    N points are given as N x 3 array
    """
    # Convert points one at a time
    tripts = np.zeros((*points.shape[:-1], 2))

    # Init to triangle centroid
    x = 1.0 / 2
    y = 1.0 / (2 * np.sqrt(3))
    # Vector 1 - bisect out of lower left vertex
    p1 = points[..., 0]
    x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
    y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
    # Vector 2 - bisect out of lower right vertex
    p2 = points[..., 1]
    x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
    y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)
    # Vector 3 - bisect out of top vertex
    p3 = points[..., 2]
    y = y + (1.0 / np.sqrt(3) * p3)

    tripts[..., :] = np.concatenate((x[...,None],y[...,None]),axis=-1)

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

    games = ['RPS',]
    n_actions = 3
    variants = ["RE_fp","QRE_fp","BE_fp"]

    # same configs as in experiment
    temperature_list  = np.exp(np.linspace(-0.5,4.5,50))
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
        simplex = matplotlib.lines.Line2D([0, 0.5, 1.0, 0], [0, np.sqrt(3 / 4), 0, 0],color='black',linewidth = 0.5)  # xcoords, ycoords
        subplot.add_line(simplex)
        subplot.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        subplot.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

        # Leave some buffer around the triangle for vertex labels
        subplot.set_xlim(0.0, 1.0)
        subplot.set_ylim(0.0, 1.0)
        subplot.set_aspect('equal')

        #norm = plt.Normalize(temperature_list[0],temperature_list[-1])

        # Create a second plot (inset) that zooms in on the interesting area
        #axins = subplot.inset_axes([1.0,0.0,0.4,0.4],aspect='equal')
        # x_lim = [0.35,0.55]
        # y_lim = [0.4,0.6]
        # axins.set_xlim(x_lim)  # Adjust the x-axis limits for the zoomed area
        # axins.set_ylim(y_lim)# Adjust the y-axis limits for the zoomed area
        #
        # # Add a box around the zoomed-in area
        # box = matplotlib.patches.Rectangle((x_lim[0], y_lim[0]), x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], linewidth=0.5, edgecolor='black', facecolor='none')
        # subplot.add_patch(box)
        # axins.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        # axins.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        #
        # connector_line = matplotlib.lines.Line2D(
        #     [x_lim[1] + subplot.get_xlim()[1], axins.get_xlim()[1]],
        #     [np.mean(y_lim) + subplot.get_ylim()[0], np.mean(y_lim)],
        #     color='black', linestyle='--'
        # )
        # subplot.add_line(connector_line)

        plot_values = np.zeros((len(variants),len(temperature_list), n_actions))
        for i,variant in enumerate(variants):


            for j, temperature in enumerate(temperature_list):
                config = args_parser.generate_config_from_kw(game=game, variant=variant,temperature=temperature,fp_iterations=fp_iterations)
                files = find('action_probs.npy', config['exp_dir'])
                action_probs = np.load(files[0])
                plot_values[i,j] = action_probs[0,0]



            color = clist.__next__()['color']
            linestyle = linestyle_cycler.__next__()['linestyle']
            marker = marker_cycler.__next__()['marker']

            # Project points to 2d simplex and draw
            projected = projectSimplex_vector(plot_values)

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
            subplot.plot(projected[i,:, 0], projected[i,:, 1],linestyle=linestyle,color=color,label= variant,alpha=0.5,linewidth=2)

            #axins.plot(projected[:, 0], projected[:, 1],linestyle=linestyle,color=color,alpha=0.5,linewidth=2)
        #plt.gcf().colorbar(line,ax=subplot)
        #lgd1 = subplot.legend(loc="lower right")
        # # plt.title(game + " " + variant)
        subplot.plot(projected[:,15,0],projected[:,15,1],color='green')
        subplot.plot(projected[:, 21, 0], projected[:, 21, 1],color='blue') # 5
        subplot.plot(projected[:, 25, 0], projected[:, 25, 1],color = 'red')
        subplot.plot(projected[:, 43, 0], projected[:, 43, 1],color='black') # 50
        #plt.xlabel(u"\u270C")
        #plt.xlabel(r'Temperature $\alpha$', fontsize=22)
        # plt.ylabel(r'action probs', fontsize=22)
        # plt.xlim([0, len(plot_vals)-1])
        # plt.xscale('symlog')
        # # plt.yscale('symlog')
        subplot.spines['top'].set_visible(False)
        subplot.spines['right'].set_visible(False)
        subplot.spines['bottom'].set_visible(False)
        subplot.spines['left'].set_visible(False)

    """ Finalize plot """
    #plt.gcf().set_size_inches(10, 8)
    plt.tight_layout(w_pad=0.0)
    plt.savefig(f'./figures/exploitability.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.savefig(f'./figures/exploitability.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()



