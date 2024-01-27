import matplotlib.pyplot as plt
""" Plot figures """

fontsize = 9
params = {'axes.labelsize': fontsize,  # fontsize for x and y labels (was 10)
          'axes.titlesize': fontsize,
          'font.size': fontsize,  # was 10
          'legend.fontsize': fontsize,  # was 10
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'font.family': 'serif',
          'font.serif': ['Times New Roman'],
          'lines.linewidth': 0.5,  # was 2.5
          'axes.linewidth': 1,
          'axes.grid': False,
          'grid.linewidth': 0.5,
          'savefig.format': 'pdf',
          'axes.xmargin': 0,
          'axes.ymargin': 0.05,
          'savefig.pad_inches': 0,
          'legend.markerscale': 2,
          'savefig.bbox': 'tight',
          'lines.markersize': 4,
          'legend.columnspacing': 0.5,
          'legend.numpoints': 4,
          'legend.handlelength': 3.5,
          'text.usetex': True
          }
plt.rcParams.update(params)