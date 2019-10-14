from pymol import cmd
from glob import glob
from os.path import join


def plot_grid(folder):
    for filename in glob(join(folder, '*.mol')):
        cmd.load(filename)

    cmd.set('grid_mode', 1)


cmd.extend(plot_grid.__name__, plot_grid)
