from pymol import cmd
from glob import glob
from os.path import join


def draw_grid(folder):
    for filename in glob(join(folder, '*.mol')):
        cmd.load(filename)

    cmd.set('grid_mode', 1)
    cmd.color('grey', 'elem C')


cmd.extend(draw_grid.__name__, draw_grid)
