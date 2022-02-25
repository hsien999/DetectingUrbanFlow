import numpy as np
from matplotlib.axes import Axes


def split_worker(iter_size: int, worker: int):
    """
    Slice a sequence of size 'iter_size' into 'worker' segments
    """
    assert iter_size >= 0 and worker >= 0
    split = []
    batch = int(iter_size / worker)
    if batch < 1:
        worker = min(iter_size, worker)
        batch = 1
    for i in range(worker):
        st, ed = i * batch, (i + 1) * batch
        if i == worker - 1:
            ed = iter_size
        split.append((st, ed))
    return split


def set_axes_equal_2d(ax: Axes):
    """
    Make axes of 2D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 2D.
    """
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim(origin[0] - radius, origin[0] + radius)
    ax.set_ylim(origin[1] - radius, origin[1] + radius)
