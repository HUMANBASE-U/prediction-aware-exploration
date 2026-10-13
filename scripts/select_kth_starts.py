"""Suggest separated, navigable start poses in a KTH map component.

This is only for pre-registering paired test starts; it does not change maps.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import block_reduce


def select_starts(map_dir, first, count=3, min_clearance=5):
    free = block_reduce(np.load(str(map_dir / 'occ_map.npy')),
                        block_size=(2, 2), func=np.min, cval=0) == 254
    valid = block_reduce(np.load(str(map_dir / 'valid_space.npy')),
                         block_size=(2, 2), func=np.max, cval=0) > 0
    labels, _ = label(free)
    seed = np.asarray(first, dtype=int) - 500
    component = labels[tuple(seed)]
    if component == 0:
        raise ValueError('Initial pose is not in a valid free-space component')
    clearance = distance_transform_edt(free)
    candidates = np.argwhere((labels == component) & valid &
                             (clearance >= min_clearance))
    picks = [seed]
    for _ in range(count - 1):
        squared_distances = np.stack([
            np.sum((candidates - pick) ** 2, axis=1) for pick in picks])
        choice = candidates[np.argmax(np.min(squared_distances, axis=0))]
        picks.append(choice)
    return [tuple((pick + 500).tolist()) for pick in picks]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map_dir', type=Path)
    parser.add_argument('first_row', type=int)
    parser.add_argument('first_col', type=int)
    args = parser.parse_args()
    print(select_starts(args.map_dir, (args.first_row, args.first_col)))
