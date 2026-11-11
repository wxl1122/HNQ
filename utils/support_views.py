"""Label-free geometric views of the same K normal reference images."""

import torch


def normal_reference_views(images, rotations=(0,), scales=(1.0,)):
    rotations = list(dict.fromkeys(rotations))
    scales = list(dict.fromkeys(scales))
    if not rotations or not scales:
        raise ValueError('Reference views require at least one rotation and scale')
    if any(angle not in (0, 90, 180, 270) for angle in rotations):
        raise ValueError('Reference rotations must be multiples of 90 degrees')
    if scales != [1.0]:
        raise ValueError('Only identity scale is supported in the release')
    if rotations == [0] and scales == [1.0]:
        return images
    return torch.cat([torch.rot90(images, angle // 90, dims=(-2, -1))
                      for angle in rotations], dim=1)
