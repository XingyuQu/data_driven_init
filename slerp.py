# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class SlerpTask(Task[torch.Tensor]):
        return self.gather_tensors.group_label()


class SlerpMerge(MergeMethod):
        )


def lerp(
    t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return maybe_torch(res, is_torch)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return maybe_torch(res, is_torch)


def maybe_torch(v: np.ndarray, is_torch: bool):
    if is_torch:
        return torch.from_numpy(v)
    return v


def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v