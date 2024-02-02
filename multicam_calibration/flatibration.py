"""
The goal of flatibration is to compute a 3D rigid transform to a coordinate system where
the XY plane corresponds to the floor of the recording arena. This code implements the
following pipeline:

Input:
  - option 1: 3D keypoints from animals in the recording arena
  - option 2: 3D keypoints corresponding to the floor of the recording arena

Output:
    3D rigid transform that maps from the coordinate system of the input keypoints to 
    a coordinate system where the XY plane is the floor of the recording arena

Algorithm:
    Suppose z=ax+by+t is the equation of the plane identified by RANSAC. This means
    that vectors (1,0,a) and (0,1,b) are parallel to the floor plane, (a, b, -1) is 
    orthogonal to the floor plane. The output transformation is uniquely determined by:
      - map the centroid of the floor points to the origin
      - map (1,0,a) to the X-axis
      - map (0,1,b) to the Y-axis
      - map (-a, -b, 1) to the Z-axis
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor

from .geometry import (
    rigid_transform_from_correspondences,
    get_transformation_matrix,
    get_transformation_vector,
)


def get_floor_points(keypoints, plus_z_is_up=True):
    """
    Get the 3D keypoints that correspond to the floor of the recording arena.

    Parameters
    ----------
    keypoints : np.array of shape (n_frames, n_keypoints, 3) or a list of np.arrays
        3D keypoints of the animal(s) in the recording arena. The z-axis should be
        roughly vertical.

    plus_z_is_up : bool
        True if the z-axis points up, False if the z-axis points down.
    """
    if isinstance(keypoints, list):
        keypoints = np.concatenate(keypoints)

    if plus_z_is_up:
        ix = np.argmin(keypoints[:, :, 2], axis=1)
    else:
        ix = np.argmax(keypoints[:, :, 2], axis=1)
    return keypoints[np.arange(keypoints.shape[0]), ix]


def flatibrate(floor_points, residual_threshold=10):
    """
    Compute a 3D rigid transform that maps the floor points to the XY plane and moves
    the centroid of the floor points to the origin.

    Parameters
    ----------
    floor_points : np.array of shape (n_points, 3) or a list of np.arrays
        3D keypoints that correspond to the floor of the recording arena.

    residual_threshold : float, default=10
        Maximum residual for a data point to be considered an inlier (passed to RANSAC)

    Returns
    -------
    transform : np.array of shape (6,)
        Rigid transform in vector format. The first three elements specify a rotation in
        axis-angle form and the last three elements specify a translation.
    """
    if isinstance(floor_points, list):
        floor_points = np.concatenate(floor_points)

    ransac = RANSACRegressor(residual_threshold=residual_threshold).fit(
        floor_points[:, :2], floor_points[:, 2]
    )
    a, b = ransac.estimator_.coef_

    source_x_axis = np.array([1, 0, a])
    source_y_axis = np.array([0, 1, b])
    source_z_axis = np.array([-a, -b, 1])
    source_origin = np.median(floor_points, axis=0)

    source_pts = np.array(
        [
            source_origin,
            source_origin + source_x_axis,
            source_origin + source_y_axis,
            source_origin + source_z_axis,
        ]
    )

    target_pts = np.array(
        [
            [0, 0, 0],
            [np.linalg.norm(source_x_axis), 0, 0],
            [0, np.linalg.norm(source_y_axis), 0],
            [0, 0, np.linalg.norm(source_z_axis)],
        ]
    )

    transform, _ = rigid_transform_from_correspondences(source_pts, target_pts)
    return transform


def flip_z_axis(transform):
    """
    Compose the input rigid transform with a 180 degree rotation around the X-axis.

    Parameters
    ----------
    transform : np.array of shape (6,)
        Rigid transform in vector format. The first three elements specify a rotation in
        axis-angle form and the last three elements specify a translation.

    Returns
    -------
    transform_flipped : np.array of shape (6,)
        Rigid transform in vector format. The first three elements specify a rotation in
        axis-angle form and the last three elements specify a translation.
    """
    T = get_transformation_matrix(transform)
    T_flipped = np.diag([1, -1, -1, 1]) @ T
    transform_flipped = get_transformation_vector(T_flipped)
    return transform_flipped
