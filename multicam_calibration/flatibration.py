"""
The goal of flatibration is to compute a 3D rigid transform to a coordinate system where
the XY plane corresponds to the floor of the recording arena. This code implements the
following pipeline:

Input:
  - option 1: 3D keypoints from animals in the recording arena
  - option 2: 3D keypoints corresponding to the floor of the recording arena

Output:
    3D rigid transform that maps from the coordinate system of the input keypoints to 
    a coordinate system where the XY plane is the floor of the recording arena. 
    
Algorithm:
    Suppose z=ax+by+t is the equation of the plane identified by RANSAC. This means
    that vectors (1,0,a) and (0,1,b) are parallel to the floor plane, (a, b, -1) is 
    orthogonal to the floor plane. The output transformation is uniquely determined by
    the following criteria
      - the origin is contained within the floor plane
      - the vector (0,0,0)->(1,0,a) maps to the X-axis
      - the vector (0,0,0)->(0,1,b) maps to the Y-axis
      - the vector (0,0,0)->(-a, -b, 1) maps to the Z-axis

Optional steps:
    - Flip the z-axis to point in the opposite direction.
    - Move the center of the arena to the origin.
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor

from .geometry import (
    rigid_transform_from_correspondences,
    get_transformation_matrix,
    get_transformation_vector,
    apply_rigid_transform,
)


def get_floor_points(keypoints, z_points_down=False):
    """
    Get the 3D keypoints that correspond to the floor of the recording arena.

    Parameters
    ----------
    keypoints : np.array of shape (n_frames, n_keypoints, 3) or a list of np.arrays
        3D keypoints of the animal(s) in the recording arena. The z-axis should be
        roughly vertical.

    z_points_down : bool, default=False
        True if the z-axis points down, False if the z-axis points up.
    """
    if isinstance(keypoints, list):
        keypoints = np.concatenate(keypoints)

    if z_points_down:
        ix = np.argmax(keypoints[:, :, 2], axis=1)
    else:
        ix = np.argmin(keypoints[:, :, 2], axis=1)
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
    source_origin = np.array([0, 0, ransac.estimator_.intercept_])

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


def center_arena(transform, floor_points, center_method="midrange", range_pctl=1):
    """
    Compose the input rigid transform with a translation that moves the center of the
    arena to the origin. This function assumes that the input transform maps the floor
    points to the XY plane, hence translation is restricted to the XY plane.

    Parameters
    ----------
    transform : np.array of shape (6,)
        Rigid transform in vector format. The first three elements specify a rotation in
        axis-angle form and the last three elements specify a translation.

    floor_points : np.array of shape (n_points, 3) or a list of np.arrays
        3D keypoints that correspond to the floor of the recording arena.

    center_method : str, default='midrange'
        Method to compute the center of the arena.
        - ``midrange``: average of robust min and max, computed using ``range_pctl``.
        - ``mean``: average of the floor points.
        - ``median``: median of the floor points.

    range_pctl : float, default=1
        Percentile to compute the robust min and max of the floor points, used only if
        ``center_method`` is 'midrange'.

    Returns
    -------
    transform_centered : np.array of shape (6,)
        Rigid transform in vector format. The first three elements specify a rotation in
        axis-angle form and the last three elements specify a translation.
    """
    if isinstance(floor_points, list):
        floor_points = np.concatenate(floor_points)

    # apply the input transform to the floor points
    pts = apply_rigid_transform(transform, floor_points)[:, :2]

    # compute the center of the arena
    if center_method == "midrange":
        center = np.percentile(pts, [range_pctl, 100 - range_pctl], axis=0).mean(axis=0)
    elif center_method == "mean":
        center = np.mean(pts, axis=0)
    elif center_method == "median":
        center = np.median(pts, axis=0)
    else:
        raise ValueError("center_method should be 'midrange', 'mean', or 'median'")

    # compose input transform with a translation that moves center to the origin
    translation = np.array([0, 0, 0, -center[0], -center[1], 0])
    T = get_transformation_matrix(translation) @ get_transformation_matrix(transform)
    transform_centered = get_transformation_vector(T)

    return transform_centered
