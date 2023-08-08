import numpy as np
import cv2

na = np.newaxis


def rodrigues(r):
    """
    Convert a rotation vector to a rotation matrix.

    Parameters
    ----------
    r : array of shape (...,3)
        Rotation vector in axis-angle form.

    Returns
    -------
    R : array of shape (...,3,3)
        Rotation matrix.
    """
    A = np.zeros((*r.shape[:-1], 3, 3))
    A[..., 0, 1] = -r[..., 2]
    A[..., 0, 2] = r[..., 1]
    A[..., 1, 0] = r[..., 2]
    A[..., 1, 2] = -r[..., 0]
    A[..., 2, 0] = -r[..., 1]
    A[..., 2, 1] = r[..., 0]
    theta = np.linalg.norm(r, axis=-1, keepdims=True).reshape(
        *r.shape[:-1], 1, 1
    )
    A = A / np.where(theta == 0, 1, theta)
    R = np.sin(theta) * A + (1 - np.cos(theta)) * np.matmul(A, A)
    R[..., 0, 0] += 1
    R[..., 1, 1] += 1
    R[..., 2, 2] += 1
    return R


def rodrigues_inv(R):
    """
    Convert a rotation matrix to a rotation vector.

    Parameters
    ----------
    R : array of shape (...,3,3)
        Rotation matrix.

    Returns
    -------
    r : array of shape (...,3)
        Rotation vector in axis-angle form.
    """
    r = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    theta = np.arccos((np.trace(R, axis1=-2, axis2=-1) - 1) / 2).reshape(
        *R.shape[:-2], 1
    )
    rnorm = np.linalg.norm(r, axis=-1, keepdims=True)
    rnorm += rnorm == 0
    return r * theta / rnorm


def get_transformation_matrix(t):
    """
    Generate 4x4 transformation matrices from a 3D rotations and translations.

    Parameters
    ----------
    t : array of shape (...,6)
        Rigid transforms in vector format. The first three elements specify a
        rotation in axis-angle form and the last three elements specify a
        translation.

    Returns
    -------
    T : array of shape (...,4,4)
        Transformation matrices.
    """
    T = np.zeros((*t.shape[:-1], 4, 4))
    T[..., :3, :3] = rodrigues(t[..., :3])
    T[..., :3, 3] = t[..., 3:]
    T[..., 3, 3] = 1
    return T


def get_transformation_vector(T):
    """
    Convert 4x4 transformation matrices to vector format.

    This function inverts
    :py:func:`multicam_calibration.get_transformation_matrix`.

    Parameters
    ----------
    T : array of shape (...,4,4)
        Transformation matrices.

    Returns
    -------
    t : array of shape (...,6)
        Rigid transforms in vector format. The first three elements specify a
        rotation in axis-angle form and the last three elements specify a
        translation.
    """
    return np.concatenate(
        [rodrigues_inv(T[..., :3, :3]), T[..., :3, 3]], axis=-1
    )


def euclidean_to_homogenous(x_euclidean):
    """
    Convert Euclidean coordinates to homogenous coordinates

    ..math::

        (x_1,...,x_d) \mapsto (x_1,...,x_d, 1)

    Parameters
    ----------
    x_euclidean : array of shape (...,d)
        Euclidean coordinates.

    Returns
    -------
    x_homogenous : array of shape (...,d+1)
        Homogenous coordinates.
    """
    padding = np.ones((*x_euclidean.shape[:-1], 1))
    x_homogenous = np.concatenate((x_euclidean, padding), axis=-1)
    return x_homogenous


def homogeneous_to_euclidean(x_homogenous):
    """
    Convert homogenous coordinates to Euclidean coordinates

    ..math::

        (x_1,...,x_d, z) \mapsto (x_1/z,...,x_d/z)

    Parameters
    ----------
    x_homogenous : array of shape (...,d+1)
        Homogenous coordinates.

    Returns
    -------
    x_euclidean : array of shape (...,d)
        Euclidean coordinates.
    """
    x_euclidean = x_homogenous[..., :-1] / x_homogenous[..., -1:]
    return x_euclidean


def project_points(points, extrinsics, camera_matrix, dist_coefs):
    """
    Project 3D points onto the image plane.

    Parameters
    ----------
    points : array of shape (..., 3)
        3D points in the world coordinate system.

    extrinsics : array of shape (6,)
        The transformation from the world coordinate system to the camera's
        coordinate system. The first three elements are the rotation vector and
        the last three elements are the translation vector.

    camera_matrix : array of shape (3, 3)
        The camera matrix.

    dist_coefs : 1d array of length at least 2
        The radial distortion coefficients k1, k2.

    Returns
    -------
    uvs : array of shape (..., 2)
        The projected 2D points in the image plane.
    """
    # Rotate and translate points
    T = get_transformation_matrix(extrinsics)
    points = euclidean_to_homogenous(points)
    points_cam = np.matmul(T, points[..., na])[..., :3, 0]

    # Apply radial distortion
    k1, k2 = dist_coefs[:2]
    r2 = np.sum((points_cam[..., :2] / points_cam[..., 2:]) ** 2, axis=-1)
    radial_distortion = 1 + k1 * r2 + k2 * r2**2

    # Distorted points in camera coordinate system
    points_cam_distorted = points_cam * np.stack(
        (radial_distortion, radial_distortion, np.ones(points_cam.shape[:-1])),
        axis=-1,
    )

    # Project points to the image plane
    uvs = np.matmul(camera_matrix, points_cam_distorted[..., na]).squeeze(-1)
    uvs = uvs[..., :2] / uvs[..., 2:]
    return uvs


def undistort_points(uvs, camera_matrix, dist_coefs):
    """
    Wrapper for `cv2.undistortPoints` that handles NaNs and batch dimensions.

    Parameters
    ----------
    uvs : array of shape (..., 2)
        2D coordinates of points to undistort.

    camera_matrix : array of shape (3, 3)
        Camera matrix.

    dist_coefs : array of shape (5,)
        Distortion coefficients.

    Returns
    -------
    undistorted_uvs : array of shape (..., 2)
        Undistorted 2D coordinates.
    """
    uvs_shape = uvs.shape
    uvs = uvs.reshape(-1, 2)
    valid_ixs = ~np.isnan(uvs).any(-1)
    uvs_undistorted = np.zeros_like(uvs) * np.nan
    uvs_undistorted[valid_ixs] = cv2.undistortPoints(
        uvs[valid_ixs], camera_matrix, dist_coefs
    ).squeeze(1)
    return uvs_undistorted.reshape(uvs_shape)