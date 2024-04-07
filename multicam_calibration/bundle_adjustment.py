import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import warnings
from .geometry import *

na = np.newaxis


def embed_calib_objpoints(calib_objpoints, calib_poses):
    """
    Embed calibration object points into the world coordinate system.

    Parameters
    ----------
    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    calib_poses : array of shape (n_frames, 6)
        Poses of the calibration object in world coordinates.

    Returns
    -------
    calib_worldpoints : array of shape (n_frames, N, 3)
        Calibration object points in world coordinates.
    """
    pose_transforms = get_transformation_matrix(calib_poses)[:, na]
    objpoints = euclidean_to_homogenous(calib_objpoints)[na, :, :, na]
    calib_worldpoints = (pose_transforms @ objpoints)[..., :3, 0]
    return calib_worldpoints


def predict_calib_uvs(all_extrinsics, all_intrinsics, calib_objpoints, calib_poses):
    """
    Predict the (u,v) coordinates of calibration object points in each frame
    for each camera.

    Parameters
    ----------
    all_extrinsics : array of shape (n_cameras, 6)
        Transforms from world coordinates to each camera's coordinates.

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Camera intrinsics for each camera (see
        :py:func:`multicam_calibration.get_intrinsics`).

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    calib_poses : array of shape (n_frames, N, 3)
        Initial calibration object poses in each frame.

    Returns
    -------
    predicted_uvs : array of shape (n_cameras, n_frames, N, 2)
        Predicted (u,v) coordinates of calibration object points in each frame
        for each camera.
    """
    pts = embed_calib_objpoints(calib_objpoints, calib_poses)
    predicted_uvs = []
    for transform, intrinsics in zip(all_extrinsics, all_intrinsics):
        predicted_uvs.append(project_points(pts, transform, *intrinsics))
    return np.stack(predicted_uvs)


def residuals(params, all_calib_uvs, calib_objpoints):
    """
    Compute residuals between observed vs. expected (u,v) coordinates for a set
    of points based on their 3D positions and camera parameters.

    Parameters
    ----------
    params : array of shape (n_cameras * 12 + n_frames * 6)
        Camera parameters and calibration object poses in a single flat array.
        The camera parameters are `fx, fy, cx, cy, k1, k2, rx, ry, rz, tx, ty, tz`
        where `fx, fy` are focal lengths, `cx, cy` are principal points, `k1, k2`
        are radial distortion coefficients, `rx, ry, rz` define a 3D rotation in
        axis-angle form and `tx, ty, tz` define a translation. The calibration
        poses are rigid transforms (expressed in 6-dof vector form) from the
        calibration object's canonical coordinates to world coordinates.

    all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
        Calibration point detections for all the frames represented in `params`.
        NaNs are used to indicate missing detections.

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration object's points in its canonical
    """
    all_extrinsics, all_intrinsics, calib_poses = deserialize_params(
        params, all_calib_uvs.shape[0]
    )

    predicted_uvs = predict_calib_uvs(
        all_extrinsics, all_intrinsics, calib_objpoints, calib_poses
    )

    residuals = (all_calib_uvs - predicted_uvs)[~np.isnan(all_calib_uvs)]
    return residuals


def bundle_adjustment_sparsity(all_calib_uvs):
    """
    Create a sparse matrix representing the structure of the Jacobian matrix for
    bundle adjustment.

    Returns
    -------
    A : scipy.sparse.lil_matrix
        Sparse matrix representing the nonzero elements of the Jacobian matrix
        of the :py:func:`multicam_calibration.residuals` function.
    """
    n_cameras, n_frames, N, _ = all_calib_uvs.shape
    n_params = n_cameras * 12 + n_frames * 6

    mask = ~np.isnan(all_calib_uvs)
    camera_ixs = np.tile(np.arange(n_cameras)[:, na, na, na], (1, n_frames, N, 2))[mask]
    frame_ixs = np.tile(np.arange(n_frames)[na, :, na, na], (n_cameras, 1, N, 2))[mask]

    A = lil_matrix((mask.sum(), n_params), dtype=int)
    i = np.arange(mask.sum())
    for s in range(12):
        A[i, camera_ixs * 12 + s] = 1
    for s in range(6):
        A[i, n_cameras * 12 + frame_ixs * 6 + s] = 1
    return A


def serialize_params(all_extrinsics, all_intrinsics, calib_poses):
    """
    Serialize camera parameters and 3D calibration point positions into a flat
    array.

    Parameters
    ----------
    all_extrinsics : array of shape (n_camera, 6)
        Transforms from world coordinates to the coordinate system of each
        camera (see :py:func:`multicam_calibration.estimate_all_extrinsics`).

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Camera intrinsics for each camera (see
        :py:func:`multicam_calibration.get_intrinsics`).

    calib_poses : array of shape (n_frames, 6)
        Rigid transforms (expressed in 6-dof vector form) from the calibration
        object's canonical coordinates to world coordinates.
    """
    x0 = []
    for transform, (camera_matrix, dist_coefs) in zip(all_extrinsics, all_intrinsics):
        fx, fy, cx, cy = (
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2],
        )
        x0.append([fx, fy, cx, cy, *dist_coefs[:2], *transform])
    x0.append(calib_poses.flatten())
    return np.concatenate(x0)


def deserialize_params(x, n_cameras):
    """
    Deserialize camera parameters and calibration poses from a flat array
    (inverts :py:func:`multicam_calibration.serialize_params`).

    Parameters
    ----------
    x : array of shape (n_params,)
        Flat array of camera parameters and 3D calibration point positions.

    n_cameras : int
        Number of cameras.

    Returns
    -------
    all_extrinsics, all_intrinsics, calib_poses
        See :py:func:`multicam_calibration.serialize_params` for details.
    """
    all_extrinsics = []
    all_intrinsics = []
    for i in range(n_cameras):
        fx, fy, cx, cy = x[i * 12 : i * 12 + 4]
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = fx
        camera_matrix[1, 1] = fy
        camera_matrix[0, 2] = cx
        camera_matrix[1, 2] = cy
        dist_coefs = np.pad(x[i * 12 + 4 : i * 12 + 6], (0, 3))
        transform = x[i * 12 + 6 : i * 12 + 12]
        all_extrinsics.append(transform)
        all_intrinsics.append((camera_matrix, dist_coefs))
    calib_poses = x[n_cameras * 12 :].reshape((-1, 6))
    return np.array(all_extrinsics), all_intrinsics, calib_poses


def bundle_adjust(
    all_calib_uvs,
    all_extrinsics,
    all_intrinsics,
    calib_objpoints,
    calib_poses,
    n_frames=10000,
    outlier_threshold=None,
    **opt_kwargs,
):
    """
    Bundle adjustment for camera parameters and 3D points.

    Camera parameters (focal length, principal point, distortion coefficients)
    and calibration object poses are all optimized simultaneously to minimize
    reprojection error, defined as the sum of squared residuals between observed
    and predicted (u,v) coordinates of calibration points. Frames with < 2
    cameras observing the calibration object are ignored.

    Parameters
    ----------
    all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
        Calibration point detections for a number of frames for each camera.
        NaNs are used to indicate missing detections.

    all_extrinsics : array of shape (n_cameras, 6)
        Transforms from world coordinates to each camera's coordinates.

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Camera intrinsics for each camera (see
        :py:func:`multicam_calibration.get_intrinsics`).

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    calib_poses : array of shape (n_frames, N, 3)
        Initial calibration object poses in each frame.

    n_frames : int, default=1000
        Number of randomly-sampled frames to use for bundle adjustment.
        If `None`, all frames are used.

    outlier_threshold : float, default=None
        Frames will be excluded from the optimization if the mean reprojection
        in at least one camera exceeds this value. If `None`, the threshold is
        set to 5 times the median error.

    opt_kwargs : dict
        Additional keyword arguments to pass to `scipy.optimize.least_squares`.
        The following arguments are set by default::

            verbose=2, x_scale='jac', ftol=1e-4, method='trf', loss='soft_l1'

    Returns
    -------
    adjusted_extrinsics : array of shape (n_cameras, 6)
        Adjusted camera extrinsics.

    adjusted_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Adjusted camera intrinsics.

    adjusted_calib_poses : array of shape (n_frames, 6)
        Adjusted calibration object poses.

    use_frames : array of shape (n_frames,)
        Indexes of the frames used in bundle adjustment.

    result : scipy.optimize.OptimizeResult
        Result object returned by `scipy.optimize.least_squares`.
    """
    n_cameras, _, N, _ = all_calib_uvs.shape
    use_frames = np.nonzero((~np.isnan(all_calib_uvs).any((-1, -2))).sum(0) > 1)[0]

    # remove frames with high reprojection error
    predicted_uvs = predict_calib_uvs(
        all_extrinsics,
        all_intrinsics,
        calib_objpoints,
        calib_poses[use_frames],
    )

    err = np.linalg.norm(all_calib_uvs[:, use_frames] - predicted_uvs, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        worst_mean_err = np.nanmax(np.nanmean(err, axis=-1), axis=0)

    if outlier_threshold is None:
        outlier_threshold = 5 * np.nanmedian(err, axis=-1)

    exclude = np.nan_to_num(worst_mean_err) > outlier_threshold
    use_frames = use_frames[~exclude]

    print(
        f"Excluding {int(exclude.sum())} out of {len(use_frames)} frames "
        f"based on an outlier threshold of {outlier_threshold}"
    )

    # sample frames
    if n_frames is None or n_frames > len(use_frames):
        n_frames = len(use_frames)
    else:
        use_frames = np.random.choice(use_frames, n_frames, replace=False)

    # setup optimization
    A = bundle_adjustment_sparsity(all_calib_uvs[:, use_frames])
    x0 = serialize_params(all_extrinsics, all_intrinsics, calib_poses[use_frames])
    default_kwargs = dict(
        verbose=2, x_scale="jac", ftol=1e-4, method="trf", loss="soft_l1"
    )
    default_kwargs.update(opt_kwargs)

    # run optimization
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=A,
        **default_kwargs,
        args=(all_calib_uvs[:, use_frames], calib_objpoints),
    )

    (
        adjusted_extrinsics,
        adjusted_intrinsics,
        adjusted_calib_poses,
    ) = deserialize_params(result.x, n_cameras)

    return (
        adjusted_extrinsics,
        adjusted_intrinsics,
        adjusted_calib_poses,
        use_frames,
        result,
    )
