import numpy as np
import networkx as nx
import cv2
import tqdm
import warnings
from .geometry import *

na = np.newaxis


def get_intrinsics(
    calib_uvs,
    calib_objpoints,
    image_size,
    n_samples=100,
    fix_k3=True,
    zero_tangent_dist=True,
):
    """
    Get camera intrinsics from the (u,v) coordinates of calibration points
    detections in a set of images.

    Parameters
    ----------
    calib_uvs : array of shape (n_frames, N, 2)
        Calibration point detections for a number of frames.

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    image_size : tuple (width, height)
        Size of the images in pixels.

    n_samples : int, default=100
        Number of samples to use for calibration.

    fix_k3 : bool, default=True
        If True, the third radial distortion coefficient is fixed to zero.

    zero_tangent_dist : bool, default=True
        If True, the tangential distortion coefficients are fixed to zero.

    Returns
    -------
    camera_matrix : array of shape (3,3)
        Camera matrix, which has the form::

            | fx  0   cx |
            | 0   fy  cy |
            | 0   0   1  |

    dist_coefs : array of shape (5,)
        Distortion coefficients (k1,  k2,  p1,  p2,  k3).
    """
    calib_uvs = calib_uvs[~np.isnan(calib_uvs).any((1, 2))]
    n_samples = min(n_samples, len(calib_uvs))

    imgpoints = [
        calib_uvs[i]
        for i in np.random.choice(len(calib_uvs), n_samples, replace=False)
    ]
    imgpoints = np.array(imgpoints).astype(np.float32)

    calib_objpoints = np.repeat(calib_objpoints[na], n_samples, axis=0).astype(
        np.float32
    )
    flags = (
        cv2.CALIB_FIX_K3 * fix_k3
        + cv2.CALIB_ZERO_TANGENT_DIST * zero_tangent_dist
    )

    camera_matrix, dist_coefs = cv2.calibrateCamera(
        calib_objpoints, imgpoints, image_size, None, None, flags=flags
    )[1:3]
    return camera_matrix, dist_coefs.squeeze()


def estimate_pose(calib_uvs, calib_objpoints, camera_matrix, dist_coeffs):
    """
    Estimate the pose of a calibration object in a series of frames.

    The pose in each frame is the rigid transform from the calibration
    object's canonical coordinate frame to the camera's coordinate frame.

    Parameters
    ----------
    calib_uvs : array of shape (n_frames, N, 2)
        Calibration point detections for a number of frames.

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    camera_matrix : array of shape (3, 3)
        Camera matrix (use :py:func:`multicam_calibration.get_intrinsics` to
        calculate).

    dist_coeffs : array of shape (5,)
        Distortion coefficients (use
        :py:func:`multicam_calibration.get_intrinsics` to calculate).

    Returns
    -------
    poses : array of shape (n_frames, 6)
        Pose of the calibration object in each frame, where the first three
        columns are the rotation in axis-angle form and the last three columns
        are the translation. NaNs in a row indecate that the pose could not be
        estimated or that the calibration object was not detected in that frame.
    """
    poses = np.zeros((len(calib_uvs), 6)) * np.nan
    for ix, imgpoints in enumerate(calib_uvs):
        if not np.isnan(imgpoints).any():
            retval, rvec, tvec = cv2.solvePnP(
                calib_objpoints, imgpoints, camera_matrix, dist_coeffs
            )
            if retval:
                poses[ix] = np.concatenate([rvec.squeeze(), tvec.squeeze()])
    return poses


def estimate_pairwise_camera_transform(camera1_poses, camera2_poses):
    """
    Estimate the transformation between two cameras.

    Parameters
    ----------
    camera1_poses : array of shape (n_frames, 6)
        Calibration object poses of the first camera (see
        :py:func:`multicam_calibration.estimate_pose`).

    camera2_poses : array of shape (n_frames, 6)
        Calibration object poses of the second camera (see
        :py:func:`multicam_calibration.estimate_pose`).

    Returns
    -------
    transform : arrays of shape (6,)
        Transformation from the first camera's coordinate system to the second
        camera's coordinate system in vector format, where the first three
        elements specify a rotation in axis-angle form and the last three
        elements specify a translation.
    """
    common_frames = ~np.isnan([camera1_poses, camera2_poses]).any((0, 2))
    T1 = get_transformation_matrix(camera1_poses[common_frames])
    T2 = get_transformation_matrix(camera2_poses[common_frames])
    T_rel = np.matmul(T2, np.linalg.inv(T1))
    transforms = get_transformation_vector(T_rel)
    return np.median(transforms, axis=0)


def get_camera_spanning_tree(all_calib_poses, root=0):
    """
    Estimate the maximal spanning tree over camera pairs.

    This function outputs the maximal spanning tree in a graph where each node
    represents a camera and each edge represents the number of frames with 
    calibration object detections for both cameras.

    Parameters
    ----------
    all_calib_poses : array-like with shape (n_cameras, n_frames, 6)
        Calibration object poses across frames for each camera. Frames with no
        calibration object detections should be set to NaN (see 
        :py:func:`multicam_calibration.estimate_pose`).

    root : int, default=0
        Index of the root camera (used for ordering edges in the spanning tree).

    Returns
    -------
    spanning_tree : list of tuples
        Each tuple contains the indexes of two cameras that are connected by an
        edge in the spanning tree. The edges are topoligically sorted by 
        distance from the root, with the first node in each tuple being nearer
        to the root than the second node. For example, the tree::

            0
            |
            1
           / \
          2   3

        with root "0" would represented as::

            [(0,1), (1,2), (1,3)]
    """
    edges = []
    n_cameras = len(all_calib_poses)
    calib_detected = ~np.isnan(all_calib_poses).any(2)
    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            n_common_frames = (
                calib_detected[:, i] & calib_detected[:, j]
            ).sum()
            edges.append((i, j, n_common_frames))

    G = nx.Graph()
    G.add_nodes_from(range(n_cameras))
    G.add_weighted_edges_from(edges)
    spanning_tree = nx.maximum_spanning_tree(G)
    root_dist = nx.shortest_path_length(spanning_tree, source=root)
    edges = [
        tuple(sorted(e, key=lambda n: root_dist[n]))
        for e in spanning_tree.edges
    ]
    edges = sorted(edges, key=lambda e: root_dist[e[0]])
    return edges


def estimate_all_extrinsics(all_calib_poses, root=0):
    """
    Estimate transformations from the root camera to all cameras.

    Parameters
    ----------
    all_calib_poses : array-like with shape (n_cameras, n_frames, 6)
        Calibration object poses across frames for each camera. Frames with no
        calibration object detections should be set to NaN (see
        :py:func:`multicam_calibration.estimate_pose`).

    root : int, default=0
        Index of the root camera.

    Returns
    -------
    all_extrinsics : array of shape (n_cameras, 6)
        Transforms from the root camera's coordinate system to the coordinate
        system of each camera (the root camera's transform is the identity).
        The first three elements specify a rotation in axis-angle form and the
        last three elements specify a translation.
    """
    all_extrinsics = [None] * len(all_calib_poses)
    all_extrinsics[root] = np.eye(4)
    spanning_tree = get_camera_spanning_tree(all_calib_poses, root=root)

    for c1, c2 in spanning_tree:
        transform = estimate_pairwise_camera_transform(
            all_calib_poses[c1], all_calib_poses[c2]
        )
        all_extrinsics[c2] = (
            get_transformation_matrix(transform) @ all_extrinsics[c1]
        )
    all_extrinsics = np.array(
        list(map(get_transformation_vector, all_extrinsics))
    )
    return all_extrinsics


def consensus_calib_poses(all_calib_poses, all_extrinsics):
    """
    Estimate the consensus pose of the calibration object in each frame.

    Poses estimated from each camera separately are mapped transformed into
    common (world) coordinates and then the median across cameras is used as
    the final estimate.

    Parameters
    ----------
    all_calib_poses : array-like with shape (n_cameras, n_frames, 6)
        Calibration object poses across frames for each camera. Frames with no
        calibration object detections should be set to NaN (see
        :py:func:`multicam_calibration.estimate_pose`).

    all_extrinsics : array of shape (n_camera, 6)
        Transforms from world coordinates to the coordinate system of each
        camera (see :py:func:`multicam_calibration.estimate_all_extrinsics`).

    Returns
    -------
    calib_poses : array of shape (n_frames, 6)
        Consensus calibration object pose in each frame. Frames with no
        calibration detections in any camera are set to NaN.
    """
    all_calib_poses_world_coords = np.zeros_like(all_calib_poses) * np.nan
    for i, (poses, transform) in enumerate(
        zip(all_calib_poses, all_extrinsics)
    ):
        calib_detected = ~np.isnan(poses).any(axis=-1)
        T_board2camera = get_transformation_matrix(poses[calib_detected])
        T_world2camera = get_transformation_matrix(transform)
        T_board2world = np.matmul(
            np.linalg.inv(T_world2camera), T_board2camera
        )
        all_calib_poses_world_coords[
            i, calib_detected
        ] = get_transformation_vector(T_board2world)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        calib_poses = np.nanmedian(all_calib_poses_world_coords, axis=0)
    return calib_poses


def calibrate(
    all_calib_uvs,
    img_sizes,
    calib_objpoints,
    root=0,
    verbose=True,
    n_samples_for_intrinsics=100,
):
    """
    Estimate 3D positions of calibration points across frames and
    the intrinsic and extrinsic parameters of each camera.

    Parameters
    ----------
    all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
        Calibration point detections for a number of frames for each camera.
        NaNs are used to indicate missing detections.

    img_sizes : list of tuples (width, height)
        Image sizes for each camera.

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    root : int, default=0
        Index of the camera to use as the root of the coordinate system.

    verbose : bool, default=True
        If True, print progress updates.

    n_samples_for_intrinsics : int, default=100
        Number of frames to use for estimating camera intrinsics (pass as
        `n_samples` to :py:func:`multicam_calibration.get_intrinsics`).

    Returns
    -------
    all_extrinsics : array of shape (n_cameras, 6)
        Transforms from the root camera's coordinate system to the coordinate
        system of each camera (the root camera's transform is the identity).
        The first three elements specify a rotation in axis-angle form and the
        last three elements specify a translation.

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Camera intrinsics for each camera (see
        :py:func:`multicam_calibration.get_intrinsics`).

    consensus_calib_poses : array of shape (n_frames, 6)
        Calibration object poses in world coordinates, defined as the rigid
        transformation from the canonical reference frame of the calibration
        object to world coordinates. Frames with no calibration object
        detections in any camera are set to NaN.
    """
    n_cameras = len(all_calib_uvs)

    all_intrinsics = []
    if verbose:
        print("Estimating camera intrinsics")
    for uvs, img_size in tqdm.tqdm(
        zip(all_calib_uvs, img_sizes),
        total=n_cameras,
        ncols=72,
        disable=not verbose,
    ):
        intrinsics = get_intrinsics(
            uvs, calib_objpoints, img_size, n_samples=n_samples_for_intrinsics
        )
        all_intrinsics.append(intrinsics)

    all_calib_poses = []
    if verbose:
        print("Initializing calibration object poses")
    for uvs, intrinsics in tqdm.tqdm(
        zip(all_calib_uvs, all_intrinsics),
        total=n_cameras,
        disable=not verbose,
        ncols=72,
    ):
        calib_poses = estimate_pose(uvs, calib_objpoints, *intrinsics)
        all_calib_poses.append(calib_poses)
    all_calib_poses = np.array(all_calib_poses)

    if verbose:
        print("Estimating camera extrinsics")
    all_extrinsics = estimate_all_extrinsics(all_calib_poses, root=root)

    if verbose:
        print("Merging calibration object poses")
    calib_poses = consensus_calib_poses(all_calib_poses, all_extrinsics)

    return all_extrinsics, all_intrinsics, calib_poses
