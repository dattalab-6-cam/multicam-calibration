import numpy as np
import networkx as nx
import cv2
import tqdm
import warnings
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
    A = np.zeros((*r.shape[:-1],3,3))
    A[...,0,1] = -r[...,2]
    A[...,0,2] =  r[...,1]
    A[...,1,0] =  r[...,2]
    A[...,1,2] = -r[...,0]
    A[...,2,0] = -r[...,1]
    A[...,2,1] =  r[...,0]
    theta = np.linalg.norm(r, axis=-1, keepdims=True).reshape(*r.shape[:-1],1,1)
    A = A / np.where(theta == 0, 1, theta)
    R = np.sin(theta)*A  + (1-np.cos(theta))*np.matmul(A, A)
    R[...,0,0] += 1
    R[...,1,1] += 1
    R[...,2,2] += 1
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
    r = np.stack([R[...,2,1] - R[...,1,2],
                  R[...,0,2] - R[...,2,0],
                  R[...,1,0] - R[...,0,1]], axis=-1)
    theta = np.arccos((np.trace(R, axis1=-2, axis2=-1) - 1) / 2).reshape(*R.shape[:-2],1)
    rnorm = np.linalg.norm(r, axis=-1, keepdims=True)
    rnorm += (rnorm == 0)
    return r * theta / rnorm


def get_transformation_matrix(t):
    """
    Generate 4x4 transformation matrices from a 3D rotations and translations.

    Parameters
    ----------
    t : array of shape (...,6)
        Rigid transforms in vector format. The first three elements specify a rotation 
        in axis-angle form and the last three elements specify a translation.

    Returns
    -------
    T : array of shape (...,4,4)
        Transformation matrices.
    """
    T = np.zeros((*t.shape[:-1],4,4))
    T[...,:3,:3] = rodrigues(t[...,:3])
    T[...,:3,3] = t[...,3:]
    T[...,3,3] = 1
    return T
    

def get_transformation_vector(T):
    """
    Convert 4x4 transformation matrices to vector format.

    This function inverts :py:func:`multicam_calibration.get_transformation_matrix`.

    Parameters
    ----------
    T : array of shape (...,4,4)
        Transformation matrices.

    Returns
    -------
    t : array of shape (...,6)
        Rigid transforms in vector format. The first three elements specify a rotation 
        in axis-angle form and the last three elements specify a translation.
    """
    return np.concatenate([rodrigues_inv(T[...,:3,:3]),T[...,:3,3]],axis=-1)


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
    padding = np.ones((*x_euclidean.shape[:-1],1))
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
    x_euclidean = x_homogenous[...,:-1] / x_homogenous[...,-1:]
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
        coordinate system. The first three elements are the rotation vector
        and the last three elements are the translation vector.

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
    points_cam = np.matmul(T, points[...,na])[...,:3,0]

    # Apply radial distortion
    k1, k2 = dist_coefs[:2]
    r2 = np.sum((points_cam[...,:2]/points_cam[...,2:])**2, axis=-1)
    radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2

    # Distorted points in camera coordinate system
    points_cam_distorted = points_cam * np.stack(
        (radial_distortion, radial_distortion, np.ones(points_cam.shape[:-1])), axis=-1)

    # Project points to the image plane
    uvs = np.matmul(camera_matrix, points_cam_distorted[...,na]).squeeze(-1)
    uvs = uvs[...,:2] / uvs[...,2:]
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
    uvs_undistorted = np.zeros_like(uvs)*np.nan
    uvs_undistorted[valid_ixs] = cv2.undistortPoints(
        uvs[valid_ixs], camera_matrix, dist_coefs).squeeze(1)
    return uvs_undistorted.reshape(uvs_shape)


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
    pose_transforms = get_transformation_matrix(calib_poses)[:,na]
    objpoints = euclidean_to_homogenous(calib_objpoints)[na,:,:,na]
    calib_worldpoints = (pose_transforms @ objpoints)[...,:3,0]
    return calib_worldpoints

def get_intrinsics(calib_uvs, calib_objpoints, image_size, n_samples=100, 
                   fix_k3=True, zero_tangent_dist=True):
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

    n_samples : int, default=50
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
    calib_uvs = calib_uvs[~np.isnan(calib_uvs).any((1,2))]
    n_samples = min(n_samples, len(calib_uvs))
    
    imgpoints = [calib_uvs[i] for i in np.random.choice(len(calib_uvs),n_samples, replace=False)]
    imgpoints = np.array(imgpoints).astype(np.float32)

    calib_objpoints = np.repeat(calib_objpoints[na],n_samples,axis=0).astype(np.float32)
    flags = cv2.CALIB_FIX_K3 * fix_k3 + cv2.CALIB_ZERO_TANGENT_DIST * zero_tangent_dist

    camera_matrix, dist_coefs = cv2.calibrateCamera(calib_objpoints, imgpoints, image_size, None, None, flags=flags)[1:3]
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
        Camera matrix (use :py:func:`multicam_calibration.get_intrinsics` to calculate).

    dist_coeffs : array of shape (5,)
        Distortion coefficients (use :py:func:`multicam_calibration.get_intrinsics` to calculate).

    Returns
    -------
    poses : array of shape (n_frames, 6)
        Pose of the calibration object in each frame, where the first three columns
        are the rotation in axis-angle form and the last three columns are the 
        translation. NaNs in a row indecate that the pose could not be estimated 
        or that the calibration object was not detected in that frame.
    """
    poses = np.zeros((len(calib_uvs),6))*np.nan
    for ix,imgpoints in enumerate(calib_uvs):
        if not np.isnan(imgpoints).any():
            retval, rvec, tvec = cv2.solvePnP(calib_objpoints, imgpoints, camera_matrix, dist_coeffs)
            if retval: poses[ix] = np.concatenate([rvec.squeeze(),tvec.squeeze()])
    return poses

    
def estimate_pairwise_camera_transform(camera1_poses, camera2_poses):
    """
    Estimate the transformation between two cameras.
     
    Parameters
    ----------
    camera1_poses : array of shape (n_frames, 6)
        Calibration object poses of the first camera (see :py:func:`multicam_calibration.estimate_pose`).

    camera2_poses : array of shape (n_frames, 6)
        Calibration object poses of the second camera (see :py:func:`multicam_calibration.estimate_pose`).

    Returns
    -------
    transform : arrays of shape (6,)
        Transformation from the first camera's coordinate system to the second 
        camera's coordinate system in vector format, where the first three
        elements specify a rotation in axis-angle form and the last three
        elements specify a translation.
    """
    common_frames = ~np.isnan([camera1_poses, camera2_poses]).any((0,2))
    T1 = get_transformation_matrix(camera1_poses[common_frames])
    T2 = get_transformation_matrix(camera2_poses[common_frames])
    T_rel = np.matmul(T2, np.linalg.inv(T1))
    transforms = get_transformation_vector(T_rel)
    return np.median(transforms, axis=0)

    
def get_camera_spanning_tree(all_calib_poses, root=0):
    """
    Estimate the maximal spanning tree over camera pairs.

    This function outputs the maximal spanning tree in a graph where 
    each node represents a camera and each edge represents the number of
    frames with calibration object detections for both cameras.

    Parameters
    ----------
    all_calib_poses : array-like with shape (n_cameras, n_frames, 6)
        Calibration object poses across frames for each camera. Frames with no
        calibration object detections should be set to NaN (see :py:func:`multicam_calibration.estimate_pose`).

    root : int, default=0
        Index of the root camera (used for ordering edges in the spanning tree).

    Returns
    -------
    spanning_tree : list of tuples
        Each tuple contains the indexes of two cameras that are connected
        by an edge in the spanning tree. The edges are topoligically sorted
        by distance from the root, with the first node in each tuple being
        nearer to the root than the second node. For example, the tree::

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
        for j in range(i+1, n_cameras):
            n_common_frames = (calib_detected[:,i] & calib_detected[:,j]).sum()
            edges.append((i, j, n_common_frames))

    G = nx.Graph()
    G.add_nodes_from(range(n_cameras))
    G.add_weighted_edges_from(edges)
    spanning_tree = nx.maximum_spanning_tree(G)
    root_dist = nx.shortest_path_length(spanning_tree, source=root)
    edges = [tuple(sorted(e, key=lambda n: root_dist[n])) for e in spanning_tree.edges]
    edges = sorted(edges, key=lambda e: root_dist[e[0]])
    return edges


def estimate_all_extrinsics(all_calib_poses, root=0):
    """
    Estimate transformations from the root camera to all cameras.

    Parameters
    ----------
    all_calib_poses : array-like with shape (n_cameras, n_frames, 6)
        Calibration object poses across frames for each camera. Frames with no
        calibration object detections should be set to NaN (see :py:func:`multicam_calibration.estimate_pose`).

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

    for c1,c2 in spanning_tree:
        transform = estimate_pairwise_camera_transform(all_calib_poses[c1], all_calib_poses[c2])
        all_extrinsics[c2] = get_transformation_matrix(transform) @ all_extrinsics[c1]
    all_extrinsics = np.array(list(map(get_transformation_vector, all_extrinsics)))
    return all_extrinsics

def consensus_calib_poses(all_calib_poses, all_extrinsics):
    """
    Estimate the consensus pose of the calibration object in each frame.

    Poses estimated from each camera separately are mapped transformed into 
    common (world) coordinates and then the median across cameras is used 
    as the final estimate.

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
        Consensus calibration object pose in each frame. Frames with no calibration
        detections in any camera are set to NaN.
    """
    all_calib_poses_world_coords = np.zeros_like(all_calib_poses)*np.nan
    for i,(poses,transform) in enumerate(zip(all_calib_poses, all_extrinsics)):
        calib_detected = ~np.isnan(poses).any(axis=-1)
        T_board2camera = get_transformation_matrix(poses[calib_detected])
        T_world2camera = get_transformation_matrix(transform)
        T_board2world = np.matmul(np.linalg.inv(T_world2camera), T_board2camera)
        all_calib_poses_world_coords[i,calib_detected] = get_transformation_vector(T_board2world)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        calib_poses = np.nanmedian(all_calib_poses_world_coords, axis=0)
    return calib_poses


def calibrate(all_calib_uvs, img_sizes, calib_objpoints, root=0, 
              verbose=True, n_samples_for_intrinsics=100):
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
        object to world coordinates. Frames with no calibration object detections 
        in any camera are set to NaN.
    """
    n_cameras = len(all_calib_uvs)

    all_intrinsics = []
    if verbose: print('Estimating camera intrinsics')
    for uvs, img_size in tqdm.tqdm(zip(all_calib_uvs, img_sizes), total=n_cameras, disable=not verbose):
        intrinsics = get_intrinsics(uvs, calib_objpoints, img_size, n_samples=n_samples_for_intrinsics)
        all_intrinsics.append(intrinsics)

    all_calib_poses = []
    if verbose: print('Initializing calibration object poses')
    for uvs, intrinsics in tqdm.tqdm(zip(all_calib_uvs, all_intrinsics), total=n_cameras, disable=not verbose):
        calib_poses = estimate_pose(uvs, calib_objpoints, *intrinsics)
        all_calib_poses.append(calib_poses)
    all_calib_poses = np.array(all_calib_poses)

    if verbose: print('Estimating camera extrinsics')
    all_extrinsics = estimate_all_extrinsics(all_calib_poses, root=root)

    if verbose: print('Merging calibration object poses')
    calib_poses = consensus_calib_poses(all_calib_poses, all_extrinsics)

    return all_extrinsics, all_intrinsics, calib_poses


# def filter_detections(all_calib_uvs, calib_objpoints, all_intrinsics, error_threshold=1):
#     """
#     Filter calibration detections that don't have the correct geometry.

#     When the calibration points are detected correctly, they should map
#     exactly onto the canonical coordinates defined by `calib_objpoints`
#     after undisortion and homography correction. This function filters
#     out detections where any point's distance from its canonical position
#     is greater than `error_threshold`.

#     Parameters
#     ----------
#     all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
#         Calibration point detections for a number of frames for each camera.
#         NaNs are used to indicate missing detections.

#     calib_objpoints : array of shape (N, 3)
#         3D coordinates of the calibration points in a canonical reference frame.

#     all_intrinsics : list of tuples (camera_matrix, dist_coefs)
#         Camera intrinsics for each camera (see :py:func:`multicam_calibration.get_intrinsics`).

#     error_threshold : float, default=1
#         Maximum error distance to allow for any point in a frame (in the same 
#         units as `calib_objpoints`). 

#     Returns
#     -------
#     filtered_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
#         Filtered calibration detections, which are identical to 
#         `all_calib_uvs` except that filtered detections are set to NaN.
#     """
#     n_cameras, n_frames, N, _ = all_calib_uvs.shape
#     filtered_calib_uvs = all_calib_uvs.copy()
#     for cam in range(n_cameras):
#         uvs_undistorted = undistort_points(all_calib_uvs[cam], *all_intrinsics[cam])
#         for t in np.nonzero(~np.isnan(uvs_undistorted).any((-1,-2)))[0]:
#             H = cv2.findHomography(uvs_undistorted[t], calib_objpoints[:,:2])[0]
#             pts = cv2.perspectiveTransform(uvs_undistorted[t][na], H)
#             error = ((pts - calib_objpoints[:,:2])**2).sum(1)
#             if error.max() > error_threshold**2: filtered_calib_uvs[cam, t] = np.nan
#     return filtered_calib_uvs