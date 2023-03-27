import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import warnings
na = np.newaxis

from .geometry import (
    get_transformation_matrix,
    euclidean_to_homogenous,
    undistort_points,
    embed_calib_objpoints)


def pad_axis_limits(xmin,xmax,ymin,ymax, pad=0.1):
    """
    Pad x/y limits by a given fraction.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Axis limits.

    pad : float, default=0.1
        Fraction of the axis range to pad.

    Returns
    -------
    xmin, xmax, ymin, ymax : float
        Padded axis limits.
    """
    x_range = xmax - xmin
    y_range = ymax - ymin
    xmin -= pad*x_range
    xmax += pad*x_range
    ymin -= pad*y_range
    ymax += pad*y_range
    return xmin, xmax, ymin, ymax

def set_axis_limits(ax, data, pctl=99, pad=0.1):
    """
    Set the axis limits of a matplotlib axis based on the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to set limits for.

    data : array of shape (...,2)
        Data to use for setting the limits. NaNs are ignored.

    pctl : float, default=99
        Percentile of the data to use for setting the limits.

    pad : float, default=0.1
        Fraction of the axis range to pad.
    """
    xmin = np.nanpercentile(data[...,0], pctl)
    xmax = np.nanpercentile(data[...,0], 100-pctl)
    ymin = np.nanpercentile(data[...,1], pctl)
    ymax = np.nanpercentile(data[...,1], 100-pctl)
    xmin, xmax, ymin, ymax = pad_axis_limits(xmin, xmax, ymin, ymax, pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_residuals(all_calib_uvs, all_extrinsics, all_intrinsics, calib_objpoints, 
                   calib_poses,  max_points=10000, marker_size=1, target_size=250, 
                   n_cols=3, inches_per_axis=5, hide_axes=True):
    """
    Visualize the reprojection error of the calibration.

    Estimated 3D coordinates of the calibration points (based on `calib_poses`)
    are projected into each camera's image plane using its extrinsic parameters. 
    If calibration were perfect, the reprojections would be identical to the
    undistorted calibration detections (i.e. `cv2.undistortPoints` applied to 
    `all_calib_uvs`). To visualize errors simultaneously across frames, a 
    perspective transform is used to map the reprojection residuals into the
    canonical reference frame of the calibration object (i.e. `calib_objpoints`).

    A scatter plot is generated for each camera. The target (canonical)
    calibration points are shown as black "+" signs, and the reprojections
    are overlayed in red.

    Parameters
    ----------
    all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
        Calibration point detections for a number of frames for each camera.
        NaNs are used to indicate missing detections.

    all_extrinsics : array of shape (n_cameras, 6)
        Transforms from world coordinates to each camera's coordinates.

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Camera intrinsics for each camera (see :py:func:`multicam_calibration.get_intrinsics`).

    calib_poses : array of shape (n_frames, 6)
        Calibration object poses in each frame.

    calib_objpoints : array of shape (N, 3)
        3D coordinates of the calibration points in a canonical reference frame.

    max_points : int, default=10000
        Maximum number of points to plot.

    marker_size : float, default=1
        Size of the markers used to plot the detections and reprojections.

    target_size : float, default=100
        Size of the markers used to plot the canonical calibration points.

    n_cols : int, default=3
        Number of columns in the figure.

    inches_per_axis : float, default=3
        Width of each axis in the figure (in inches).

    hide_axes : bool, default=True
        If True, the axes are hidden.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.

    median_error: array of shape (n_cameras,)
        Median distance between the reprojections and their target locations
        for each camera (in the same units as `calib_objpoints`).

    reprojections: array of shape (n_cameras, n_frames, N, 2)
        Reprojections of 3D calibration points into each camera's image plane
        in normalized (u,v) coordinates. These should ideally be identical to
        the detections (`all_calib_uvs`) after undisortion.

    transformed_reprojections: array of shape (n_cameras, n_frames, N, 2)
        Predicted coordinates of the calibration points transformed into the
        canonical reference frame of the calibration object. NaNs are used for
        camera/frame combinations where the calibration object was not visible.
    """
    
    n_cameras, n_frames, N, _ = all_calib_uvs.shape
    median_error = np.zeros(n_cameras)
    reprojections = np.zeros((n_cameras, n_frames, N, 2))
    transformed_reprojections = np.zeros((n_cameras, n_frames, N, 2))*np.nan
    pts = embed_calib_objpoints(calib_objpoints, calib_poses)

    for cam in tqdm.trange(n_cameras):
        T = get_transformation_matrix(all_extrinsics[cam])
        points_cam = np.matmul(T, euclidean_to_homogenous(pts)[...,na])[...,:3,0]
        reprojections[cam] = points_cam[...,:2]/points_cam[...,2:]
        uvs_undistorted = undistort_points(all_calib_uvs[cam], *all_intrinsics[cam])
        
        valid_ixs = np.nonzero(~np.isnan(uvs_undistorted).any((-1,-2)))[0]
        for t in valid_ixs:
            H = cv2.findHomography(uvs_undistorted[t], calib_objpoints[:,:2])[0]
            transformed_reprojections[cam, t] = cv2.perspectiveTransform(reprojections[cam,t][na], H)[0]

        median_error[cam] = np.median(
            np.sqrt((transformed_reprojections[cam,valid_ixs] - calib_objpoints[:,:2])**2).sum(-1))
    
    n_rows = int(np.ceil(n_cameras/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols)
    for cam in range(n_cameras):
        pts = transformed_reprojections[cam].reshape(-1,2)
        plot_ixs = np.nonzero(~np.isnan(pts).any(-1))[0]
        if len(plot_ixs) > max_points:
            plot_ixs = np.random.choice(plot_ixs, max_points, replace=False)
        
        
        axes.flat[cam].scatter(*calib_objpoints[:,:2].T, c='k', s=target_size, marker='+', linewidth=0.5)
        axes.flat[cam].scatter(*pts[plot_ixs].T, c='r', s=marker_size, linewidth=0)
        axes.flat[cam].set_title(f'camera {cam} (median error={median_error[cam]:.2f})', fontsize=10)
        axes.flat[cam].set_aspect('equal')
        set_axis_limits(axes.flat[cam], pts[plot_ixs], pctl=99, pad=0.1)
        if hide_axes: axes.flat[cam].axis('off')

    aspect_ratio = calib_objpoints[:,1].ptp()/calib_objpoints[:,0].ptp()
    fig.set_size_inches((n_cols*inches_per_axis, n_rows*inches_per_axis*aspect_ratio))
    return fig, median_error, reprojections, transformed_reprojections
