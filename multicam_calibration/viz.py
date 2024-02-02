from vidio.read import OpenCVReader
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import imageio
import tqdm
import h5py
import cv2
import os

na = np.newaxis

from .geometry import *
from .bundle_adjustment import embed_calib_objpoints


def pad_axis_limits(xmin, xmax, ymin, ymax, pad=0.1):
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
    xmin -= pad * x_range
    xmax += pad * x_range
    ymin -= pad * y_range
    ymax += pad * y_range
    return xmin, xmax, ymin, ymax


def set_axis_limits(ax, data, pctl=1, pad=0.1):
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
    xmin = np.nanpercentile(data[..., 0], pctl)
    xmax = np.nanpercentile(data[..., 0], 100 - pctl)
    ymin = np.nanpercentile(data[..., 1], pctl)
    ymax = np.nanpercentile(data[..., 1], 100 - pctl)
    xmin, xmax, ymin, ymax = pad_axis_limits(xmin, xmax, ymin, ymax, pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_residuals(
    all_calib_uvs,
    all_extrinsics,
    all_intrinsics,
    calib_objpoints,
    calib_poses,
    max_points=10000,
    marker_size=1,
    target_size=250,
    n_cols=3,
    inches_per_axis=5,
    hide_axes=True,
):
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
        Camera intrinsics for each camera (see
        :py:func:`multicam_calibration.get_intrinsics`).

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
    transformed_reprojections = np.zeros((n_cameras, n_frames, N, 2)) * np.nan
    pts = embed_calib_objpoints(calib_objpoints, calib_poses)

    for cam in tqdm.trange(n_cameras):
        reprojections[cam] = project_points(
            pts, all_extrinsics[cam], all_intrinsics[cam][0]
        )
        uvs_undistorted = undistort_points(all_calib_uvs[cam], *all_intrinsics[cam])
        valid_ixs = np.nonzero(~np.isnan(uvs_undistorted).any((-1, -2)))[0]
        for t in valid_ixs:
            H = cv2.findHomography(uvs_undistorted[t], calib_objpoints[:, :2])
            transformed_reprojections[cam, t] = cv2.perspectiveTransform(
                reprojections[cam, t][na], H[0]
            )[0]

        errors = np.linalg.norm(
            transformed_reprojections[cam, valid_ixs] - calib_objpoints[:, :2],
            axis=-1,
        )
        median_error[cam] = np.median(errors)

    n_rows = int(np.ceil(n_cameras / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols)
    for cam in range(n_cameras):
        pts = transformed_reprojections[cam].reshape(-1, 2)
        plot_ixs = np.nonzero(~np.isnan(pts).any(-1))[0]
        if len(plot_ixs) > max_points:
            plot_ixs = np.random.choice(plot_ixs, max_points, replace=False)

        axes.flat[cam].scatter(
            *calib_objpoints[:, :2].T,
            c="k",
            s=target_size,
            marker="+",
            linewidth=0.5,
        )
        axes.flat[cam].scatter(*pts[plot_ixs].T, c="r", s=marker_size, linewidth=0)
        axes.flat[cam].set_title(
            f"camera {cam} (median error={median_error[cam]:.2f})", fontsize=10
        )
        axes.flat[cam].set_aspect("equal")
        set_axis_limits(axes.flat[cam], pts[plot_ixs], pctl=99, pad=0.1)
        if hide_axes:
            axes.flat[cam].axis("off")

    aspect_ratio = calib_objpoints[:, 1].ptp() / calib_objpoints[:, 0].ptp()
    fig.set_size_inches(
        (n_cols * inches_per_axis, n_rows * inches_per_axis * aspect_ratio)
    )
    return fig, median_error, reprojections, transformed_reprojections


def overlay_detections(
    video_path,
    detections=None,
    output_path=None,
    overwrite=False,
    dotsize=3,
    draw_lines=True,
    frame_label_color=(0, 0, 255),
    frame_label_size=1,
    quality=6,
):
    """
    Overlay calibration object detections on a video.

    Parameters
    ----------
    video_path : str
        Path to a video of the calibration object.

    detections : dict, optional
        Dictionary mapping frame indexes to calibration object detections as
        arrays of shape (N,2). If not provided, the detections will be loaded
        from `{video_path}.detections.h5` (see
        :py:func:`multicam_calibration.detection.process_video` for the file
        format.

    output_path : str, optional
        Path to save the output video. If not provided, the video will be saved
        to `{video_path}.overlay.mp4`.

    overwrite : bool, default=False
        If True, overwrite the output video if it already exists.

    dotsize : int, default=3
        Size of the dots used to plot the calibration object points.

    draw_lines : bool, default=True
        If True, lines will be drawn between the calibration object points.

    frame_label_color : tuple (r,g,b), default=(0,0,255)
        Color of the frame labels.

    frame_label_size : int, default=1
        Font size of the frame labels.

    quality : int, default=6
        Quality of the output video (passed to `imageio.get_writer`)
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".overlay.mp4"

    if os.path.exists(output_path) and not overwrite:
        print(f'{output_path} already exists. Set "overwrite=True" to overwrite.')
        return

    if detections is None:
        detections_path = os.path.splitext(video_path)[0] + ".detections.h5"

        assert os.path.exists(
            detections_path
        ), f"No detections found at {detections_path}."

        with h5py.File(detections_path, "r") as f:
            uvs = f["uvs"][()]
            frame_ixs = f["frame_ixs"][()]
            detections = {t: uvs[i] for i, t in enumerate(frame_ixs)}

    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".overlay.mp4"

    reader = OpenCVReader(video_path)

    with imageio.get_writer(
        output_path, pixelformat="yuv420p", fps=reader.fps, quality=quality
    ) as writer:
        for frame_ix, frame in tqdm.tqdm(
            enumerate(reader), total=len(reader), ncols=72, unit="frame"
        ):
            if frame_ix in detections:
                uvs = detections[frame_ix]
                positions = [(int(u), int(v)) for u, v in uvs]
                colors = plt.cm.jet(np.linspace(0, 1, len(uvs)))[:, :3] * 255
                colors = [(int(r), int(g), int(b)) for r, g, b in colors]

                for pos, color in zip(positions, colors):
                    frame = cv2.circle(
                        frame, pos, dotsize, color, -1, lineType=cv2.LINE_AA
                    )

                if draw_lines:
                    for pos1, pos2, color in zip(positions[:-1], positions[1:], colors):
                        frame = cv2.line(
                            frame, pos1, pos2, color, 2, lineType=cv2.LINE_AA
                        )

            frame = cv2.putText(
                frame,
                str(frame_ix),
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                frame_label_size,
                frame_label_color,
                2,  # thickness
                cv2.LINE_AA,
            )
            writer.append_data(frame)


def visualize_flatibration(
    transform, floor_points, keypoints=None, max_points_to_plot=5000, figsize=(12, 6)
):
    """
    Visualize the output of flatibration.

    Parameters
    ----------
    transform : array of shape (6,)
        Rigid transform output by :py:func:`multicam_calibration.flatibration.flatibrate`.

    floor_points : np.array of shape (n_points, 3) or a list of np.arrays
        3D keypoints that correspond to the floor of the recording arena.

    keypoints : np.array of shape (n_frames, n_keypoints, 3) or a list of np.arrays, optional
        3D keypoints of the animal(s) in the recording arena that were used for flatibration.

    max_points_to_plot : int, default=10000
        Maximum number of points to plot. If the number of points exceeds this value, a random
        subset of points will be plotted.

    figsize : tuple of int, default=(12, 12)
        Size of the figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    if isinstance(floor_points, list):
        floor_points = np.concatenate(floor_points)

    if len(floor_points) > max_points_to_plot:
        ix = np.random.choice(len(floor_points), max_points_to_plot, replace=False)
        floor_points = floor_points[ix]

    # plot untransformed floor points
    axs[0, 0].scatter(*floor_points[:, [0, 2]].T, s=1, label="floor points", zorder=1)
    set_axis_limits(axs[0, 0], floor_points[:, [0, 2]], pctl=99, pad=0.1)
    axs[1, 0].scatter(*floor_points[:, [1, 2]].T, s=1, label="floor points", zorder=1)
    set_axis_limits(axs[1, 0], floor_points[:, [1, 2]], pctl=99, pad=0.1)

    # plot transformed floor points
    floor_points = apply_rigid_transform(transform, floor_points)
    axs[0, 1].scatter(*floor_points[:, [0, 2]].T, s=1, label="floor points", zorder=1)
    set_axis_limits(axs[0, 0], floor_points[:, [0, 2]], pctl=99, pad=0.1)
    axs[1, 1].scatter(*floor_points[:, [1, 2]].T, s=1, label="floor points", zorder=1)
    set_axis_limits(axs[1, 0], floor_points[:, [1, 2]], pctl=99, pad=0.1)

    if keypoints is not None:
        if isinstance(keypoints, list):
            keypoints = np.concatenate(keypoints)

        keypoints = keypoints.reshape(-1, 3)
        if len(keypoints) > max_points_to_plot:
            ix = np.random.choice(len(keypoints), max_points_to_plot, replace=False)
            keypoints = keypoints[ix]

        # plot untransformed keypoints
        axs[0, 0].scatter(*keypoints[:, [0, 2]].T, s=1, label="keypoints", zorder=0)
        set_axis_limits(axs[0, 0], keypoints[:, [0, 2]], pctl=99, pad=0.1)
        axs[1, 0].scatter(*keypoints[:, [1, 2]].T, s=1, label="keypoints", zorder=0)
        set_axis_limits(axs[1, 0], keypoints[:, [1, 2]], pctl=99, pad=0.1)

        # plot transformed keypoints
        keypoints = apply_rigid_transform(transform, keypoints)
        axs[0, 0].scatter(*keypoints[:, [0, 2]].T, s=1, label="keypoints", zorder=0)
        set_axis_limits(axs[0, 0], keypoints[:, [0, 2]], pctl=99, pad=0.1)
        axs[1, 0].scatter(*keypoints[:, [1, 2]].T, s=1, label="keypoints", zorder=0)
        set_axis_limits(axs[1, 0], keypoints[:, [1, 2]], pctl=99, pad=0.1)

    for ax in axs[:, 0]:
        ax.set_title("untransformed")

    for ax in axs[:, 1]:
        ax.set_title("transformed")

    for ax in axs[0]:
        ax.set_xlabel("x")

    for ax in axs[1]:
        ax.set_xlabel("y")

    for ax in axs.flat:
        ax.set_ylabel("z")
        ax.axhline(0, color="k", lw=1, zorder=2)

    fig.tight_layout()
    return fig
