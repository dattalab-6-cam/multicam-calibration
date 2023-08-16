from vidio.read import OpenCVReader
from multiprocessing import Queue, Process
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import h5py
import cv2
import os

from .geometry import euclidean_to_homogenous, homogeneous_to_euclidean

# -----------------------------------------------------------------------------#
#               General calibration object detection utilities                #
# -----------------------------------------------------------------------------#


def _worker(frame_queue, result_queue, detection_fun, detection_kwargs):
    detections = {}
    while True:
        item = frame_queue.get()
        if item == "STOP":
            break
        frame_ix, frame = item
        detection = detection_fun(frame, **detection_kwargs)
        if detection is not None:
            detections[frame_ix] = detection
    result_queue.put(detections)


def process_video(
    video_path, detection_fun, detection_options, use_frames=None, n_workers=1
):
    """
    Detect a calibration object in a video and save the detections.

    Detections are saved to an .h5 file with path `{video_path}.detections.h5`.
    The file contains the following datasets:

    - `uvs` : array of shape (n_frames, N, 2)
        Detected (u,v) coordinates of the calibration object points.

    - `qc_data` : array of shape (n_frames, M)
        Quality control information. The exact shape depends on the detection
        function used.

    - `frames_ixs` : array of shape (n_frames,)
        Index of the frame corresponding to each detection.

    - `img_size` : tuple (width, height)
        Image dimensions.

    Parameters
    ----------
    video_path : str
        Path to the video to process.

    detection_fun : callable
        Function to detect the calibration object in a single frame. Should
        take an image as input and return an array of (u,v) coordinates of the
        calibration object points.

    detection_options : dict
        Keyword arguments to pass to `detection_fun`.

    use_frames : array of shape (n_frames,), optional
        Indices of frames to use. If not provided, all frames will be used.

    n_workers : int, default=1
        Number of processes to use for multiprocessing.
    """
    reader = OpenCVReader(video_path)
    img_size = reader[0].shape[:2][::-1]
    total_frames = len(reader)

    if use_frames is None:
        use_frames = np.arange(total_frames)

    if n_workers > 1:
        frame_queue = Queue(maxsize=10)
        result_queue = Queue()

        processes = []
        for i in range(n_workers):
            p = Process(
                target=_worker,
                args=(
                    frame_queue,
                    result_queue,
                    detection_fun,
                    detection_options,
                ),
            )
            p.start()
            processes.append(p)

        try:
            for frame_ix in tqdm.tqdm(use_frames, ncols=72, unit="frame"):
                frame_queue.put((frame_ix, reader[frame_ix]))
        except KeyboardInterrupt:
            print("Early termination, saving results so far...")

        for _ in range(n_workers):
            frame_queue.put("STOP")

        detections = {}
        for _ in range(n_workers):
            detections.update(result_queue.get())

        for p in processes:
            p.join()

    else:
        detections = {}
        try:
            for frame_ix in tqdm.tqdm(use_frames, ncols=72, unit="frame"):
                frame = reader[frame_ix]
                detection = detection_fun(frame, **detection_options)
                if detection is not None:
                    detections[frame_ix] = detection
        except KeyboardInterrupt:
            print("Early termination, saving results so far...")

    if len(detections) == 0:
        raise ValueError("No detections found")

    frame_ixs = np.array(sorted(detections.keys()))
    uvs = np.stack([detections[i][0] for i in frame_ixs])
    qc_data = np.stack([detections[i][1] for i in frame_ixs])

    save_path = os.path.splitext(video_path)[0] + ".detections.h5"
    with h5py.File(save_path, "w") as h5:
        h5.create_dataset("uvs", data=uvs)
        h5.create_dataset("qc_data", data=qc_data)
        h5.create_dataset("frame_ixs", data=frame_ixs)
        h5.create_dataset("img_size", data=img_size)


def run_calibration_detection(
    video_paths,
    detection_fun,
    detection_options={},
    aligned_frame_ixs=None,
    overwrite=False,
    n_workers=1,
):
    """
    Detect a calibration object in a set of videos.

    Parameters
    ----------
    video_paths : list of str
        Paths to the videos to process.

    detection_fun : callable
        Function to detect the calibration object in a single frame. Should
        take an image as input and return an array of (u,v) coordinates of the
        calibration object points.

    detection_options : dict, optional
        Keyword arguments to pass to the detection function.

    aligned_frame_ixs : ndarray of shape (n_framesets, n_cameras), optional
        Indices of frames that are aligned across cameras. For example, if
        `aligned_frame_ixs[0,:2]==[0,1]` then frame 0 of the first camera is
        simultaneous with frame 1 of the second camera. NaNs indicate dropped
        frames. If `aligned_frame_ixs` is not provided, then the videos are
        assumed to be synchronized from the start and not lack dropped frames.

    overwrite : bool, default=False
        If True, ignore any saved detections and re-run the detection function
        for all frames.

    n_workers : int, default=1
        Number of processes to use for parallel processing.

    Returns
    -------
    all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
        Synchronized calibration detections each camera. NaNs are used when no
        detection was made or the frame was dropped.

    img_sizes : list of tuples (width, height)
        Image dimensions for each camera.
    """
    all_uvs = []
    all_img_sizes = []
    all_frame_ixs = []

    if aligned_frame_ixs is None:
        n_vids = len(video_paths)
        n_frames = min(len(OpenCVReader(vp)) for vp in video_paths)
        aligned_frame_ixs = np.stack([np.arange(n_frames)] * n_vids).T

    for i, video_path in enumerate(video_paths):
        save_path = os.path.splitext(video_path)[0] + ".detections.h5"
        if overwrite or not os.path.exists(save_path):
            print(f"Processing {video_path}")
            process_video(
                video_path,
                detection_fun,
                detection_options,
                aligned_frame_ixs[:, i],
                n_workers,
            )

        with h5py.File(save_path, "r") as h5:
            all_uvs.append(h5["uvs"][:])
            all_img_sizes.append(h5["img_size"][:])
            all_frame_ixs.append(h5["frame_ixs"][:])

    assert np.all(
        [len(uvs) > 0 for uvs in all_uvs]
    ), "At least one video has no detections"

    n_calib_points = all_uvs[0].shape[1]
    n_frames = aligned_frame_ixs.shape[0]
    n_cameras = len(video_paths)

    all_calib_uvs = np.zeros((n_cameras, n_frames, n_calib_points, 2)) * np.nan
    for i, (uvs, frame_ixs) in enumerate(zip(all_uvs, all_frame_ixs)):
        uvs = uvs[np.argsort(frame_ixs)]
        frame_ixs = np.sort(frame_ixs)
        all_calib_uvs[i, aligned_frame_ixs[:, i].searchsorted(frame_ixs)] = uvs

    return all_calib_uvs, all_img_sizes


def summarize_detections(all_calib_uvs):
    """
    Print the number of detections shared between each pair of cameras.

    Parameters
    ----------
    all_calib_uvs : array of shape (n_cameras, n_frames, N, 2)
        Synchronized calibration detections each camera. NaNs are used when no
        detection was made or the frame was dropped.

    Returns
    -------
    table : pandas.DataFrame
        Table of the number of shared detections between each pair of cameras.
    """
    has_detection = ~np.isnan(all_calib_uvs).any(axis=(2, 3))
    n_shared = (has_detection[:, None, :] & has_detection[None, :, :]).sum(2)
    names = [f"Camera {i}" for i in range(len(all_calib_uvs))]
    table = pd.DataFrame(n_shared, index=names, columns=names)
    return table


# -----------------------------------------------------------------------------#
#                            Chessboard detection                             #
# -----------------------------------------------------------------------------#


def extend_grid(uv_grid, extend_rows, extend_cols):
    """
    Given (u,v) coordinates from a chessboard detection in an image, simulate
    the coordinates of an enlarged chessboard using a homography transform.

    Parameters
    ----------
    uv_grid : array of shape (rows,cols,2)
        Grid of (u,v) coordinates (e.g. representing chessboard vertices).

    extend_rows : int
        Number of rows to extend by in each direction.

    extend_cols : int
        Number of columns to extend by in each direction.

    Returns
    -------
    extended_uv_grid : array of shape (rows+2*extend_rows,cols+2*extend_cols,2)
        Extended grid of (u,v) coordinates.
    """
    rows = uv_grid.shape[0] + 2 * extend_rows
    cols = uv_grid.shape[1] + 2 * extend_cols
    
    xy_grid_full = np.mgrid[0:cols, 0:rows].T
    xy_grid = xy_grid_full[extend_rows:-extend_rows, extend_cols:-extend_cols]
    
    H, _ = cv2.findHomography(xy_grid.reshape(-1, 2), uv_grid.reshape(-1, 2))
    
    extended_uv_grid = homogeneous_to_euclidean(
        euclidean_to_homogenous(xy_grid_full.reshape(-1, 2)) @ H.T
    ).reshape(xy_grid_full.shape)
    
    return extended_uv_grid


def detect_chessboard(
    image,
    *,
    board_shape=(7, 10),
    reorder=True,
    subpix_winSize=(5, 5),
    scale_factor=1,
    adaptive_threshold=True,
    normalize_image=True,
):
    """
    Detect corners of a chessboard and order them using a anchor point.

    To use this function, the chessboard should have a dark circle in the
    top-left corner when viewed from the font. The dark circle serves as an
    anchor: Points are ordered such that the point nearest the dark circle is
    indexed first, followed by the others in row-major order.::

        ●
        ██░░██░░██
        ░░██░░██░░
        ██░░██░░██


    Parameters
    ----------
    image : array
        Image to detect the chessboard in.

    board_shape : tuple (rows,columns)
        Number of squares in each dimension minus one. For example the board
        shown above would have shape (2,4).

    reorder : bool, default=True
        Whether to reorder the points using an anchor.

    subpix_winSize : tuple (width,height), default=(5,5)
        Size of the window to use for subpixel refinement.

    scale_factor : float, default=1
        How much to shrink the image before initial chessboard detection.
        In all cases, the full resolution image is subsequently used for
        subpixel refinement.

    adaptive_threshold : bool, default=True
        Whether to use the flat `CALIB_CB_ADAPTIVE_THRESH` when applying
        `findChessboardCorners` in OpenCV.

    normalize_image : bool, default=True
        Whether to use the flat `CALIB_CB_NORMALIZE_IMAGE` when applying
        `findChessboardCorners` in OpenCV.

    Returns
    -------
    uvs: array of shape (N,2) or None
        Either an array of corner coordinates or None if no board was found.

    match_scores : array of shape (4,)
        Sorted template-matching correlations for the four possible anchor
        points locations.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = (
        cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_ADAPTIVE_THRESH * adaptive_threshold
        + cv2.CALIB_CB_NORMALIZE_IMAGE * normalize_image
    )
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if scale_factor != 1:
        resized_image = cv2.resize(
            image, None, fx=scale_factor, fy=scale_factor
        )
    else:
        resized_image = image

    ret, corners_approx = cv2.findChessboardCorners(
        resized_image, board_shape, flags
    )

    if ret:
        corners_approx = corners_approx / scale_factor
        uvs = cv2.cornerSubPix(
            image, corners_approx, subpix_winSize, (-1, -1), criteria
        ).squeeze()

        if reorder:
            uvs, match_scores,_ = reorder_chessboard_corners(
                image, uvs, board_shape
            )
        else:
            match_scores = None

        return uvs, match_scores
    else:
        return None


def _generate_chessboard_anchor_template(size):
    """Generate a white square with a black circle in the top-left corner."""
    template = np.ones((size, size), dtype=np.uint8) * 255
    template = cv2.circle(template, (size // 4, size // 4), size // 4, 0, -1)
    return template


def _extract_region(image, source_pts, target_pts, template):
    """Extract a region of an image that will be matched to a template."""
    M = cv2.getPerspectiveTransform(source_pts, target_pts)
    warped_image = cv2.warpPerspective(image, M, template.shape[:2][::-1])
    return warped_image


def _match_to_template(image, all_source_pts, target_pts, template):
    """Compute correlation between a template and a region of an image."""
    match_scores, regions = [], []
    for source_pts in all_source_pts:
        region = _extract_region(image, source_pts, target_pts, template)
        if np.std(region) > 0:
            match_score = np.corrcoef(region.ravel(), template.ravel())[0, 1]
        else:
            match_score = 0
        match_scores.append(match_score)
        regions.append(region)
    return match_scores, regions


def reorder_chessboard_corners(image, uvs, board_shape, template_size=40):
    """
    Reorder chessboard points using an anchor point. See
    py:func:`multicam_calibration.detection.detect_chessboard` for details.

    Parameters
    ----------
    image : array
        Image that the chessboard was detected in.

    uvs : array of shape (N,2)
        Coordinates of the chessboard points.

    board_shape : tuple (rows,columns)
        Shape of the chessboard (number of squares in each dimension).

    template_size : int, default=40
        Size of template to generate.

    Returns
    -------
    reordered_uvs: array of shape (N,2)
        Reordered coordinates of the chessboard points.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    uv_grid = uvs.reshape(board_shape[1], board_shape[0], 2)
    ext = extend_grid(uv_grid, 3, 1)

    all_source_pts = [
        np.float32([[ext[2, 0], ext[0, 0], ext[0, 2], ext[2, 2]]]),
        np.float32([[ext[0, -3], ext[0, -1], ext[2, -1], ext[2, -3]]]),
        np.float32([[ext[-3, -1], ext[-1, -1], ext[-1, -3], ext[-3, -3]]]),
        np.float32([[ext[-1, 2], ext[-1, 0], ext[-3, 0], ext[-3, 2]]]),
    ]

    s = 40  # size of anchor template
    template = _generate_chessboard_anchor_template(s)
    target_pts = np.float32([[0, s], [0, 0], [s, 0], [s, s]])
    match_scores, regions = _match_to_template(
        image, all_source_pts, target_pts, template
    )

    # Reorder points so best match is in top-left corner
    if np.argmax(match_scores) in [2, 3]:  # best match is in bottom row
        uv_grid = uv_grid[::-1, :]
    if np.argmax(match_scores) in [1, 2]:  # best match is in right column
        uv_grid = uv_grid[:, ::-1]

    uvs_reordered = uv_grid.reshape(-1, 2)
    sorted_match_scores = np.sort(match_scores)[::-1]
    vis_info = (all_source_pts, regions, template, match_scores)
    return uvs_reordered, sorted_match_scores, vis_info


def generate_chessboard_objpoints(chess_board_shape, chess_board_square_size):
    """
    Generate a list of 3D coordinates for the points on a chessboard
    in a standard orientation.

    The first point is assumed to be at (0,0,0) and the remaining points are
    spaced `square_size` apart in the x and y directions (with z=0).

    Parameters
    ----------
    chess_board_shape : tuple (rows,columns)
        Shape of the chessboard (number of squares in each dimension).

    square_size : float
        Size of a chessboard square (e.g. in mm).

    Returns
    -------
    objpoints : array of shape (N,3)
        Coordinates of the chessboard points.
    """
    rows, cols = chess_board_shape
    objpoints = np.zeros((rows * cols, 3), np.float32)
    objpoints[:, :2] = (
        np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * chess_board_square_size
    )
    return objpoints
