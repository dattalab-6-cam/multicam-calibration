import numpy as np
import cv2

na = np.newaxis


def detect_uvs(image, board_shape):
    """
    Detect (u,v) coordinates of chessboard points in an image.

    Parameters
    ----------
    image : array
        Image to detect the chessboard in.

    board_shape : tuple (rows,columns)
        Shape of the chessboard (number of squares in each dimension), e.g. the
        following board would have shape (3,5)::

            ██░░██░░██
            ░░██░░██░░
            ██░░██░░██

    Returns
    -------
    uvs: array of shape (N,2) or None
        Either the coordinates of the interior points of the chessboard or None
        if no chessboard was detected.
    """
    pass


def reorder_uvs(image, uvs):
    """
    Reorder the points detected by `detect_chessboard_uvs`

    This function ensures that points are indexed consistently across cameras.
    The chessboard is assumed to have a dark circle in top-left corner when
    viewed from the font. Points are ordered such that the point nearest the
    dark circle is indexed first, followed by the others in row-major order.::

        ●
        ██░░██░░██
        ░░██░░██░░
        ██░░██░░██


    Parameters
    ----------
    image : array
        Image that the chessboard was detected in.

    uvs : array of shape (N,2)
        Coordinates of the chessboard points.

    Returns
    -------
    reordered_uvs: array of shape (N,2)
        Reordered coordinates of the chessboard points.
    """
    pass


def generate_objpoints(chess_board_shape, chess_board_square_size):
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
