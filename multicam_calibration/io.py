import cv2, os, json

from .geometry import get_transformation_matrix


def save_calibration(all_extrinsics, all_intrinsics, save_path, 
                     camera_names, save_format='json'):
    """
    Save calibration results.

    A range of formats are supported. In every case, the extrinsics are saved
    as 3x3 rotation matrices and 3x1 translation vectors. The intrinsics are
    saved as 3x3 camera matrices and 5x1 distortion coefficients. The available
    formats include

    - `json`: A single JSON file mapping camera names to calibration parameters.

    - `jarvis`: Directory with one file per camera in OpenCV-specific YAML format.

    Parameters
    ----------
    all_extrinsics : array of shape (n_cameras, 6)
        Transformations from the world coordinate system to each camera's
        coordinate system. The first three elements are the rotation vector
        and the last three elements are the translation vector.

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Intrinsic parameters for each camera (see 
        :py:func:`multicam_calibration.get_intrinsics`).

    save_path : str
        Path to save calibration. Interpreted as a file if `format=="YAML"`
        and as a directory if `format=="JARVIS"`. For files, dont include
        the file extension. For directories, the directory will be created
        if it doesn't exist.

    camera_names : list of str
        Names of the cameras. Must be the same length as `all_extrinsics`

    save_format : str, default='json'
        Format to save calibration. See above for options.
    """
    assert len(all_extrinsics) == len(all_intrinsics) == len(camera_names), \
        "Number of camera names must match number of extrinsics and intrinsics"
    
    transforms = get_transformation_matrix(all_extrinsics)
    
    if save_format == 'json':
        calibration_data = {}
        for i,camera_name in enumerate(camera_names):
            calibration_data[camera_name] = {
                'rotation': transforms[i,:3,:3].tolist(),
                'translation' : transforms[i,:3,3:].tolist(),
                'camera_matrix' : all_intrinsics[i][0].tolist(),
                'distortion_coefs' : all_intrinsics[i][1].tolist()}
        with open(f'{save_path}.json', 'w') as f:
            json.dump(calibration_data, f, indent=4)

    elif save_format == 'jarvis':
        if not os.path.exists(save_path): os.makedirs(save_path)
        for i,camera_name in enumerate(camera_names):
            filename = os.path.join(save_path, f'{camera_name}.yaml')
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
            fs.write("intrinsicMatrix", all_intrinsics[i][0])
            fs.write("distortionCoefficients", all_intrinsics[i][1].reshape(1,-1))
            fs.write("R", transforms[i,:3,:3])
            fs.write("T", transforms[i,:3,3:])
            fs.release()

    else:
        raise ValueError(f"Unknown format {save_format}")