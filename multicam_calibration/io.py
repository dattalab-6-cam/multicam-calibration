import cv2, os, json
import numpy as np

from .geometry import get_transformation_matrix, rodrigues_inv


def save_calibration(all_extrinsics, all_intrinsics, save_path, 
                     camera_names, save_format='json'):
    """
    Save calibration results.

    Several formats are supported. In every case, the extrinsics are saved
    as 3x3 rotation matrices and 3x1 translation vectors that map from world
    coordinates to each camera's coordinates. The intrinsics are saved as
    3x3 camera matrices and distortion coefficients `k1, k2, p1, p2, k3`. 
    The available formats include:

    - `json`: Single JSON file mapping camera names to calibration parameters.

    - `jarvis`: Directory with one file per camera in OpenCV-specific YAML format.
      The rotation and camera matrices are transposed relative to the JSON format.

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
        Path to save calibration. Interpreted as a file if `format=="json"`
        and as a directory if `format=="jarvis"`. 

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
                'R': transforms[i,:3,:3].tolist(),
                'T' : transforms[i,:3,3:].tolist(),
                'camera_matrix' : all_intrinsics[i][0].tolist(),
                'distortion_coefs' : all_intrinsics[i][1].tolist()}
        if not save_path.endswith('.json'): save_path += '.json'
        with open(save_path, 'w') as f:
            json.dump(calibration_data, f, indent=4)

    elif save_format == 'jarvis':
        if not os.path.exists(save_path): os.makedirs(save_path)
        for i,camera_name in enumerate(camera_names):
            filename = os.path.join(save_path, f'{camera_name}.yaml')
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
            fs.write("intrinsicMatrix", all_intrinsics[i][0].T)
            fs.write("distortionCoefficients", all_intrinsics[i][1].reshape(1,-1))
            fs.write("R", transforms[i,:3,:3].T)
            fs.write("T", transforms[i,:3,3:])
            fs.release()

    else:
        raise ValueError(f"Unknown format {save_format}")
    

def load_calibration(load_path, camera_names=None, load_format='json'):
    """
    Load calibration results.

    A range of formats are supported. In every case, the extrinsics are loaded
    as 6-vectors where the first three elements are a rotation vector and the
    last three elements are a translation vector. The intrinsics are loaded as 
    3x3 camera matrices and 5x1 distortion coefficients. The available
    formats include

    - `json`: A single JSON file mapping camera names to calibration parameters.

    - `jarvis`: Directory with one file per camera in OpenCV-specific YAML format. 
        The rotation and camera matrices are transposed relative to the JSON format. 

    Parameters
    ----------
    load_path : str
        Path to calibration. Interpreted as a file if `format=="json"`
        and as a directory if `format=="jarvis"`.

    camera_names : list of str
        Desired camera order. Can all be used to only load a subset of 
        cameras. If `camera_names==None` the order will be alphabetical.

    load_format : str, default='json'
        Format to load calibration. See above for options.

    Returns
    -------
    all_extrinsics : array of shape (n_cameras, 6)
        Extrinsics for each camera (see above for format).

    all_intrinsics : list of tuples (camera_matrix, dist_coefs)
        Intrinsic parameters for each camera (see above for format).

    camera_names : list of str
        Names of the cameras, in an order that matches `all_extrinsics` and
        `all_intrinsics`.
    """
    if load_format == 'json':
        with open(load_path, 'r') as f:
            calibration_data = json.load(f)
        if camera_names is None:
            camera_names = sorted(calibration_data.keys())
        else:
            assert set(camera_names) == set(calibration_data.keys()), \
                "Camera names must match keys in calibration file"

        all_extrinsics,all_intrinsics = [],[]
        for camera_name in camera_names:
            all_extrinsics.append(np.concatenate([
                rodrigues_inv(calibration_data[camera_name]['rotation']),
                np.array(calibration_data[camera_name]['translation'])]))
            all_intrinsics.append((
                np.array(calibration_data[camera_name]['camera_matrix']),
                np.array(calibration_data[camera_name]['distortion_coefs'])))
        return all_extrinsics, all_intrinsics, camera_names


    elif load_format == 'jarvis':
        yaml_files = [f for f in sorted(os.listdir(load_path)) if os.path.splitext(f)[1] in ['.yaml','.YAML']]
        names_to_files = {os.path.splitext(f)[0] : f for f in yaml_files}
        if camera_names is None:
            camera_names = sorted(names_to_files.keys())
        else:
            assert set(camera_names) <= set(names_to_files.keys()), \
                "Camera names must be a subset of yaml files in calibration directory"

        all_extrinsics,all_intrinsics = [],[]
        for camera_name in camera_names:
            filepath = os.path.join(load_path, names_to_files[camera_name])
            fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
            all_extrinsics.append(np.concatenate([
                rodrigues_inv(fs.getNode("R").mat().T),
                fs.getNode("T").mat().squeeze()]))
            all_intrinsics.append((
                fs.getNode("intrinsicMatrix").mat().T,
                fs.getNode("distortionCoefficients").mat().squeeze()))
        return all_extrinsics, all_intrinsics, camera_names

    else:
        raise ValueError(f"Unknown format {load_format}")