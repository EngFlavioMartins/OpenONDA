import os
import glob
import json
import numpy
import pyvista

def remove_files(directory:str, filesnames: str):
    """ 
    Routine to delete files in a directory (string) that follow a certain pattern (string).
    
    Example:
        directory = "./"
        filesnames = "myfile*"
        This code will iteratively delete the files "./myfile*" from your system
    """

    pattern = filesnames + "_*"
    files_to_delete = glob.glob(os.path.join(directory, pattern))

    # Loop through and delete each file
    n_files_deleted = 0
    for file in files_to_delete:
        try:
            os.remove(file)
            n_files_deleted += 1
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    print(f"(info) Deleted {n_files_deleted} files with pattern '{directory}/{pattern}' ")



def vtp_series_to_json(directory:str, filesnames: str, json_filename:str, time_step: float):
    """
    Routine to create a json file to help ParaView (or similar tools) to interpret the data time-series

    Arguments:
    ----------
    file_pattern: str
        Defines the pattern of the data, it should be in the form "filenames*.vtp"
    json_filename: str
        Defines the name of the json file
    time_step: float
        Defines the time-step of the simulation (s)
    """
    # Split the input string by commas to get individual files and times
    pattern = filesnames + "_*"
    entries_pattern = sorted(glob.glob(os.path.join(directory, pattern)))

    # Filter out directories, keeping only files
    files_only_pattern = [os.path.basename(entry) for entry in entries_pattern]
    
    # Initialize the dictionary structure
    vtk_json = {
        "file-series-version": "1.0",
        "files": []
    }

    # Iterate through the entries and add each to the JSON structure
    for e, entry in enumerate(files_only_pattern):
        # Split the file name and time by colon
        vtk_json['files'].append({
            "name": entry,
            "time": float(e*time_step)
        })
    
    json_file_dir = os.path.join(directory, json_filename + ".vtp.series")

    # Write the JSON structure to a file
    with open(json_file_dir, 'w') as json_file:
        json.dump(vtk_json, json_file, indent=4)


def import_from_vtp(filename: str):
    """
    Import the particle data from a VTP file.

    Arguments:
    ----------
    filename : str
        The name of the VTP file to read.

    Returns:
    --------
    positions : numpy.ndarray
        Array of particle positions.
    velocities : numpy.ndarray
        Array of particle velocities.
    strengths : numpy.ndarray
        Array of particle strengths.
    radii : numpy.ndarray
        Array of particle radii.
    """
    # Load the VTP file
    point_cloud = pyvista.read(filename)

    # Extract positions (points), velocities, strengths, and radii from the point cloud
    positions  = numpy.array(point_cloud.points, dtype=numpy.float64)
    velocities = numpy.array(point_cloud.point_data['Velocity'], dtype=numpy.float64)
    strengths  = numpy.array(point_cloud.point_data['Strength'], dtype=numpy.float64)
    radii      = numpy.array(point_cloud.point_data['Radius'], dtype=numpy.float64)

    return positions, velocities, strengths, radii
