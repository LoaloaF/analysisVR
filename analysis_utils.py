import ast
import pandas as pd
import platform 
import os

# def join_indices(index1, index2):
#     # Ensure both indices have the same length
#     if len(index1) != len(index2):
#         raise ValueError("Both indices must have the same length")
    
#     # Check if the second index is a MultiIndex
#     if isinstance(index2, pd.MultiIndex):
#         # Combine the tuples from both indices
#         combined_tuples = [tuple1 + tuple2 for tuple1, tuple2 in zip(index1, index2)]
#         # Combine the names from both indices
#         combined_names = index1.names + index2.names
#     else:
#         # Combine the tuples from the first index with the values from the second index
#         combined_tuples = [tuple1 + (value2,) for tuple1, value2 in zip(index1, index2)]
#         # Combine the names from the first index with the name of the second index
#         combined_names = index1.names + [index2.name]
    
#     # Create a new MultiIndex with the combined tuples and names
    
#     return pd.MultiIndex.from_tuples(combined_tuples, names=combined_names)

# def str2list(string):
#     """
#     Convert a string representation of a list to an actual list.

#     Parameters:
#     - string (str): The string representation of the list.

#     Returns:
#     - list: The actual list.
#     """
#     try:
#         # Use ast.literal_eval to safely evaluate the string
#         result = ast.literal_eval(string)
#         if isinstance(result, list):
#             return result
#         else:
#             raise ValueError("The provided string does not represent a list.")
#     except (ValueError, SyntaxError) as e:
#         raise ValueError(f"Invalid string representation of a list: {string}") from e

def device_paths():
    which_os = platform.system()
    user = os.getlogin()
    # print(f"OS: {which_os}, User: {user}")
    
    if which_os == 'Linux' and user == 'houmanjava':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/houmanjava/local_data/"
        project_dir = "/home/houmanjava/meatesting/"
    
    elif which_os == 'Linux' and user == 'vrmaster':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        project_dir = "/home/vrmaster/Projects/VirtualReality/"
    
    elif which_os == 'Linux' and user == 'loaloa':
        nas_dir = "/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        project_dir = "/home/loaloa/homedataXPS/projects/ratvr/VirtualReality/"
    
    elif which_os == "Darwin" and user == "root":
        nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
        folders = [f for f in os.listdir("/Users") if os.path.isdir(os.path.join("/Users", f))]

        if "loaloa" in folders:
            local_data_dir = "/Users/loaloa/local_data/analysisVR_cache"
            project_dir = "/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/"
        elif "yaohaotian" in folders:
            local_data_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/data/analysisVR_cache"
            project_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/"
        else:
            raise ValueError("Unknown MacOS user")
    
    else:
        nas_dir, local_data_dir, project_dir = None, None, None
        raise ValueError("Unknown OS or user: ", which_os, user)
    
    if not os.path.exists(nas_dir):
        msg = f"NAS directory not found: {nas_dir} - VPN connected?"
        raise FileNotFoundError(msg)
    return nas_dir, local_data_dir, project_dir