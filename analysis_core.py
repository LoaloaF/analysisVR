import os
import sys
import platform

# sys.path.insert(1, os.path.dirname(__file__)) # analysis VR main dir
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'data_loading')) # project dir
# sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../CoreRatVR')) # project dir
# sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../SessionWiseProcessing')) # project dir
# sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../AnimalWiseProcessing')) # project dir

from CustomLogger import CustomLogger as Logger

def init_analysis(loglevel='INFO'):
    Logger().init_logger(None, None, loglevel)
    
def _delfault_paths():
    which_os = platform.system()
    user = os.getlogin()
    print(f"OS: {which_os}, User: {user}")
    
    if which_os == 'Linux' and user == 'houmanjava':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/houmanjava/local_data/"
        project_dir = "/home/houmanjava/meatesting/"
    
    elif which_os == 'Linux' and user == 'vrmaster':
        nas_dir = "/mnt/SpatialSequenceLearning/"
        local_data_dir = "/home/vrmaster/local_data/"
        project_dir = "/home/vrmaster/Projects/VirtualReality/"
    
    elif which_os == "Darwin" and user == "root":
        nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
        local_data_dir = "/Users/loaloa/local_data/analysisVR_cache"
        project_dir = "/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/"
    
    else:
        nas_dir, local_data_dir, project_dir = None, None, None
        raise ValueError("Unknown OS or user")
    
    if not os.path.exists(nas_dir):
        msg = f"NAS directory not found: {nas_dir} - VPN connected?"
        raise FileNotFoundError(msg)
    return nas_dir, local_data_dir, project_dir

NAS_DIR, LOCAL_DATA_DIR, PROJECT_DIR = _delfault_paths()