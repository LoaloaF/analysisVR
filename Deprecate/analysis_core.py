# import os
# import sys
# import platform

# # sys.path.insert(1, os.path.dirname(__file__)) # analysis VR main dir
# sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'data_loading')) # project dir
# # sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../CoreRatVR')) # project dir
# # sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../SessionWiseProcessing')) # project dir
# # sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../AnimalWiseProcessing')) # project dir

# from CustomLogger import CustomLogger as Logger

# def init_analysis(loglevel='INFO'):
#     Logger().init_logger(None, None, loglevel)

# # def _delfault_paths():
# #     which_os = platform.system()
# #     user = os.getlogin()
# #     print(f"OS: {which_os}, User: {user}")
    
# #     if which_os == 'Linux' and user == 'houmanjava':
# #         nas_dir = "/mnt/SpatialSequenceLearning/"
# #         local_data_dir = "/home/houmanjava/local_data/"
# #         project_dir = "/home/houmanjava/meatesting/"
    
# #     elif which_os == 'Linux' and user == 'vrmaster':
# #         nas_dir = "/mnt/SpatialSequenceLearning/"
# #         local_data_dir = "/home/vrmaster/local_data/"
# #         project_dir = "/home/vrmaster/Projects/VirtualReality/"
    
# #     elif which_os == "Darwin" and user == "root":
# #         nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/"
        
# #         folders = [f for f in os.listdir("/Users") if os.path.isdir(os.path.join("/Users", f))]

# #         if "loaloa" in folders:
# #             local_data_dir = "/Users/loaloa/local_data/analysisVR_cache"
# #             project_dir = "/Users/loaloa/homedataAir/phd/ratvr/VirtualReality/"
# #         elif "yaohaotian" in folders:
# #             local_data_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/data/analysisVR_cache"
# #             project_dir = "/Users/yaohaotian/Downloads/Study/BME/Research/MasterThesis/code/"
# #         else:
# #             raise ValueError("Unknown MacOS user")
# #     else:
# #         nas_dir, local_data_dir, project_dir = None, None, None
# #         raise ValueError("Unknown OS or user")
    
# #     if not os.path.exists(nas_dir):
# #         msg = f"NAS directory not found: {nas_dir} - VPN connected?"
# #         raise FileNotFoundError(msg)
# #     return nas_dir, local_data_dir, project_dir

# # NAS_DIR, LOCAL_DATA_DIR, PROJECT_DIR = _delfault_paths()

# PARADIGM_COLORS = {
#     0: "#ccccccff", # AutoLickReward
#     100: "#666666ff", # Spout Association

#     500: "#ff72a4ff", # Motor Learning
#     900: "#cc72ffff", # Motor Lick Learning
#     1000: "#7279ffff", # Motor Stop Learning

#     200: "#1577d6ff", # 2D arena 1 Pillar
#     400: "#2f15d6ff", # 2D arena 4 Pillars

#     800: "#0ee1beff", # 1D track, Cue1 Cue2
#     801: "#0ee136ff", # 1D track, Cue1 Cue2, Lick triggered
#     1100: "#d64215ff", # 1D track, first both rewarded, then Cue1 Cue2, Stop triggered
# }

# PARADIGM_VISIBLE_NAMES = {
#     0: "learn-Spout-Licking",
#     100: "learn-Sound->R-Association",
    
#     500: "learn-Ball-Control",
#     900: "learn-Lick-Triggers-Reward",
#     1000: "learn-Complete-Stopping",
    
#     200: "2D-Arena: One-Pillar",
#     400: "2D-Arena: Four-Pillars",
    
#     800: "1D-Track: slowing -> R1 or R2",
#     801: "1D-Track: lick -> R1 or R2",
#     1100: "1D-Track: stopping -> R1 and R2, then or",
# }