import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../')) # project dir
sys.path.insert(1, os.path.join(sys.path[0], '../../CoreRatVR')) # project dir
sys.path.insert(1, os.path.join(sys.path[0], '../SessionWiseProcessing')) # project dir

import matplotlib.pyplot as plt

from sessionWiseProcessing.session_loading import get_session_modality

nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning"
animal_id = 1

paradigm_id = 200
paradigm_dir = f"RUN_rYL00{animal_id}/rYL{animal_id:03}_P{paradigm_id:04d}"
data = []
session_dirs = [sd for sd in os.listdir(os.path.join(nas_dir, paradigm_dir)) if sd.endswith("min")]
data2 = []
for i, session_name in enumerate(sorted(session_dirs, reverse=True)):
    print(session_name)
    if not session_name.endswith("min"):
        continue
    
    session_dir = os.path.join(paradigm_dir, session_name)
    
    
    frames = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
                                  modality="unity_frame", rename2oldkeys=True, us2s=True)
    
    frames = frames[(frames.trial_id > 0) & (frames.trial_id < 40)]
    plt.scatter(frames["X"], frames["Z"], c=frames["trial_id"] + 1, cmap='tab20', label=session_name, s=1)
    plt.colorbar(label='Trial ID')
    plt.show()