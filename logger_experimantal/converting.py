import os
import sys

from analytics_constants import device_paths

import h5py
import numpy as np  
import pandas as pd

import matplotlib.pyplot as plt

nas_dir = device_paths()[0]
session_path = os.path.join(nas_dir, 'RUN_rYL010/rYL010_P0000/2025-05-09_15-51_rYL010_P0000_AutoLickReward_164min')
data = np.memmap(os.path.join(session_path, "Rat_20250509.uint16"), dtype=np.int16).reshape((1024, -1), order='F')
print(data.shape)

out_fullfname = os.path.join(session_path, "2025-05-08_15-51_rYL010_P1100_FreelyMoving_3min_738_ephys_traces_mapping.dat")

mapping = pd.read_csv(os.path.join(session_path, "2025-05-08_15-51_rYL010_P1100_FreelyMoving_3min_738_ephys_traces_mapping.csv"))
print(mapping)

# plt.plot(data[:10, 1_000_000:1_000_000+1000].T)
# plt.savefig('test.png')
# plt.show()

open(out_fullfname, 'w').close() # this overwrite, careful


chunk_size = 1 * 20_000 # 1 minute of data
chunks = list(range(0, data.shape[1], chunk_size))
for chunk_idx in chunks:
    chunk_data = data[:, chunk_idx:chunk_idx+chunk_size]
    # print(chunk)
    
    chunk_data = chunk_data[mapping.amplifier_id.values, :]
    print(chunk_data.shape)
    
    with open(out_fullfname, 'ab') as f:
        chunk_data.T.flatten().tofile(f)
    print(f'Chunk {chunk_idx//chunk_size+1}/{len(chunks)} written to {os.path.basename(out_fullfname)}')
    # exit()