import h5py
import numpy as np  
import pandas as pd

session_fullfname = '/home/vrmaster/logger/Rat_20250509_1300_deflate1.h5'
with h5py.File(session_fullfname, 'r') as session_file:
    data = np.array(session_file['dataset1'][:-20_000, :], dtype=np.int16)

print(data[:, -1000:])
print(data.shape)

mapping = pd.read_csv('/home/vrmaster/logger/2025-04-04_18-15_rYL010_P1100_LinearTrackStop_5min_738_ephys_traces_mapping.csv')
print(mapping)

data = data[:, mapping.amplifier_id.values]
# print(data[:, -20_000:])
print(data.shape)


out_fullfname = '/home/vrmaster/logger/2025-04-04_18-15_rYL010_P1100_LinearTrackStop_5min_738_ephys_traces.dat'
open(out_fullfname, 'w').close() # this overwrite, careful
with open(out_fullfname, 'ab') as f:
    # data.flatten(order="F").tofile(f)
    data.flatten().tofile(f)


# o = np.memmap(out_fullfname, dtype=np.int16, mode='r').reshape(len(mapping), -1, order='F')
# print(o.shape)