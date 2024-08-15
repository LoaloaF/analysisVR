import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = '/home/ntgroup/Project/data/2024-08-14_14-31_dummyAnimal_P0800_LinearTrack_3min/'
fname = "ephys_output.raw.h5"

fullfname = os.path.join(folder, fname)

with h5py.File(fullfname, 'r') as file:
    data = file['bits']["0000"][:]
# convert the value into single bits and create time series
msg = np.array([(a[0],a[1]) for a in data])


df_bit = pd.DataFrame(msg, columns=['time', 'value'])

for column_id in range(8):
    df_bit[f'bit{column_id}'] = (df_bit['value'] & (1 << column_id))/2**column_id
    df_bit[f'bit{column_id}'] = df_bit[f'bit{column_id}'].astype(int)

print(msg.shape)