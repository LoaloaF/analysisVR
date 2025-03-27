import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import matplotlib.pyplot as plt
import numpy as np

from analytics_processing.analytics_constants import device_paths
from analytics_processing.modality_loading import session_modality_from_nas


final_nrows = 393
dtype = np.int16

n_tables = 3
n_rows = 1024
n_columns_list = [2, 3, 4]  # Different number of columns for each table
# dtype = np.uint16
to_path = './fuse_mount/'
from_path = '/home/loaloa/fuse_root/concatenated_ss/'
from_path = '/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/fuse_root5/rYL006_concat_ss/'

def create_test_tables():
    for i in range(n_tables):
        n_columns = n_columns_list[i]
        table_size = n_rows * n_columns
        
        data = np.arange(i*table_size, i*table_size + table_size, dtype=dtype).reshape(n_rows, n_columns)
        print('table', i)
        print(data)
        
        fullfname = f"{to_path}table_{i}.dat"
        # Save the raw data in F order
        with open(fullfname, 'wb') as f:
            data.flatten(order="F").tofile(f)
            
        # # Test read it again
        # with open(fullfname, 'rb') as f:
        #     data2 = np.fromfile(f, dtype=dtype).reshape(n_rows, n_columns, order="F")
        #     print(data2)
        #     print()

def test_read():
    for i in range(n_tables):
        fullfname = f"{from_path}table_{i}.dat"
        with open(fullfname, 'rb') as f:
            
            data = np.fromfile(f, dtype=dtype).reshape(n_rows, -1, order="F")
            print(data)
            print()
            
# read as memory map
def test_read_memmap():
    for i in range(n_tables):
        fullfname = f"{from_path}table_{i}.dat"
        data = np.memmap(fullfname, dtype=dtype, mode='r', shape=(n_rows, -1), order="F")
        print("Row 0")
        print(data[1])
        print() 
    
def test_read_concat_raw():
    # open file
    fullfname = f"{from_path}concat.dat"
    print(fullfname)
    print(os.path.exists(fullfname))
    # with open(fullfname, 'rb') as f:
    #     # read the first 3 rows of the first column (seek to 0, read 3*2 bytes)
    #     # f.seek()
    #     out = f.read(9*1024*2 +2)
    #     # convert to numpy array
    #     data = np.frombuffer(out, dtype=dtype)
    #     print(data)

    n_rows = 393
    
    # fname = "/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min/2024-11-11_16-11_rYL006_P1000_MotorLearningStop_25min_393_ephys_traces.dat"
    # data = np.memmap(fname, dtype=dtype, mode='r').reshape((n_rows, -1), order="F")
    # print(data.shape)
    # print(data[0,30289600-100:30289600])
    
    
    
    with open(fullfname, 'rb') as f:
        data = np.memmap(fullfname, dtype=dtype, mode='r').reshape((n_rows, -1), order="F")
        print(data.shape)
        
        # print(data)
        # d = (data[0,30289600-100:30289600+100])
        # print(d)
        # print(d.shape)
        # print(data[0,30289600-100:30289600+100])
        d = data[0]
        print(f"{d.shape[0]/20000 /60:,}")
        
        
        print()

def final_test():
    nas_dir = device_paths()[0]
    concat_fullfname = os.path.join(nas_dir, "fuse_root/rYL006_concat_ss/concat.dat")
    concat_data = np.memmap(concat_fullfname, dtype=dtype, mode='r').reshape((final_nrows, -1), order='F')

    f1_length = 17636800
    conc_data = concat_data[[10,100], f1_length-100:f1_length+100, ]

    f2_length = 27459000
    
    f1_conc = (concat_data[:, :f1_length])
    print(f1_conc)
    print("  ".join([f"{s:,}" for s in f1_conc.shape]))
    
    f2_conc = (concat_data[:, f1_length:f2_length])
    print(f2_conc)
    print("  ".join([f"{s:,}" for s in f2_conc.shape]))
    
    
    conc_data = concat_data[[10,100], f1_length-100:f1_length+100, ]
    print(conc_data.shape)
    plt.plot(conc_data[0], label='concat.dat Row=10', alpha=.2, color='blue', linewidth=4)
    plt.plot(conc_data[1], label='concat.dat Row=100', alpha=.2, color='pink', linewidth=4)
    
    s1_name = "/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-10-25_15-41_rYL006_P1000_MotorLearningStop_14min/2024-10-25_15-41_rYL006_P1000_MotorLearningStop_14min_393_ephys_traces.dat"
    s2_name = "/mnt/NTnas/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1000/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min/2024-11-04_16-13_rYL006_P1000_MotorLearningStop_22min_393_ephys_traces.dat"
    
    d1, mapping = session_modality_from_nas(s1_name, key='ephys_traces')
    print(d1.shape)
    d2, mapping = session_modality_from_nas(s2_name, key='ephys_traces', )
    print(d2.shape)
    
    plt.plot(d1[10, -100:], linestyle='--', color='blue', label='single.dat Row=10')
    plt.plot(d1[100, -100:], linestyle='--', color='pink', label='single.dat Row=100')
    plt.plot(np.arange(100,200), d2[10, :100], linestyle='--', color='blue')
    plt.plot(np.arange(100,200), d2[100, :100], linestyle='--', color='pink')
    
    plt.legend()
    
    
    plt.show()
    

    
if __name__ == '__main__':
    # create_test_tables()
    # test_read()
    # test_read_memmap()
    # test_read_concat_raw()
    final_test()