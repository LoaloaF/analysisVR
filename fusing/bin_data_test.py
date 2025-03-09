import numpy as np
import os

n_tables = 3
n_rows = 1024
n_columns_list = [2, 3, 4]  # Different number of columns for each table
dtype = np.uint16
to_path = './fuse_mount/'
from_path = '/home/loaloa/fuse_root/concatenated_ss/'

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
        print(data[0])
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


    with open(fullfname, 'rb') as f:
        data = np.memmap(fullfname, dtype=dtype, mode='r', shape=(n_rows, sum(n_columns_list)), order="F")
        
        # print(data)
        print(data[0])
        print()


    
if __name__ == '__main__':
    # create_test_tables()
    # test_read()
    # test_read_memmap()
    test_read_concat_raw()