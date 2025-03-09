from fuse import FUSE, FuseOSError, Operations, LoggingMixIn
import os
import logging
from errno import EACCES
from os.path import realpath
from threading import Lock
import time
import stat

class Loopback(LoggingMixIn, Operations):
    def __init__(self, root):
        self.root = realpath(root)
        self.rwlock = Lock()
        self.singlefiles_prefix = 'table_'
        self.singlefiles_suffix = '.dat'
        self.output_path = 'concatenated_ss' 

    def _full_path(self, partial):
        if partial.startswith("/"):
            partial = partial[1:]
        path = os.path.join(self.root, partial)
        return path

    def access(self, path, mode):
        full_path = self._full_path(path)
        if not os.access(full_path, mode):
            raise FuseOSError(EACCES)

    chmod = os.chmod
    chown = os.chown

    def create(self, path, mode):
        full_path = self._full_path(path)
        return os.open(full_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)

    def flush(self, path, fh):
        return os.fsync(fh)

    def fsync(self, path, datasync, fh):
        if datasync != 0:
            return os.fdatasync(fh)
        else:
            return os.fsync(fh)

    def getattr(self, path, fh=None):
        full_path = self._full_path(path)
        if os.path.basename(full_path) == 'concat.dat':
            total_size = self.get_concatfile_size()
            return {
                'st_atime': time.time(),
                'st_mtime': time.time(),
                'st_ctime': time.time(),
                'st_mode': stat.S_IFREG | 0o644,
                'st_nlink': 1,
                'st_size': total_size,
                'st_uid': os.getuid(),
                'st_gid': os.getgid(),
            }
            
        st = os.lstat(full_path)
        return dict((key, getattr(st, key)) for key in (
            'st_atime', 'st_ctime', 'st_gid', 'st_mode', 'st_mtime',
            'st_nlink', 'st_size', 'st_uid'))

    getxattr = None

    def link(self, target, source):
        full_target = self._full_path(target)
        full_source = self._full_path(source)
        return os.link(full_source, full_target)

    listxattr = None
    mkdir = os.mkdir
    mknod = os.mknod

    def open(self, path, flags):
        full_path = self._full_path(path)
        return os.open(full_path, flags)

    def read(self, path, size, offset, fh):
        print("------", flush=True)
        print(path, size, offset, fh, type(fh), flush=True)
        if path == '/concatenated_ss/concat.dat':
            # full_fnames = [os.path.join(self._full_path('/'), fn) 
            #                for fn in self.get_singlefiles_fnames()]
            full_fnames = self.get_singlefiles_fullfnames()
            sizes = self.get_singlefiles_sizes()
            blocksizes = self.get_singlefiles_blocks()
            cumul_sizes = self.get_singlefiles_cum_sizes()
            print("looping with ", full_fnames, sizes, blocksizes, 
                  cumul_sizes, flush=True)
            
            aggr_content = b''
            first_file = True
            for i in range(len((full_fnames))):
                if offset >= cumul_sizes[i]:
                    continue
                
                with open(full_fnames[i], 'rb') as f:
                    f.seek(offset if first_file else 0) # needs to be converted to singlefile bytecount
                    nbytes = 4096 # min
                    if size > nbytes:
                        nbytes = size # read more than one block, round to nearset block up?
                        
                        # end of file?
                        if nbytes > sizes[i]:
                            nbytes = blocksizes[i]
                    print("Reading ", nbytes, " bytes from ", os.path.basename(full_fnames[i]), 
                          'with offset ', offset if first_file else 0, "Total bytes requested: ", size, flush=True)
                    aggr_content += f.read(nbytes)
                    first_file = False
                print("After this file, aggr_content:", len(aggr_content), 'size requested', size)
                if len(aggr_content) > size:
                    # read to much, remeber we always round up to blocksize
                    aggr_content = aggr_content[:size]
                if len(aggr_content) == size:
                    return aggr_content
            
            # print("Calling with ", full_fnames[0], flush=True)
            #         # Open the first file in the list (e.g., table_0.dat)
            # with open(full_fnames[0], 'rb') as f:
            #     f.seek(offset)
            #     file1_data = f.read(4096)
            # with open(full_fnames[1], 'rb') as f:
            #     f.seek(offset)
            #     file2_data = f.read(4096)
            print("Before returning", flush=True)
            return aggr_content
            # return "Teststring".encode('utf-8')[offset:offset + size]
      
      
      
        
        with self.rwlock:
            os.lseek(fh, offset, 0)
            return os.read(fh, size)

    def readdir(self, path, fh):
        full_path = self._full_path(path)
        return ['.', '..'] + os.listdir(full_path)

    def get_singlefiles_fullfnames(self):
        full_path = self._full_path('/')
        matching_files = []
        
        for root, dirs, files in os.walk(full_path):
            for fn in files:
                if fn.startswith(self.singlefiles_prefix) and fn.endswith(self.singlefiles_suffix):
                    matching_files.append(os.path.join(root, fn))
        
        print('\n')
        print(matching_files)
        print('\n', flush=True)
        
        return matching_files
    
    # def get_singlefiles_fnames(self):
    #     full_path = self._full_path('/')
    #     return [fn for fn in os.listdir(full_path)
    #             if fn.startswith(self.singlefiles_prefix) and fn.endswith(self.singlefiles_suffix)]
        
    def get_singlefiles_sizes(self):
        return [os.stat(fn).st_size for fn in self.get_singlefiles_fullfnames()]
        
    def get_singlefiles_blocks(self):
        return [os.stat(fn).st_blocks * 512 for fn in self.get_singlefiles_fullfnames()]
        
    def get_concatfile_size(self):
        return sum(self.get_singlefiles_sizes())
    
    def get_singlefiles_cum_sizes(self):
        cum_sizes = [0]
        for size in self.get_singlefiles_sizes():
            cum_sizes.append(cum_sizes[-1] + size)
        return cum_sizes[1:]
    
    # def get_datfile_size_mapping(self):
    #     # return {full_fn: os.path.getsize(full_fn) for full_fn in self.get_fnames_collection()}
    #     return {full_fn: os.stat(full_fn).st_blocks * 512 for full_fn in self.get_fnames_collection()}
    
    
    
    
    
    
    readlink = os.readlink

    def release(self, path, fh):
        return os.close(fh)

    def rename(self, old, new):
        full_old = self._full_path(old)
        full_new = self._full_path(new)
        return os.rename(full_old, full_new)

    rmdir = os.rmdir

    def statfs(self, path):
        full_path = self._full_path(path)
        stv = os.statvfs(full_path)
        return dict((key, getattr(stv, key)) for key in (
            'f_bavail', 'f_bfree', 'f_blocks', 'f_bsize', 'f_favail',
            'f_ffree', 'f_files', 'f_flag', 'f_frsize', 'f_namemax'))

    def symlink(self, target, source):
        full_target = self._full_path(target)
        full_source = self._full_path(source)
        return os.symlink(full_source, full_target)

    def truncate(self, path, length, fh=None):
        full_path = self._full_path(path)
        with open(full_path, 'r+') as f:
            f.truncate(length)

    unlink = os.unlink
    utimens = os.utime

    def write(self, path, data, offset, fh):
        with self.rwlock:
            os.lseek(fh, offset, 0)
            return os.write(fh, data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mount')
    parser.add_argument('root')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    fuse = FUSE(Loopback(args.root), args.mount, nothreads=True, foreground=True)