from fuse import FuseOSError, Operations, LoggingMixIn
import os
import logging
from errno import EACCES
from os.path import realpath
from threading import Lock
import time
import stat

class VirtualConcatFS(LoggingMixIn, Operations):
    def __init__(self, root, concat_fullfname='/rYL006_concat_ss/concat.dat', 
                 singlefiles_fullfnames=[]):
        self.root = realpath(root)
        self.rwlock = Lock()
        self.concat_fullfname = concat_fullfname
        self._singlefiles_fullfnames = singlefiles_fullfnames

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
        if full_path.endswith(self.concat_fullfname):
            total_size = self.get_concatfile_size()
            logging.info(f"\nRequested fstat for concat.dat - size set to: {total_size:,}")
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
        logging.info("\n\n----Requested:-----", )
        logging.info(f"{path=}, {size=:,}, {offset=:,}, {fh=}", )
        if path == self.concat_fullfname:
            return self.virtual_concat_read(size, offset, fh)
        
        with self.rwlock:
            os.lseek(fh, offset, 0)
            return os.read(fh, size)

    def readdir(self, path, fh):
        full_path = self._full_path(path)
        return ['.', '..'] + os.listdir(full_path)

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
        
    def get_singlefiles_fullfnames(self):
        return self._singlefiles_fullfnames
        
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
    
    def virtual_concat_read(self, size, offset, fh):
        full_fnames = self.get_singlefiles_fullfnames()
        sizes = self.get_singlefiles_sizes()
        blocksizes = self.get_singlefiles_blocks()
        cumul_sizes = self.get_singlefiles_cum_sizes()
        logging.info(f"Looping with over {len(full_fnames)} .dat files, "
                f"sizes:{sizes}, blocksizes:{blocksizes}, "
                f"cumulative_sizes:{cumul_sizes}", )
        
        aggr_content = b''
        first_file = True
        for i in range(len((full_fnames))):
            logging.info(f"File {i}...")
            if offset >= cumul_sizes[i]:
                logging.info(f"Offset outside of this files size cumul:{cumul_sizes[i]:,}...")
                continue
            
            with open(full_fnames[i], 'rb') as f:
                if first_file:
                    # move offset wrt to current file if i != 0
                    within_file_offset = offset - cumul_sizes[i-1] if i > 0 else offset
                else:
                    # continuous read over more than the first file
                    within_file_offset = 0
                logging.info(f"Within-offset: {within_file_offset:,}, general offset: {offset:,}")
                f.seek(within_file_offset)

                nbytes = 4096 # min: single block read
                if size > nbytes: 
                    nbytes = size # requested more than one block
                    # # end of file not needed? EOF handles this
                    # if nbytes > sizes[i]:
                    #     nbytes = blocksizes[i]
                    logging.info(f"Reading {nbytes} bytes from {os.path.basename(full_fnames[i])}, "
                            f"Total bytes requested: {size}, still needed: "
                            f"{size - len(aggr_content)}", )
                    
                aggr_content += f.read(nbytes) # read
                first_file = False
            logging.info(f"After this file, aggr_content: {len(aggr_content):,} size requested {size:,}", )
            if len(aggr_content) > size:
                # read to much, remember we always round up to blocksize
                logging.info("Sslicing...")
                aggr_content = aggr_content[:size]
            if len(aggr_content) == size:
                logging.info("MATCH!!")
                
                return aggr_content
        return aggr_content