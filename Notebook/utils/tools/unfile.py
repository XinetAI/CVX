import tarfile
import os


def untar(tar_name, root):
    '''
    解压 tar.gz 文件并返回路径
    root：解压的根目录
    '''
    with tarfile.open(tar_name) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner) 
            
        
        safe_extract(tar, root)
        tar_root = tar.getnames()[0]
    return os.path.join(root, tar_root)