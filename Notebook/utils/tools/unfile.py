import tarfile
import os


def untar(tar_name, root):
    '''
    解压 tar.gz 文件并返回路径
    root：解压的根目录
    '''
    with tarfile.open(tar_name) as tar:
        tar.extractall(root)
        tar_root = tar.getnames()[0]
    return os.path.join(root, tar_root)