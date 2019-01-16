import os
import pickle
import shutil
from sklearn.model_selection import train_test_split


def read_file(path):
    '''
    读取 string 文件
    '''
    with open(path, "rb") as fp:
        content = fp.read()
    return content.decode()  # 转换为 string


def save_file(save_path, content):
    '''
    保存 string 文件
    '''
    with open(save_path, 'wb') as fp:
        fp.write(content.encode())


def load_file(path):
    with open(path, 'rb') as fp:
        bunch = pickle.load(fp)
    return bunch


def dump(path, bunch):
    with open(path, 'wb') as fp:
        pickle.dump(bunch, fp)


def make_dir(root, dir_name):
    '''
    在 root 下生成目录
    '''
    _dir = root + dir_name + "/"  # 拼出分完整目录名
    if not os.path.exists(_dir):  # 是否存在目录，如果没有创建
        os.makedirs(_dir)
    return _dir


def get_dir_names(root):
    dir_names = []
    for k in os.listdir(root):
        if os.path.isdir(root + k):  # 判断是否是目录
            dir_names.append(root + k)
    return dir_names

def copyfile(original_dir, obj_dir, fnames):
    '''
    将 original_dir 目录下的 fnames 复制到 obj_dir
    '''
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(obj_dir, fname)
        shutil.copyfile(src, dst)    