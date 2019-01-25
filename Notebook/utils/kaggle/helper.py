import os
import zipfile


def unzip(root, NAME):
    '''
    将从 Kaggle 下载的数据进行解压，并返回数据所在目录
    '''
    source_path = os.path.join(root, 'all.zip')
    dataDir = os.path.join(root, NAME)
    with zipfile.ZipFile(source_path) as fp:
        fp.extractall(dataDir) # 解压 all.zip 数据集到 dataDir
    os.remove(source_path) # 删除 all.zip
    return dataDir # 返回数据所在目录