import os
import gzip
import struct
import tarfile
import pickle
import shutil

import numpy as np
import tables as tb

from .timerx import timer


def extractall(tar_name, root):
    '''
    解压 tar 文件并返回路径
    root：解压的根目录
    '''
    with tarfile.open(tar_name) as tar:
        tar.extractall(root)
        tar_root = tar.getnames()[0]
    return os.path.join(root, tar_root)


class MNIST(dict):
    def __init__(self, root, namespace, *args, **kw):
        """
        (MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist)
        (A dataset of Zalando's article images consisting of fashion products,
        a drop-in replacement of the original MNIST dataset from https://github.com/zalandoresearch/fashion-mnist)

        Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

        Parameters
        ----------
        root : 数据根目录，如 'E:/Data/Zip/'
        namespace : 'mnist' or 'fashion_mnist'
        """
        super().__init__(*args, **kw)
        self.__dict__ = self
        if namespace == 'mnist':
            self.url = 'http://yann.lecun.com/exdb/mnist'
            self.label_names = tuple(range(10))
        elif namespace == 'fashion_mnist':
            self.url = 'https://github.com/zalandoresearch/fashion-mnist'
            self.label_names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                                'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                                'Ankle boot')
        self.namespace = os.path.join(root, namespace)
        self._dataset(self.namespace)

    def _get_data(self, root, _train):
        '''
        官方网站的数据是以 `[offset][type][value][description]` 的格式封装的，因而 `struct.unpack` 时需要注意
        _train : bool, default True
            Whether to load the training or testing set.
        '''
        _train_data = os.path.join(root, 'train-images-idx3-ubyte.gz')
        _train_label = os.path.join(root, 'train-labels-idx1-ubyte.gz')
        _test_data = os.path.join(root, 't10k-images-idx3-ubyte.gz')
        _test_label = os.path.join(root, 't10k-labels-idx1-ubyte.gz')
        if _train:
            data, label = _train_data, _train_label
        else:
            data, label = _test_data, _test_label

        with gzip.open(label, 'rb') as fin:
            struct.unpack(">II", fin.read(8))
            label = np.frombuffer(fin.read(), dtype='B').astype('int32')

        with gzip.open(data, 'rb') as fin:
            Y = struct.unpack(">IIII", fin.read(16))
            data = np.frombuffer(fin.read(), dtype=np.uint8)
            data = data.reshape(Y[1:])
        return data, label

    def _dataset(self, root):
        self.trainX, self.trainY = self._get_data(root, True)
        self.testX, self.testY = self._get_data(root, False)


class Cifar(dict):
    def __init__(self, root, namespace, *args, **kwds):
        """CIFAR image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html

        Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

        Parameters
        ----------
        meta : 保存了类别信息
        root : str, 数据根目录
        namespace : 'cifar-10' 或 'cifar-100'
        """
        super().__init__(*args, **kwds)
        self.__dict__ = self
        self.url = 'https://www.cs.toronto.edu/~kriz/cifar.html'
        self.namespace = namespace
        self._read_batch(root)

    def _get_dataset(self, root):
        dataset = {}
        tar_name = os.path.join(root, f'{self.namespace}-python.tar.gz')
        tar_root = extractall(tar_name, root)
        for name in os.listdir(tar_root):
            k = name.split('/')[-1]
            path = os.path.join(tar_root, name)
            if name.startswith('data_batch') or name.startswith(
                    'test') or name.startswith('train'):
                with open(path, 'rb') as fp:
                    dataset[k] = pickle.load(fp, encoding='bytes')
            elif name.endswith('meta'):
                with open(path, 'rb') as fp:
                    dataset['meta'] = pickle.load(fp)
        shutil.rmtree(tar_root)
        return dataset

    def _read_batch(self, root):
        _dataset = self._get_dataset(root)
        if self.namespace == 'cifar-10':
            self.trainX = np.concatenate([
                _dataset[f'data_batch_{i}'][b'data'] for i in range(1, 6)
            ]).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            self.trainY = np.concatenate([
                np.asanyarray(_dataset[f'data_batch_{i}'][b'labels'])
                for i in range(1, 6)
            ])
            self.testX = _dataset['test_batch'][b'data'].reshape(
                -1, 3, 32, 32).transpose((0, 2, 3, 1))
            self.testY = np.asanyarray(_dataset['test_batch'][b'labels'])
            self.label_names = _dataset['meta']['label_names']
        elif self.namespace == 'cifar-100':
            self.trainX = _dataset['train'][b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            self.train_fine_labels = np.asanyarray(
                _dataset['train'][b'fine_labels'])  # 子类标签
            self.train_coarse_labels = np.asanyarray(
                _dataset['train'][b'coarse_labels'])  # 超类标签
            self.testX = _dataset['test'][b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            self.test_fine_labels = np.asanyarray(
                _dataset['test'][b'fine_labels'])  # 子类标签
            self.test_coarse_labels = np.asanyarray(
                _dataset['test'][b'coarse_labels'])  # 超类标签
            self.fine_label_names = _dataset['meta']['fine_label_names']
            self.coarse_label_names = _dataset['meta']['coarse_label_names']


class DataBunch(dict):
    def __init__(self, root, save_dir, *args, **kwds):
        '''
        封装 Cifar10、Cifar100、MNIST、Fashion MNIST 数据集
        '''
        super().__init__(*args, **kwds)
        self.__dict__ = self
        self.mnist = MNIST(root, 'mnist')
        self.fashion_mnist = MNIST(root, 'fashion_mnist')
        self.cifar10 = Cifar(root, 'cifar-10')
        self.cifar100 = Cifar(root, 'cifar-100')
        self.bunch2hdf5(save_dir)

    @timer
    def bunch2hdf5(self, h5_root):
        filters = tb.Filters(complevel=7, shuffle=False)
        # 这里我采用了压缩表，因而保存为 `.h5c` 但也可以保存为 `.h5`
        with tb.open_file(os.path.join(h5_root, 'X.h5'), 'w', filters=filters,
                title='Xinet\'s dataset') as h5:
            for name in self.keys():
                h5.create_group('/', name, title=f'{self[name].url}')
                h5.create_array(
                    h5.root[name],
                    'trainX',
                    self[name].trainX,
                    title='train X')
                h5.create_array(
                    h5.root[name], 'testX', self[name].testX, title='test X')
                if name != 'cifar100':
                    h5.create_array(
                        h5.root[name],
                        'trainY',
                        self[name].trainY,
                        title='train Y')
                    h5.create_array(
                        h5.root[name],
                        'testY',
                        self[name].testY,
                        title='test Y')
                    h5.create_array(
                        h5.root[name],
                        'label_names',
                        self[name].label_names,
                        title='标签名称')
                else:
                    h5.create_array(
                        h5.root[name],
                        'train_coarse_labels',
                        self[name].train_coarse_labels,
                        title='train_coarse_labels')
                    h5.create_array(
                        h5.root[name],
                        'test_coarse_labels',
                        self[name].test_coarse_labels,
                        title='test_coarse_labels')
                    h5.create_array(
                        h5.root[name],
                        'train_fine_labels',
                        self[name].train_fine_labels,
                        title='train_fine_labels')
                    h5.create_array(
                        h5.root[name],
                        'test_fine_labels',
                        self[name].test_fine_labels,
                        title='test_fine_labels')
                    h5.create_array(
                        h5.root[name],
                        'coarse_label_names',
                        self[name].coarse_label_names,
                        title='coarse_label_names')
                    h5.create_array(
                        h5.root[name],
                        'fine_label_names',
                        self[name].fine_label_names,
                        title='fine_label_names')


def copy_hdf5(path, topath='D:/temp/datasets.h5'):
    '''
    path:: HDF5 文件所在路径
    topath:: 副本的路径
    '''
    with tb.open_file(path) as h5:
        h5.copy_file(topath, overwrite=True)