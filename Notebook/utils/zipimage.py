import zipfile
import os
import json
from math import ceil

import numpy as np
import pandas as pd
import cv2

from .tools.timerx import timer


def get_path2shape(dataset):
    '''
    获取所有图片的 shape
    '''
    p2size = {}
    for img, label in dataset:
        p2size[label] = img.shape
    return p2size


class ImageZ:
    '''
    Working with compressed files under the images
    '''

    def __init__(self, root, dataType):
        '''
        root:: root dir
        dataType in ['test2014', 'test2015',
                    'test2017', 'train2014',
                    'train2017', 'unlabeled2017',
                    'val2014', 'val2017']
        '''
        self.Z = self.__get_Z(root, dataType)
        self.names = self.__get_names(self.Z)

    @staticmethod
    def __get_Z(root, dataType):
        '''
        Get the file name of the compressed file under the images
        '''
        dataType = dataType + '.zip'
        return zipfile.ZipFile(os.path.join(root, dataType))

    @staticmethod
    def __get_names(Z):
        names = []
        for name in Z.namelist():
            if not name.endswith('/'):
                names.append(name)
        return names

    def buffer2array(self, image_name):
        '''
        Get picture data directly without decompression

        Parameters
        ===========
        Z:: Picture data is a ZipFile object
        '''
        buffer = self.Z.read(image_name)
        image = np.frombuffer(buffer, dtype="B")  # 将 buffer 转换为 np.uint8 数组
        img_cv = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
        assert len(img_cv.shape) >= 2, "图片至少有两个维度"
        if len(img_cv.shape) == 3:
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # BGR 格式转换为 RGB
        else:
            return img_cv

    def __getitem__(self, item):
        names = self.names[item]
        if isinstance(item, slice):
            return [self.buffer2array(name) for name in names]
        else:
            return self.buffer2array(names)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        for name in self.names:
            yield self.buffer2array(name)


class AnnZ(dict):
    '''
    Working with compressed files under annotations
    '''

    def __init__(self, root, annType, *args, **kwds):
        '''
        dataType in [
              'annotations_trainval2014',
              'annotations_trainval2017',
              'image_info_test2014',
              'image_info_test2015',
              'image_info_test2017',
              'image_info_unlabeled2017',
              'panoptic_annotations_trainval2017',
              'stuff_annotations_trainval2017'
        ]
        '''
        super().__init__(*args, **kwds)
        self.__dict__ = self
        self.Z = self.__get_Z(root, annType)
        self.names = self.__get_names(self.Z)

    @staticmethod
    def __get_Z(root, annType):
        '''
        Get the file name of the compressed file under the annotations
        '''
        annType = annType + '.zip'
        annDir = os.path.join(root, 'annotations')
        return zipfile.ZipFile(os.path.join(annDir, annType))

    @staticmethod
    def __get_names(Z):
        names = [name for name in Z.namelist() if not name.endswith('/')]
        return names

    @timer
    def json2dict(self, name):
        with self.Z.open(name) as fp:
            dataset = json.load(fp)
        return dataset


class Loader(dict):
    def __init__(self, batch_size, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.batch_size = batch_size
        self.dataset = dataset
        self.nrows = len(dataset)

    def __iter__(self):
        for start in range(0, self.nrows, self.batch_size):
            end = min(start + self.batch_size, self.nrows)
            yield self.dataset[start:end]

    def __len__(self):
        return ceil(self.nrows / self.batch_size)  # 向上取整


class CSVCat:
    def __init__(self, root, name):
        self.csv2dict(root, name)  # name 是 .csv 文件

    @staticmethod
    def read_csv(root, name):
        return pd.read_csv(os.path.join(root,
                                        name))  # 从本地读取标签信息，格式 (id, label)

    def csv2dict(self, root, name):
        rec = CSVCat.read_csv(root, name).to_records()  # 将 CSV 转换为 Records
        self.cat_dict = {}  # 格式为 {'cat':[id1, id2, ...], ...}
        for _, p, class_name in rec:
            self.cat_dict[class_name] = self.cat_dict.get(class_name,
                                                          []) + [p]  # 列表加法
        self.class_names = tuple(self.cat_dict.keys())  # 获取类别名称列表

    def split(self, test_size=.3):
        import random
        train_dict = {}
        val_dict = {}
        for class_name, id_list in self.cat_dict.items():
            random.shuffle(id_list)
            n = len(id_list)
            test_num = int(n * test_size)
            val_dict[class_name], train_dict[
                class_name] = id_list[:test_num], id_list[test_num:]
        return train_dict, val_dict


class Dataset(ImageZ):
    def __init__(self, dataDir, dataType, items, shuffle=False):
        super().__init__(dataDir, dataType)
        self.items = items
        self.names = sorted(self.items.keys())
        self.class_names = sorted(set(self.items.values()))
        if shuffle:
            np.random.shuffle(self.names)
        self._get_file_idx_dict()
        self._get_file_idx_mapping()
        self.class_weight = self._get_class_weight()

    def _get_file_idx_dict(self):
        '''
        将 names 转换为 dict
        '''
        class_dict = {
            class_name: idx
            for idx, class_name in enumerate(self.class_names)
        }
        self.file_idx_dict = {name: class_dict[self.items[name]] for name in self.names}

    def _get_file_idx_mapping(self):
        '''
        直接通过 id 获取类别 self.class_names[id] 下的所有图片
        '''
        self.file_idx_mapping = {}
        f = self.file_idx_dict
        for fname in f:
            self.file_idx_mapping[f[fname]] = self.file_idx_mapping.get(
                f[fname], []) + [fname]

    def _get_class_weight(self):
        class_weight = []
        f = self.file_idx_mapping
        for idx in range(len(self.class_names)):
            class_weight.append(len(f[idx]))
        return np.array(class_weight) / len(self.items)

    def resize(self, image_name, image_size=224):
        '''
        将图片 resize 后数据类型转换为 'float32'
        '''
        resize_img = cv2.resize(
            self.buffer2array(image_name), (image_size, image_size))
        return resize_img.astype('float32')

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [(self.buffer2array(name), self.items[name])
                    for name in self.names[item]]
        else:
            name = self.names[item]
            return self.buffer2array(name), self.items[name]

    def __iter__(self):
        for fname in self.names:
            yield self.buffer2array(fname), self.items[fname]

    def get_sample(self):
        # 选一个id的位置下标，按照id的权重选
        class_idx = np.random.choice(
            range(len(self.class_names)), 1, p=self.class_weight)[0]
        # 从这个class名中随机选2个图片在class_to_list_files[class]中的位置，有重复（如[0,0]）因为有的类只有1张图片
        filenames = self.file_idx_mapping[class_idx]
        examples_class_idx = np.random.choice(range(len(filenames)), 2)
        # 得到两个正样本图片名（可能相同）
        positive_example_1, positive_example_2 = filenames[examples_class_idx[0]
                                                           ], filenames[examples_class_idx[1]]
        # 负样本选择只要和正样本不同类即可
        negative_example_class = np.random.choice(
            list(set(self.file_idx_mapping.keys()) - {class_idx}), 1)[0]
        neg_filenames = self.file_idx_mapping[negative_example_class]
        negative_example_idx = np.random.choice(
            range(len(neg_filenames)), 1)[0]
        negative_example = self.file_idx_mapping[negative_example_class][negative_example_idx]
        # 返回的都是图片名
        return positive_example_1, positive_example_2, negative_example

    @staticmethod
    def augment(im_array):
        '''
        0.1 概率翻转
        '''
        if np.random.uniform(0, 1) > 0.9:
            im_array = np.fliplr(im_array)
        return im_array

    def hstack(self, image_size=224):
        '''
        将 positive_example_1, negative_example, positive_example_2 横向拼接
        '''
        return np.hstack(
            Dataset.augment(self.resize(fname, image_size))
            for fname in self.get_sample())

    def get_batch(self, batch_size=24, image_size=224):
        '''
        获取 batch_size 个 (positive_example_1, negative_example, positive_example_2)
        '''
        return np.stack(self.hstack(image_size) for _ in range(batch_size))