'''
@Xint: https://www.jianshu.com/p/f12b545c63a4
@博客园: http://www.cnblogs.com/q735613050/

实现使用 dHash 除去本地重复图片
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


class XHash:
    '''
    感知 Hash 算法
    '''

    def __init__(self, image_path, hash_type):
        self.image_path = image_path
        self.hash_size = 8
        self.type = hash_type
        if self.type == 'aHash':
            self.hash = self.__aHash()
        elif self.type == 'dHash':
            self.hash = self.__dHash()

    def __get_gray(self, img):
        '''
        读取 RGB 图片 并转换为灰度图
        '''
        # 由于 cv2.imread 无法识别中文路径，所以使用 plt.imread
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    def __difference(self):
        '''
        比较左右像素的差异
        '''
        #img = plt.imread(self.image_path)
        img = cv2.imdecode(np.fromfile(self.image_path,dtype=np.uint8),cv2.IMREAD_COLOR) # 解决无法取得中文路径问题
        resize_img = cv2.resize(img, (self.hash_size+1, self.hash_size))
        gray = self.__get_gray(resize_img)  # 灰度化
        # 计算差异
        differences = []
        for t in range(resize_img.shape[1] - 1):
            differences.append(gray[:, t] > gray[:, t + 1])
        return np.stack(differences).T

    def __average(self):
        '''
        与像素均值进行比较
        '''
        #img = plt.imread(self.image_path)
        img = cv2.imdecode(np.fromfile(self.image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        resize_img = cv2.resize(img, (self.hash_size, self.hash_size))
        gray = self.__get_gray(resize_img)
        return gray > gray.mean()

    def __binarization(self, hash_image):
        '''
        二值化
        '''
        return ''.join(hash_image.astype('B').flatten().astype('U').tolist())

    def __seg(self, hash_image):
        img_bi = self.__binarization(hash_image)
        return list(
            map(lambda x: '%x' % int(img_bi[x:x + 4], 2), range(0, 64, 4)))

    def __aHash(self):
        return self.__seg(self.__average())

    def __dHash(self):
        return self.__seg(self.__difference())


# class XHash_Haming:
#     '''
#     计算两张图片的相似度
#     '''

#     def __init__(self, image_path1, image_path2, hash_type):
#         self.hash_img1 = XHash(image_path1, hash_type).hash
#         self.hash_img2 = XHash(image_path2, hash_type).hash

#     def hash_haming(self):
#         '''
#         计算两张通过哈希感知算法编码的图片的汉明距离
#         '''
#         return np.array(
#             [self.hash_img1[x] != self.hash_img2[x] for x in range(16)],
#             dtype='B').sum()


class Pairs:
    '''
    使用 dHash 实现哈希感知算法
    '''

    def __init__(self, root):
        self.root = root
        self.__del_all_null_images()
        self.__hashs = np.array([
            XHash(self.names[name], 'dHash').hash
            for name in self.names.keys()
        ])
        self.__cal_haming_distance(self.__hashs)

    def __del_null_image(self, image_path):
        '''
        删除无效图片
        '''
        os.remove(image_path)
        print('删除无效图片：', image_path)

    def __del_all_null_images(self):
        self.names = {}
        for j, name in enumerate(os.listdir(self.root)):
            if not name.endswith('txt' or '.py' or '.ipynb'):
                image_path = os.path.join(self.root, name)
                if os.path.getsize(image_path)<1024*2:  # 过滤掉小于 2kB 的图片
                    self.__del_null_image(image_path)
                else:
                    # 解决无法取得中文路径问题
                    img = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
                    if img is None:
                        self.__del_null_image(image_path)
                    else:
                        self.names[j] = image_path

    def __cal_haming_distance(self, hashs):
        '''
        计算两两之间的距离
        '''
        j = 0
        pairs = {}
        while j < hashs.shape[0]:
            for i in range(j + 1, hashs.shape[0]):  # 图片对，过滤到已经计算过的 pairs
                pairs[j] = pairs.get(j, []) + \
                    [np.array(hashs[i] != hashs[j]).sum()]
                continue
            j += 1
        self.pairs = pairs

    def get_names(self):
        n = len(self.pairs)
        temp = {}
        while n > 0:
            n -= 1
            for i, d in enumerate(self.pairs[n]):
                if d == 0:
                    temp[n] = temp.get(n, []) + [i + n + 1]
                    continue
        return temp

    def del_repeat(self):
        P = self.get_names()
        for j in P:
            for i in P[j]:
                try:
                    os.remove(self.names[i])
                    print('删除重复图片：', self.names[i])
                except FileNotFoundError:
                    print(f'已经移除 {self.names[i]}，无需再次移除！')
        print('删除完成！')


if __name__ == '__main__':
    change = input('是否需要改变当前图片所在根目录？如果是请输入 y, 否则不改变根目录。')
    if change == 'y':
        root = input('请输入图片所在根目录：')
    else:
        root = os.getcwd()
    for path in os.listdir(root):
        paths = os.path.join(root, path)
        if os.path.isdir(paths):
            xhash = Pairs(paths)
            xhash.del_repeat()   # 删除重复图片