import numpy as np


class Anchor:
    def __init__(self, base_size=16):
        self.base_size = base_size  # window's size
        if not base_size:
            raise ValueError("Invalid base_size: {}.".format(base_size))
        self._anchor = np.array([1, 1, self.base_size, self.base_size]) - 1

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, new_anchor):
        self._anchor = new_anchor

    @property
    def w(self):
        '''
        锚框的宽度
        '''
        return self.anchor[2] - self.anchor[0] + 1

    @property
    def h(self):
        '''
        锚框的高度
        '''
        return self.anchor[3] - self.anchor[1] + 1

    @property
    def size(self):
        '''
        锚框的面积
        '''
        return self.w * self.h

    @property
    def _whctrs(self):
        """
        Return x center, and y center for an anchor (window). 锚框的中心坐标
        """
        x_ctr = self.anchor[0] + 0.5 * (self.w - 1)
        y_ctr = self.anchor[1] + 0.5 * (self.h - 1)
        return np.array([x_ctr, y_ctr])

    @staticmethod
    def _coordinate(aspect, ctr):
        '''
        依据高宽组合计算锚框的坐标
        '''
        k = (aspect - 1) * .5
        return np.concatenate([ctr - k, ctr + k], axis=1)


def iou(anchor, anchor1):
    '''
    计算 IoU
    '''
    A = Anchor()
    B = Anchor()
    A.anchor = anchor
    B.anchor = anchor1
    T = np.stack([A.anchor, B.anchor])
    xmin, ymin, xmax, ymax = np.split(T, 4, axis=1)
    w = xmax.min() - xmin.max()
    h = ymax.min() - ymin.max()
    I = w * h
    U = A.size + B.size - I
    return I / U