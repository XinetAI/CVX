import numpy as np


class AnchorBase:
    def __init__(self, base_size, scales, ratios):
        self.scales = np.array(scales)  #
        self.ratios = np.array(ratios)  #
        self.num_anchors = len(self.ratios) * len(self.scales)  # 锚框的个数
        self.base_size = base_size  # 滑动窗口的大小
        if isinstance(base_size, int):
            self._w, self._h = [base_size]*2
        elif len(base_size) == 2:
            self._w, self._h = base_size
        elif len(base_size) == 1:
            self._w, self._h = base_size*2

        self._anchor = np.array([1, 1, self._w, self._h]) - 1

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
        依据宽高组合计算锚框的坐标
        '''
        k = (aspect - 1) / 2
        return np.concatenate([ctr - k, ctr + k], axis=1)


class AnchorRCNN(AnchorBase):
    def __init__(self, base_size, scales, ratios):
        super().__init__(base_size, scales, ratios)
        self.anchors = self.gen_anchors()

    @property
    def ratio_aspects(self):
        '''
        依据 ratios 获取锚框的所有宽高组合
        '''
        size_ratios = self.size / self.ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * self.ratios)
        return np.stack([ws, hs], axis=1)

    @property
    def ratio_anchors(self):
        return self._coordinate(self.ratio_aspects, self._whctrs)

    @property
    def scale_aspects(self):
        '''
        依据 scales 获取锚框的所有宽高组合
        '''
        ws = self.w * self.scales
        hs = self.h * self.scales
        return np.stack([ws, hs], axis=1)

    @property
    def scale_anchors(self):
        return self._coordinate(self.scale_aspects, self._whctrs)

    def gen_anchors(self):
        '''
        获取最终的 base_anchors
        '''
        anchors = []
        for anchor in self.ratio_anchors:
            self.anchor = anchor
            anchors.append(self.scale_anchors)
        return np.concatenate(anchors)


class Anchor(AnchorBase):
    def __init__(self, base_size, scales, ratios):
        super().__init__(base_size, scales, ratios)
        self.ratios = self.ratios[:, None]

    @property
    def W(self):
        '''
        计算 w_1/ w
        '''
        W = self.scales / np.sqrt(self.ratios)
        return np.round(W)

    @property
    def H(self):
        '''
        计算 h_1/ h
        '''
        H = self.W * self.ratios
        return np.round(H)

    @property
    def aspect(self):
        '''
        所有的宽高组合
        '''
        return np.stack([self.W.flatten(), self.H.flatten()], axis=1)

    @property
    def base_anchors(self):
        return self._coordinate(self.aspect, self._whctrs)

    @property
    def anchors(self):
        '''
        获取最终的 base_anchors
        '''
        return self.base_anchors * np.array([self.w, self.h]*2)