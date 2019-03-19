from mxnet.gluon import loss as gloss, nn
from mxnet import contrib, nd, autograd


def flatten_pred(pred):  # 转换为通道维度在后，并展开为向量
    return pred.transpose((0, 2, 3, 1)).flatten()  


def concat_preds(preds):    # 拼接不同尺度的类别预测
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1) 


class DownSampleBlock(nn.Block):
    def __init__(self, num_channels, **kwargs):
        '''
        高和宽减半块
        '''
        super().__init__(**kwargs)
        self.block = nn.Sequential()
        with self.block.name_scope():
            for _ in range(2):
                self.block.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                               nn.BatchNorm(in_channels=num_channels),
                               nn.Activation('relu'))
            self.block.add(nn.MaxPool2D(2))

    def forward(self, X):
        return self.block(X)


class BaseNet(nn.Block):
    def __init__(self, **kwargs):
        '''
        基网络
        '''
        super().__init__(**kwargs)
        self.block = nn.Sequential()
        with self.block.name_scope():
            for num_filters in [16, 32, 64]:
                self.block.add(DownSampleBlock(num_filters))

    def forward(self, X):
        return self.block(X)


class AnchorY(nn.Block):
    def __init__(self, block, size, ratio, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        self._size = size
        self._ratio = ratio

    def forward(self, X):
        Y = self.block(X)
        anchors = contrib.ndarray.MultiBoxPrior(
            Y, sizes=self._size, ratios=self._ratio)
        return Y, anchors


class ClassPredictor(nn.Block):
    def __init__(self, num_anchors, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 类别预测层
        self.cls_predictor = nn.Conv2D(
            self.num_anchors * (self.num_classes + 1), kernel_size=3, padding=1)

    def forward(self, Y):
        cls_preds = self.cls_predictor(Y)
        return cls_preds


class BBoxPredictor(nn.Block):
    def __init__(self, num_anchors, **kwargs):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors
        # 边界框预测层
        self.bbox_predictor = nn.Conv2D(
            self.num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, Y):
        bbox_preds = self.bbox_predictor(Y)
        return bbox_preds


class TinySSD(nn.Block):
    def __init__(self, sizes, ratios, num_classes, **kwargs):
        super().__init__(**kwargs)
        sizes, ratios, self.num_classes = sizes, ratios, num_classes
        self.num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, self.block(i))
            setattr(self, 'cls_%d' % i, ClassPredictor(self.num_anchors,
                                                       self.num_classes))
            setattr(self, 'bbox_%d' % i, BBoxPredictor(self.num_anchors))
            setattr(self, 'anchor_%d' % i, AnchorY(
                getattr(self, 'blk_%d' % i), sizes[i], ratios[i]))

    def block(self, i):
        if i == 0:
            blk = BaseNet()
        elif i == 4:
            blk = nn.GlobalMaxPool2D()
        else:
            blk = DownSampleBlock(128)
        return blk

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i)即访问self.blk_i
            Y, anchors[i] = getattr(self, 'anchor_%d' % i)(X)
            cls_preds[i] = getattr(self, 'cls_%d' % i)(Y)
            bbox_preds[i] = getattr(self, 'bbox_%d' % i)(Y)
            X = Y
        # reshape函数中的0表示保持批量大小不变
        cls_preds = concat_preds(cls_preds).reshape(
            (0, -1, self.num_classes + 1))
        return nd.concat(*anchors, dim=1), cls_preds, concat_preds(bbox_preds)