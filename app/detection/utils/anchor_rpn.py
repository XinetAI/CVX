import numpy as np

from .anchor import Anchor


class ReferenceAnchor(Anchor):
    r"""Anchor generator for Region Proposal Netoworks.

    Parameters
    ----------
    stride : int
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    base_size : int
        The width(and height) of reference anchor box.
    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.

    """
    def __init__(self, base_size, ratios, scales):
        super().__init__(base_size)
        if not isinstance(ratios, (tuple, list)):
            ratios = [ratios]
        if not isinstance(scales, (tuple, list)):
            scales = [scales]
        self.ratios = np.array(ratios)
        self.scales = np.array(scales)

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

    def reference_anchors(self):
        '''
        获取最终的 reference anchors
        '''
        anchors = []
        for anchor in self.ratio_anchors:
            self.anchor = anchor
            anchors.append(self.scale_anchors)
        return np.concatenate(anchors)

    def generate_anchors(self, alloc_size, stride):
        base_anchors = self.reference_anchors()
        # propagete to all locations by shifting offsets
        height, width = alloc_size
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                            offset_x.ravel(), offset_y.ravel()), axis=1)
        return base_anchors.reshape((1, -1, 4)) + offsets.reshape((-1, 1, 4))