from mxnet.gluon.data.vision import transforms as gtf

# 自定义
from ..zipimage import ImageZ

transform_train = gtf.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
    # 宽都是为 224 的新图
    gtf.RandomResizedCrop(
        224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    gtf.RandomFlipLeftRight(),
    # 随机变化亮度、对比度和饱和度
    gtf.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # 随机加噪声
    gtf.RandomLighting(0.1),
    gtf.ToTensor(),
    # 对图像的每个通道做标准化
    gtf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = gtf.Compose([
    gtf.Resize(256),
    # 将图像中央的高和宽均为 224 的正方形区域裁剪出来
    gtf.CenterCrop(224),
    gtf.ToTensor(),
    gtf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])