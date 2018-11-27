# 目标检测资源汇总

## 1  主要文献

### 1.1  滑窗法

论文：[Mitosis Detection in Breast Cancer Histology Images
with Deep Neural Networks](papers/滑窗法.pdf)[^1]

[^1]: Ciresan D C, Giusti A, Gambardella L M, et al. Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks[C]. medical image computing and computer assisted intervention, 2013: 411-418.

滑窗法（Sliding Window）的思路及其简单，首先需要已经训练好的一个分类器，然后把图片按照一定间隔和不同的大小分成一个个窗口，在这些窗口上执行分类器。如果得到较高的分数分类，就认为是检测到了物体。把每个窗口都用分类器执行一遍之后，再对得到的分数做一些后处理，如非极大值抑制（Non-Maximum Suppression，NMS）等，最后得到物体类别和对应区域。

![Sliding Window](../images/SlidingWindow.png)

滑窗法非常简单，但是效率低下，尤其是还要考虑物体的长宽比。如果执行比较耗时的分类器算法，用滑窗法就不太现实。常见的都是一些小型分类网络和滑窗法结合的应用，如论文《Mitosis Detection in Breast Cancer Histology Images
with Deep Neural Networks》所做的检测胸切片图像中有丝分裂用于辅助癌症诊断。

### 1.2  非极大值抑制

- 论文：[Efﬁcient Non-Maximum Suppression](papers/NMS.pdf)[^2]
- [解读](https://www.jianshu.com/p/325e3747fc56)

[^2]: Neubeck A, Van Gool L. Efficient Non-Maximum Suppression[C]. international conference on pattern recognition, 2006: 850-855.

### 1.3  选择性搜索

- [Selective Search for Object Recognition](papers/UijlingsIJCV2013.pdf)[^3]

[^3]: Uijlings J R, De Sande K E, Gevers T, et al. Selective Search for Object Recognition[J]. International Journal of Computer Vision, 2013, 104(2): 154-171.

### 1.4  R-CNN

- 论文：[Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](papers/R-CNN.pdf)[^4]
- [解读](R-CNN.md)
- [Slides](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- [项目地址](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf): https://github.com/rbgirshick/rcnn (基于MATLAB)

[^4]: Girshick R B, Donahue J, Darrell T, et al. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation[J]. computer vision and pattern recognition, 2014: 580-587.

### 1.5  SPP

- 论文：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](papers/SPPNETs.pdf)[^5]

[^5]: He K, Zhang X, Ren S, et al. Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 37(9): 1904-1916.

### 1.6  Fast R-CNN

- 论文：[Fast R-CNN](papers/Fast_R-CNN.pdf)[^6]

[^6]: Girshick R B. Fast R-CNN[J]. international conference on computer vision, 2015: 1440-1448.

### 1.7  Faster R-CNN

- 论文1：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](papers/faster-r-cnn.pdf)[^6]
- 论文2：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](papers/Faster_R-CNN_2017.pdf)[^8]

[^7]: Ren S, He K, Girshick R B, et al. Faster R-CNN: towards real-time object detection with region proposal networks[C]. neural information processing systems, 2015: 91-99.

[^8]: Ren S, He K, Girshick R B, et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(6): 1137-1149.

### 1.8  Mask R-CNN

- 论文1：[Even faster sorting of (not only) integers](papers/1703.00687.pdf)[^9]
- 论文2：[MaskR-CNN](papers/Mask-R-CNN.pdf)[^10]

[^9]: Kokot M, Deorowicz S, Dlugosz M, et al. Even Faster Sorting of (Not Only) Integers[J]. arXiv: Data Structures and Algorithms, 2017: 481-491.

[^10]: He K, Gkioxari G, Dollar P, et al. Mask R-CNN[J]. international conference on computer vision, 2017: 2980-2988.

### 1.9  YOLO

- 论文1：[You Only Look Once: Unified, Real-Time Object Detection](papers/YOLO.pdf)[^11]
- 论文2：[YOLO9000: Better, Faster, Stronger](papers/YOLO9000.pdf)[^12]
- 论文3：[YOLOv3: An Incremental Improvement](papers/YOLOv3.pdf)[^13]

[^11]: Redmon J, Divvala S K, Girshick R B, et al. You Only Look Once: Unified, Real-Time Object Detection[J]. computer vision and pattern recognition, 2016: 779-788.

[^12]: Redmon J, Farhadi A. YOLO9000: Better, Faster, Stronger[J]. computer vision and pattern recognition, 2017: 6517-6525.

[^13]: Redmon J, Farhadi A. YOLOv3: An Incremental Improvement.[J]. arXiv: Computer Vision and Pattern Recognition, 2018.

### 1.10  SSD

- 论文：[SSD: Single Shot MultiBox Detector](papers/1512.02325v5.pdf)[^14]

[^14]: Liu W, Anguelov D, Erhan D, et al. SSD: Single Shot MultiBox Detector[J]. european conference on computer vision, 2016: 21-37

## 2  预训练模型下载链接

- [vgg16_reduced.zip](model/vgg16_reduced.zip)
- [ssd_300_vgg16_reduced_voc0712_trainval.zip](model/ssd_300_vgg16_reduced_voc0712_trainval.zip)