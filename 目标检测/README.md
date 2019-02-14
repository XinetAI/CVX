# 目标检测资源汇总

[目标检测模型的评估指标 mAP 详解(附代码）](https://zhuanlan.zhihu.com/p/37910324)

## 1  主要文献

### 1.1  滑窗法

滑窗法（Sliding Window）的思路及其简单，首先需要已经训练好的一个分类器，然后把图片按照一定间隔和不同的大小分成一个个窗口，在这些窗口上执行分类器。如果得到较高的分数分类，就认为是检测到了物体。把每个窗口都用分类器执行一遍之后，再对得到的分数做一些后处理，如非极大值抑制（Non-Maximum Suppression，NMS）等，最后得到物体类别和对应区域。

![Sliding Window](../images/SlidingWindow.png)

滑窗法非常简单，但是效率低下，尤其是还要考虑物体的长宽比。如果执行比较耗时的分类器算法，用滑窗法就不太现实。常见的都是一些小型分类网络和滑窗法结合的应用，如论文《[Mitosis Detection in Breast Cancer Histology Images
with Deep Neural Networks](papers/滑窗法.pdf)[^1]》所做的检测胸切片图像中有丝分裂用于辅助癌症诊断。

[^1]: Ciresan D C, Giusti A, Gambardella L M, et al. Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks[C]. medical image computing and computer assisted intervention, 2013: 411-418.

### 1.2  非极大值抑制

- 论文：[Efﬁcient Non-Maximum Suppression](papers/NMS.pdf)[^2]
- [解读](https://www.jianshu.com/p/325e3747fc56)

[^2]: Neubeck A, Van Gool L. Efficient Non-Maximum Suppression[C]. international conference on pattern recognition, 2006: 850-855.

### 1.3  选择性搜索

选择性搜索（Selective Search）是主要运用图像分割技术来进行物体检测。

#### 1.3.1 简介

Selective Search 属于传统机器学习的方法，在 Faster R-CNN 中被 RPN 所取代。

在较高层次上进行选择性搜索通过不同大小的窗口查看图像，并且对于每个尺寸，尝试通过纹理、颜色或强度将相邻像素组合在一起以标识对象。类似一个聚类的过程。在窗口的 size 更大的时候，相邻聚类尝试合并。最后把不同窗口大小下的不同聚类区块都提交作为 proposal。

- [Selective Search for Object Recognition](papers/UijlingsIJCV2013.pdf)[^3]
- [论文笔记《Selective Search for object recognition》](https://blog.csdn.net/niaolianjiulin/article/details/52950797)
- [[初窥目标检测]——《目标检测学习笔记（2）:浅析Selective Search论文——“Selective Search for object recognition”》](https://blog.csdn.net/u011478575/article/details/80041921)

[^3]: Uijlings J R, De Sande K E, Gevers T, et al. Selective Search for Object Recognition[J]. International Journal of Computer Vision, 2013, 104(2): 154-171.

---------------------

- 输入：彩色图片（三通道）
- 输出：物体位置的可能结果 $L$
    1. 使用《Efficient Graph-Based Image Segmentation》方法，获取初始分割区域 $R=\{r_1,r_2, \ldots, r_n\}$
    2. 初始化相似度集合 $S=∅$
    3. 计算 $R$ 中两两相邻区域 $r_i, r_j$ 之间的相似度，将其添加到相似度集合 $S$ 中。
    4. 从相似度集合 $S$ 中找出，相似度最大的两个区域 $r_i$ 和 $r_j$，将其合并成为一个区域 $r_t$。然后从相似度集合中除去原先与 $r_i$ 和 $r_j$ 相邻区域之间计算的相似度。计算新的 $r_t$ 与其相邻区域（原先与 $r_i$ 或 $r_j$ 相邻的区域）的相似度，将其结果添加的到相似度集合 $S$ 中。同时将新区域 $r_t$ 添加到区域集合 $R$ 中。迭代直至 $S$ 为空，即可合并区域的都已合并完。区域的合并方式类似于哈夫曼树的构造过程，因此称之有**层次**（hierarchical）。
    5. 获取 $R$ 中每个区域的 Bounding Boxes，这个结果就是图像中物体可能位置的可能结果集合 $L$。

-------------

#### 1.3.2  解读

- [选择性搜索 Selective Search -- 算法详解+源码分析](https://blog.csdn.net/Tomxiaodai/article/details/81412354)
- [目标检测--Selective Search for Object Recognition(IJCV, 2013)](http://www.cnblogs.com/zhao441354231/p/5941190.html)
- 项目地址：http://disi.unitn.it/~uijlings/MyHomepage/index.php#page=projects1
- GitHub：https://github.com/CodeXZone/selectivesearch

### 1.4  R-CNN

- 论文：[Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](papers/R-CNN.pdf)[^4]
- [解读](R-CNN.md)
- [Slides](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- [项目地址](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf): https://github.com/rbgirshick/rcnn (基于MATLAB)

[^4]: Girshick R B, Donahue J, Darrell T, et al. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation[J]. computer vision and pattern recognition, 2014: 580-587.

- [R-CNN论文翻译](http://www.cnblogs.com/pengsky2016/p/7921857.html)

R-CNN 方法结合了两个关键的因素：

1. 将大型卷积神经网络(CNNs)应用于自下而上的候选区域以定位和分割物体。
2. 当带标签的训练数据不足时，先针对辅助任务进行有监督预训练，再进行特定任务的调优，就可以产生明显的性能提升。

![R-CNN 时间复杂度](images/R-CNN-timer.jpg)

[R-CNN论文详解](https://blog.csdn.net/WoPawn/article/details/52133338?tdsourcetag=s_pcqq_aiomsg)

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

[一文读懂 Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

#### 1.7.1  翻译与总结

- [Faster R-CNN论文翻译——中文版](https://www.jianshu.com/p/7adc34483e4a?utm_campaign=maleskine)

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

- [YOLO论文翻译——中英文对照](https://www.jianshu.com/p/a2a22b0c4742?utm_campaign=maleskine)

[目标检测|YOLO原理与实现](https://zhuanlan.zhihu.com/p/32525231)

### 1.10  SSD

- 论文：[SSD: Single Shot MultiBox Detector](papers/1512.02325v5.pdf)[^14]

[^14]: Liu W, Anguelov D, Erhan D, et al. SSD: Single Shot MultiBox Detector[J]. european conference on computer vision, 2016: 21-37

#### 1.10.1 翻译与总结

- [Single Shot MultiBox Detector论文翻译——中文版](https://www.jianshu.com/p/467419cf94dd?utm_campaign=maleskine)

### 1.11  A Survey

- [Deep Learning for Generic Object Detection: A Survey](papers/1809.02165.pdf)[^15]

[^15]: Liu L, Ouyang W, Wang X, et al. Deep Learning for Generic Object Detection: A Survey.[J]. arXiv: Computer Vision and Pattern Recognition, 2018.

## 2  预训练模型下载链接

- [vgg16_reduced.zip](model/vgg16_reduced.zip)
- [ssd_300_vgg16_reduced_voc0712_trainval.zip](model/ssd_300_vgg16_reduced_voc0712_trainval.zip)

## 3 52CV.NET

- [Grid R-CNN解读：商汤最新目标检测算法](https://www.52cv.net/?p=1800)
- [Grid R-CNN 论文](papers/Grid R-CNN.pdf)
- [计算机视觉研究入门全指南](https://www.52cv.net/?p=524)