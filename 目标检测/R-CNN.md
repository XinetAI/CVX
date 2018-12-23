# R-CNN

- [R-CNN论文翻译](http://www.cnblogs.com/pengsky2016/p/7921857.html)

R-CNN[^1] 方法结合了两个关键的因素：

1. 将大型卷积神经网络(CNNs)应用于自下而上的候选区域以定位和分割物体。
2. 当带标签的训练数据不足时，先针对辅助任务进行有监督预训练，再进行特定任务的调优，就可以产生明显的性能提升。

![R-CNN 时间复杂度](images/R-CNN-timer.jpg)

[^1]: Girshick R B, Donahue J, Darrell T, et al. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation[J]. computer vision and pattern recognition, 2014: 580-587.

[R-CNN论文详解](https://blog.csdn.net/WoPawn/article/details/52133338?tdsourcetag=s_pcqq_aiomsg)