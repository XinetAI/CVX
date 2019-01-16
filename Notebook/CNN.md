# 第7章 Kaggle 实战：猫狗分类

本章利用 Kaggle 上的数据集： https://www.kaggle.com/c/dogs-vs-cats/data 来学习卷积神经网络。利用了第 6 章的几个模块： AnnZ, ImageZ, Loader, Dataset。它们被封装进了一个名字的 `zipimage.py` 的文件中。接下的几章也会利用改模块。

本章快报：

- 

先在 https://www.kaggle.com/c/dogs-vs-cats/data 查看下数据的基本信息。从该网址可以知道：训练数据 `tain.zip` 包括 $50\,000$ 个样本，其中猫和狗各半。而其任务是预测 `test1.zip` 的标签（$1 = dog, 0 = cat$）。为了方便将数据下载到本地，然后做如下操作：