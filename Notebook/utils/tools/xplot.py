# -*- coding: UTF-8 -*-
'''
我的画图库
'''

from pylab import plt
import numpy as np
from IPython import display


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体, 为在 Matplotlib 中显示中文，设置特殊字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
#plt.rcParams['figure.size'] = (15.8, 8.0)  # 设置全局图片显示尺寸

def use_svg_display():
    # 用矢量图显示。
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸。
    plt.rcParams['figure.figsize'] = figsize


def show_images(imgs, num_rows, num_cols, scale=2):
    '''
    画出多张图片
    '''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

if __name__ == '__main__':
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = np.random.normal(scale=1, size=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += np.random.normal(scale=0.01, size=labels.shape)
    set_figsize()
    plt.scatter(features[:, 1], labels, 1)
    plt.show()