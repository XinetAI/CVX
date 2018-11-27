"""
测试 baiduimages
"""

from baiduimages import BaiduImgDownloader

    
if __name__ == '__main__':
    words = input("请输入你要下载的图片关键词（一定要输入有意义的词汇），多个关键词请使用空格隔开：\n").split()
    maxNums = input(
        f"请分别输入 {words} 图片的下载数量列表（最好是 60 的倍数，并使用空格隔开, 若你仅输入一个数字，则默认为所有类别的下载数目）：\n")
    maxNums = maxNums.split()

    if len(words) != len(maxNums):
        if len(maxNums) == 1:
            maxNums = len(words) * maxNums
        else:
            maxNums = input('长度不匹配，请重新输入图片的下载数量的列表：')
            maxNums = maxNums.split()
 
    dirpath = input("请输入你要下载的图片的保存路径：\n")
    for word, maxNum in zip(words, maxNums):
        print("^=^" * 25)
        print(f"下载结果保存在脚本目录下的 {word} 文件夹中。")
        down = BaiduImgDownloader(word, dirpath=dirpath)
        down.start(int(maxNum))