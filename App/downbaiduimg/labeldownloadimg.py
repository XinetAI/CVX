"""
测试 baiduimages
"""

from baiduimages import BaiduImgDownloader

    
if __name__ == '__main__':
    word_names = input("请输入你要下载的图片关键词所在文本路径：\n")
    try:
        with open(word_names, 'r', encoding='utf-8') as f:
            keywords = f.read().strip()
            if '\ufeff' in keywords:
                keywords = keywords.encode('utf-8').decode('utf-8-sig')
    except UnicodeDecodeError as e:
        with open(word_names, 'r') as f:
            keywords = f.read().strip()
    keywords = keywords.replace('\n', '').replace(' ', '').split(',')
    maxNum = input('请输入每类图片下载的数量：')
 
    dirpath = input("请输入你要下载的图片的保存路径：\n")
    for word in keywords:
        print("^=^" * 25)
        print(f"下载结果保存在脚本目录下的 {word} 文件夹中。")
        down = BaiduImgDownloader(word, dirpath=dirpath)
        down.start(int(maxNum))