"""带进度条的方式下载文件"""
import os
import hashlib
import requests
from tqdm import tqdm  # 进度条工具


def get_url_path(url, path=None):
    '''
    根据 url 与 path 的输入情况，输出下载文件的路径
    '''
    try:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    except:
        fname = url.split('/')[-1]
    return fname


def check_sha1(filename, sha1_hash):
    """检查文件内容的 sha1 哈希码是否与预期的哈希码匹配
    参数
    ----------
    filename : str
        文件路径
    sha1_hash : str
        十六进制数字的预期 sha1 哈希
    Returns
    -------
    bool
        文件内容是否与预期的哈希匹配。
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    L = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:L] == sha1_hash[0:L]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """下载给定网址的文件
    Parameters
    ----------
    url : str
        文件网址
    path : str, optional
        存储下载文件的目标路径。默认情况下, 存储到与 url 中同名的当前目录

    overwrite : bool, optional
       是否是否覆盖目标文件 (如果已存在)
    sha1_hash : str, optional
        十六进制数字的预期 sha1 哈希；在指定现有文件哈希不匹配时将忽略
    Returns
    -------
    str
       下载文件的文件路径
    """
    fname = get_url_path(url, path)

    if overwrite or not os.path.exists(fname) or (
            sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                        r.iter_content(chunk_size=1024),
                        total=int(total_length / 1024. + 0.5),
                        unit='KB',
                        unit_scale=False,
                        dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. '
                'The repo may be outdated or download may be incomplete. '
                'If the "repo_url" is overridden, consider switching to '
                'the default repo.'.format(fname))

    return fname