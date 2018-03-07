---
title: Python Web客户端和服务器
date: 2018-03-06 09:20:10
categories: Python
copyright: true
tags:
    - Python
    - Web客户端和服务器
description:
---

## Python Web客户端工具
浏览器只是Web客户端的一种。任何一个向Web服务器端发送请求来获取数据的应用程序都是“客户端”。使用urllib模块下载或者访问Web上信息的应用程序就是简单的Web客户端。

### 统一资源定位符
>URL(统一资源定位符)适用于网页浏览的一个地址，这个地址用来在Web上定位一个文档，或者调用一个CGI程序来为客户端生成一个文档。URL是多种统一资源标识符(Uniform Resource Identifier, URI)的一部分。一个URL是一个简单的URI，它使用已有的协议或方案(http/ftp等)。非URL的URI有时称为统一资源名称(Uniform Resource Name, URN)，现在唯一使用的URI只有URL。

URL使用以下格式：
`post_sch://net_loc/path;parans?query#frag`
* Web地址的各个组件

| URL组件  |                 描述                 |
|:--------:|:------------------------------------:|
| post_sch |          网络协议或下载方案          |
| net_loc  |    服务器所在地(也许含有用户信息)    |
|   path   | 使用斜杠(/)分割的文件或CGI应用的路径 |
|  params  |               可选参数               |
|  query   |     连接符(&)分割的一系列键值对      |
|   frag   |        指定文档内特定锚的部分        |

net_loc可以拆分为多个组件，一些可选一些必备：
`user:passwd@host:port`
* 网络地址的各个组件

|  组件  |                  描述                   |
|:------:|:---------------------------------------:|
|  user  |            用户名或登录(FTP)            |
| passwd |              用户密码(FTP)              |
|  host  | 运行Web服务器的计算机名称或地址(必需的) |
|  port  |        端口号(如果不是默认的80)         |

Python3 使用[urllib.parse](https://docs.python.org/3/library/urllib.parse.html)和[urllib.request](https://docs.python.org/3/library/urllib.request.html)两种不同的模块分别以不同的功能和兼容性来处理URL

### urllib.parse模块
* urllib.parse核心函数

|                                         urllib.parse函数                                         |                                                       描述                                                       |
|:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
|                 urllib.parse.urlparse(urlstring, scheme='',allow_fragments=True)                 | 将urlstring解析成各个组件，如果在urlstring中没有给定协议或者方法，使用scheme；allow_fragments决定是否允许URL片段 |
|                                  urllib.parse.urlunparse(parts)                                  |                                         将URL数据的一个元组拼成URL字符串                                         |
|                       urllib.parse.urljoin(base,url,allow_fragments=True)                        |                   将URL的根域名和url拼合成一个完整的URL；allow_fragments的决定是否允许URL片段                    |
|                  urllib.parse.quote(string,safe='/',encoding=None,errors=None)                   |                           对string在URL里无法使用的字符进行编码，safe中的字符无需编码                            |
|                     urllib.parse.quote_plus(string,safe='',encoding,errors)                      |                           除了将空格编译成加(+)号(而非20%)之外，其他功能与quote()相似                            |
|                  urllib.parse.unquote(string,encoding='utf-8',errors='replace')                  |                                             将string编译过的字符解码                                             |
|               urllib.parse.unquote_plus(string,encoding='utf-8',errors='replace')                |                                  除了将加好转换为空格，其他功能与unquote()相同                                   |
| urllib.parse.urlencode(query,doseq=False,safe='',encoding=None,errors=None,quote_via=quote_plus) |               将query通过quote_plus()编译成有效的CGI查询自妇产，用quote_plus()对这个字符串进行编码               |

下面将对每个方法进行演示,首先导入urllib.parse下面的所有方法
`from urllib.parse import *`

* *urllib.parse.urlparse(urlstring, scheme='',allow_fragments=True)*
```Python
urlparse('http://coldjune.com/categories/')
# 输出结果
ParseResult(scheme='http', netloc='coldjune.com', path='/categories/', params='', query='', fragment='')
```

* *urllib.parse.urlunparse(parts)*
```Python
urlunparse(('http', 'coldjune.com', '/categories/', '', '', ''))
# 输出结果
'http://coldjune.com/categories/'
```
* *urllib.parse.urljoin(base,url,allow_fragments=True)*
```Python
# 如果是绝对路径将整个替换除根域名以外的所有内容
urljoin('http://coldjune.com/categories/1.html','/tags/2.html')
# 输出结果
'http://coldjune.com/tags/2.html'

# 如果是相对路径将会将末端文件去掉与心得url连接
urljoin('http://coldjune.com/categories/1.html','tags/2.html')
# 输出结果
'http://coldjune.com/categories/tags/2.html'
``
* *urllib.parse.quote(string,safe='/',encoding=None,errors=None)*
> 逗号、下划线、句号、斜线和字母数字这类符号不需要转换，其他均需转换。URL不能使用的字符前面会被加上百分号(%)同时转换为十六进制(%xx,xx表示这个字母的十六进制)

  ```Python
  quote('http://www.~coldjune.com/tag categoriese?name=coold&search=6')
  # 输出结果
  'http%3A//www.%7Ecoldjune.com/tag%20categoriese%3Fname%3Dcoold%26search%3D6'
  ```

* *urllib.parse.unquote(string,encoding='utf-8',errors='replace')*
```Python
unquote('http%3A//www.%7Ecoldjune.com/tag%20categoriese%3Fname%3Dcoold%26search%3D6')
# 输出结果
'http://www.~coldjune.com/tag categoriese?name=coold&search=6'
```

* *urllib.parse.quote_plus(string,safe='',encoding,errors)*
```Python
quote_plus('http://www.~coldjune.com/tag categoriese?name=coold&search=6')
# 输出结果
'http%3A%2F%2Fwww.%7Ecoldjune.com%2Ftag+categoriese%3Fname%3Dcoold%26search%3D6'
```

* *urllib.parse.unquote_plus(string,encoding='utf-8',errors='replace')*
```Python
unquote_plus('http%3A%2F%2Fwww.%7Ecoldjune.com%2Ftag+categoriese%3Fname%3Dcoold%26search%3D6')
# 输出结果
'http://www.~coldjune.com/tag categoriese?name=coold&search=6'
```

* *urllib.parse.urlencode(query,doseq=False,safe='',encoding=None,errors=None,quote_via=quote_plus)*
```Python
query={'name':'coldjune','search':'6'}
urlencode(query)
# 输出结果
'name=coldjune&search=6'
```

### urllib.request模块/包
* urllib.request模块核心函数

|                                            urllib.request函数                                             |                                                              描述                                                              |
|:---------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|
| urllib.request.urlopen(url, data=None, [timeout,]*,cafile=None, capath=None,cadefault=False,context=None) | 打开url(string或者Request对象)，data为发送给服务器的数据，timeout为超时属性， cafile,capath,cadefault为调用HTTPS请求时证书认证 |
|                  urllib.request.urlretrieve(url,filename=None,reporthook=None,data=None)                  | 将url中的文件下载到filename或临时文件中(如果没有指定filename)；如果函数正在执行，reporthook将会获得下载的统计信息                                                                                                                               |

1. *urllib.request.urlopen(url, data=None, [timeout,],*
  *cafile=None, capath=None,cadefault=False,context=None)*
>urlopen()打开url所指向的URL；如果没有给定协议或者下载方案，或者传入"file"方案，urlopen()会打开一个本地文件。对于所有的HTTP请求，使用"GET"请求，向Web服务器发送的请求字符串应该是url的一部分；使用"POST"请求，请求的字符串应该放到data变量中。连接成功后返回的是一个文件类型对象

* urlopen()文件类型对象的方法

|      方法       |             描述              |
|:---------------:|:-----------------------------:|
| f.read([bytes]) |  从f中读出所有或bytes个字节   |
|  f.readline()   |         从f中读取一行         |
|  f.readlines()  | 从f中读取所有行，作为列表返回 |
|    f.close()    |        关闭f的URL连接         |
|   f.fileno()    |        返回f的文件句柄        |
|    f.info()     |       获取f的MIME头文件       |
|   f.geturl()    |        返回f的真正URL         |

2. *urllib.request.urlretrieve(url,*
  *filename=None,reporthook=None,data=None)*
>urlretrieve（）用于下载完整的HTML

如果提供了reporthook函数，则在每块数据下载或传输完成后调用这个函数。调用使用目前读入的块数、块的字节数和文件的总字节数三个参数。`urlretrieve()`返回一个二元组(local_filename, headers)，local_filename是含有下载数据的本地文件名，headers是Web服务器响应后返回的一系列MIME文件头。

### HTTP验证示例
> 需要先启动本地的tomcat并访问tomcat地址

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

import urllib.request
import urllib.error
import urllib.parse

# 初始化过程
# 后续脚本使用的常量
LOGIN = 'wesly'
PASSWD = "you'llNeverGuess"
URL = 'http://localhost:8080/docs/setup.html'
REALM = 'Secure Archive'


def handler_version(url):
    # 分配了一个基本处理程序类，添加了验证信息。
    # 用该处理程序建立一个URL开启器
    # 安装该开启器以便所有已打开的URL都能用到这些验证信息
    hdlr = urllib.request.HTTPBasicAuthHandler()
    hdlr.add_password(REALM,
                      urllib.parse.urlparse(url)[1],
                      LOGIN,
                      PASSWD)
    opener = urllib.request.build_opener(hdlr)
    urllib.request.install_opener(opener=opener)
    return url


def request_version(url):
    # 创建了一个Request对象，在HTTP请求中添加了简单的base64编码的验证头
    # 该请求用来替换其中的URL字符串
    from base64 import encodebytes
    req = urllib.request.Request(url)
    b64str = encodebytes(bytes('%s %s' % (LOGIN, PASSWD), 'utf-8'))[:-1]
    req.add_header("Authorization", 'Basic %s' % b64str)
    return req


for funcType in ('handler', 'request'):
    # 用两种技术分别打开给定的URL，并显示服务器返回的HTML页面的第一行
    print('***Using %s:' % funcType.upper())
    url = eval('%s_version' % funcType)(URL)
    f = urllib.request.urlopen(url)
    print(str(f.readline(), 'utf-8'))
    f.close()
```

* 输出结果

```
***Using HANDLER:
<!DOCTYPE html SYSTEM "about:legacy-compat">

***Using REQUEST:
<!DOCTYPE html SYSTEM "about:legacy-compat">
```

## Web客户端
一个稍微复杂的Web客户端例子就是 *网络爬虫*。这些程序可以为了不同目的在因特网上探索和下载页面。
> 通过起始地址(URL)，下载该页面和其他后续连接页面，但是仅限于那些与开始页面有相同域名的页面。

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 导入相关的包，其中bs4中的BeautifulSoup负责解析html文档
import os
import sys
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup


class Retriever(object):
    """
    从Web下载页面，解析每个文档中的连接并在必要的时候把它们加入"to-do"队列。
    __slots__变量表示实例只能拥有self.url和self.file属性
    """
    __slots__ = ('url', 'file')

    def __init__(self, url):
        """
        创建Retriever对象时调用，将get_file()返回的URL字符串和对
        应的文件名作为实例属性存储起来
        :param url: 需要抓取的连接
        """
        self.url, self.file = self.get_file(url)

    def get_file(self, url, default='index.html'):
        """
         把指定的URL转换成本地存储的更加安全的文件，即从Web上下载这个文件
        :param url: 指定URL获取页面
        :param default: 默认的文件名
        :return: 返回url和对应的文件名
        """
        # 将URL的http://前缀移除，丢掉任何为获取主机名
        # 而附加的额外信息，如用户名、密码和端口号
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.split('@')[-1].split(':')[0]
        # 将字符进行解码，连接域名创建文件名
        filepath = '%s%s' % (host, urllib.parse.unquote(parsed.path))
        if not os.path.splitext(parsed.path)[1]:
            # 如果URL没有文件扩展名后这将default文件加上
            filepath = os.path.join(filepath, default)
        # 获取文件路径
        linkdir = os.path.dirname(filepath)
        if not os.path.isdir(linkdir):
            # 如果linkdir不是一个目录
            if os.path.exists(linkdir):
                # 如果linkdir存在则删除
                os.unlink(linkdir)
            # 创建同名目录
            os.makedirs(linkdir)
        return url, filepath

    def download(self):
        """
        通过给定的连接下载对应的页面，并将url作为参数调用urllib.urlretrieve()
        将其另存为文件名。如果出错返回一个以'*'开头的错误提示串
        :return: 文件名
        """
        try:
            retval = urllib.request.urlretrieve(self.url, filename=self.file)
        except IOError as e:
            retval = (('***ERROR: bad URL "%s": %s' % (self.url, e)),)
        return retval

    def parse_links(self):
        """
        通过BeautifulSoup解析文件，查看文件包含的额外连接。
        :return: 文件中包含连接的集合
        """
        with open(self.file, 'r', encoding='utf-8') as f:
            data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        parse_links = []
        for x in soup.find_all('a'):
            if 'href' in x.attrs:
                parse_links.append(x['href'])
        return parse_links


class Crawler(object):
    """
    管理Web站点的完整抓取过程。添加线程则可以为每个待抓取的站点分别创建实例
    """
    # 用于保持追踪从因特网上下载下来的对象数目。没成功一个递增1
    count = 0

    def __init__(self, url):
        """
        self.q 是待下载的连接队列，这个队列在页面处理完毕时缩短，每个页面中发现新的连接则增长
        self.seen 是已下载连接的集合
        self.dom 用于存储主链接的域名，并用这个值判定后续连接的域名与主域名是否一致
        :param url: 抓取的url
        """
        self.q = [url]
        self.seen = set()
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.split('@')[-1].split(':')[0]
        self.dom = '.'.join(host.split('.')[-2:])

    def get_page(self, url, media=False):
        """
        用于下载页面并记录连接信息
        :param url:
        :param media:
        :return:
        """
        # 实例化Retriever类并传入需要抓取的连接
        # 下在对应连接并取到文件名
        r = Retriever(url)
        fname = r.download()[0]
        if fname[0] == '*':
            print(fname, '....skipping parse')
            return
        Crawler.count += 1
        print('\n(', Crawler.count, ')')
        print('URL:', url)
        print('FILE:', fname)
        self.seen.add(url)
        # 跳过所有非Web页面
        ftype = os.path.splitext(fname)[1]
        if ftype not in ('.htm', '.html'):
            return
        for link in r.parse_links():
            if link.startswith('mailto:'):
                print('...discarded , mailto link')
                continue

            if not media:
                ftype = os.path.splitext(link)[1]
                if ftype in ('.mp3', '.mp4', '.m4av', '.wav'):
                    print('... discarded, media file')
                    continue

            if not link.startswith('http://') and ':' not in link:
                link = urllib.parse.quote(link, safe='#')
                link = urllib.parse.urljoin(url, link)
            print('*', link)
            if link not in self.seen:
                if self.dom not in link:
                    print('... discarded, not in domain')
                else:
                    # 如果没有下载过并且是属于该网站就加入待下载列表
                    if link not in self.q:
                        self.q.append(link)
                        print('...New, added to Q')
                    else:
                        print('...discarded, already in Q')
            else:
                print('...discarded, already processed')

    def go(self, media=False):
        """
        处理所有待下载连接
        :param media:
        :return:
        """
        while self.q:
            url = self.q.pop()
            self.get_page(url, media)


def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        try:
            url = input('Enter starting URL:')
        except (KeyboardInterrupt, EOFError):
            url = ''
    if not url:
        return
    if not url.startswith('http://') and not url.startswith('ftp://') and not url.startswith('https://'):
        url = 'http://%s' % url

    robot = Crawler(url)
    robot.go()


if __name__ == '__main__':
    main()
```

### 解析Web页面
[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html)是解析页面的常用库，这个库不是标准库，需要单独下载。其使用可以参照上例中的代码。

### 可编程的Web浏览
可以使用[MechanicalSoup](https://pypi.python.org/pypi/MechanicalSoup/)用来模拟浏览器。
