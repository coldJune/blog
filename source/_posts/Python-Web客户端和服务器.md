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

* **urllib.parse.urlparse(urlstring, scheme='',allow_fragments=True)**
```Python
urlparse('http://coldjune.com/categories/')
# 输出结果
ParseResult(scheme='http', netloc='coldjune.com', path='/categories/', params='', query='', fragment='')
```

* **urllib.parse.urlunparse(parts)**
```Python
urlunparse(('http', 'coldjune.com', '/categories/', '', '', ''))
# 输出结果
'http://coldjune.com/categories/'
```
* **urllib.parse.urljoin(base,url,allow_fragments=True)**
```Python
# 如果是绝对路径将整个替换除根域名以外的所有内容
urljoin('http://coldjune.com/categories/1.html','/tags/2.html')
# 输出结果
'http://coldjune.com/tags/2.html'

# 如果是相对路径将会将末端文件去掉与心得url连接
urljoin('http://coldjune.com/categories/1.html','tags/2.html')
# 输出结果
'http://coldjune.com/categories/tags/2.html'
```

* **urllib.parse.quote(string,safe='/',encoding=None,errors=None)**
> 逗号、下划线、句号、斜线和字母数字这类符号不需要转换，其他均需转换。URL不能使用的字符前面会被加上百分号(%)同时转换为十六进制(%xx,xx表示这个字母的十六进制)

```Python
quote('http://www.~coldjune.com/tag categoriese?name=coold&search=6')
# 输出结果
'http%3A//www.%7Ecoldjune.com/tag%20categoriese%3Fname%3Dcoold%26search%3D6'
```

* **urllib.parse.unquote(string,encoding='utf-8',errors='replace')**
```Python
unquote('http%3A//www.%7Ecoldjune.com/tag%20categoriese%3Fname%3Dcoold%26search%3D6')
# 输出结果
'http://www.~coldjune.com/tag categoriese?name=coold&search=6'
```

* **urllib.parse.quote_plus(string,safe='',encoding,errors)**
```Python
quote_plus('http://www.~coldjune.com/tag categoriese?name=coold&search=6')
# 输出结果
'http%3A%2F%2Fwww.%7Ecoldjune.com%2Ftag+categoriese%3Fname%3Dcoold%26search%3D6'
```

* **urllib.parse.unquote_plus(string,encoding='utf-8',errors='replace')**
```Python
unquote_plus('http%3A%2F%2Fwww.%7Ecoldjune.com%2Ftag+categoriese%3Fname%3Dcoold%26search%3D6')
# 输出结果
'http://www.~coldjune.com/tag categoriese?name=coold&search=6'
```

* **urllib.parse.urlencode(query,doseq=False,safe='',encoding=None,errors=None,quote_via=quote_plus)**
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

#### **urllib.request.urlopen(url, data=None, [timeout,]*,cafile=None, capath=None,cadefault=False,context=None)**
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

#### **urllib.request.urlretrieve(url,filename=None,reporthook=None,data=None)**
>urlretrieve（）用于下载完整的HTML

如果提供了reporthook函数，则在每块数据下载或传输完成后调用这个函数。调用使用目前读入的块数、块的字节数和文件的总字节数三个参数。`urlretrieve()`返回一个二元组(local_filename, headers)，local_filename是含有下载数据的本地文件名，headers是Web服务器响应后返回的一系列MIME文件头。
