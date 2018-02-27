---
title: Python多线程(二)
date: 2018-02-26 22:16:16
categories: Python
copyright: true
tags:
    - Python
    - 多线程
description:
---
在上篇主要对线程的概念做了一个简要的介绍，同时介绍了_thread模块和threading模块的使用方法，通过几个简短的程序实现了线程的调用。这篇将会记录一些多线程简单的应用以及相关生产者和消费者的问题。
<!--More-->
## 多线程实践
Python虚拟机是单线程（GIL）的原因，只有线程在执行I/O密集型的应用时才会更好地发挥Python的并发性。
下面的例子是通过多线程下载图书排名信息的调用

```Python
#!/usr/bin/python3
# -*-  coding:UTF-8 -*-

from atexit import register
import re
import threading
import time
import urllib.request

# 匹配排名的正则表达式
# 亚马逊的网站
REGEX = re.compile(b'#([\d,]+) in Books')
AMZN = 'https://www.amazon.com/dp/'

# ISBN编号和书名
ISBNs = {
    '0132269937': 'Core Python Programming',
    '0132356139': 'Python Web Development with Django',
    '0137143419': 'Python Fundamentals'
}

# 请求头
# 因为亚马逊会检测爬虫,所以需要加上请求头伪装成浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36 TheWorld 7'
}


def get_ranking(isbn):
    # 爬取网页,获取数据
    # 使用str.format()格式化数据
    url = '{0}{1}'.format(AMZN, isbn)
    # 爬取网页并解析
    req = urllib.request.Request(url, headers=headers)
    page = urllib.request.urlopen(req)
    data = page.read()
    page.close()
    return str(REGEX.findall(data)[0], 'utf-8')


def _show_ranking(isbn):
    # 显示结果
    print('- %r ranked %s' % (ISBNs[isbn], get_ranking(isbn)))


def _main():
    print('At', time.ctime(), 'on Amazon...')
    for isbn in ISBNs:
        (threading.Thread(target=_show_ranking, args=(isbn,))).start()
        #_show_ranking(isbn)


@register
def _atexit():
    # 注册一个退出函数，在脚本退出先请求调用这个函数
    print('all DONE at:', time.ctime())

if __name__ == '__main__':
    _main()
```

* 输出结果

```
At Tue Feb 27 10:40:51 2018 on Amazon...
- 'Python Fundamentals' ranked 4,358,513
- 'Python Web Development with Django' ranked 1,354,091
- 'Core Python Programming' ranked 458,510
all DONE at: Tue Feb 27 10:42:39 2018
```
## 锁实例
所有两种状态:**锁定** 和 **未锁定**。同时它也支持两个函数：**获得锁** 和 **释放锁**。当多线程争夺锁时，允许第一个获得锁的线程进入临界区，并执行。之后到达的线程被阻塞，知道第一个线程执行结束，退出临界区，并释放锁。其他等待的线程随机获得锁并进入临界区。

### 锁和更多的随机性

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

from __future__ import with_statement
from atexit import  register
from random import randrange
from threading import Thread, Lock, current_thread
from time import sleep, ctime


class CleanOutputSet(set):
    # 集合的子类，将默认输出改变为将其所有元素
    # 按照逗号分隔的字符串
    def __str__(self):
        return ', '.join(x for x in self)


# 锁
# 随机数量的线程(3~6)，每个线程暂停或睡眠2~4秒
lock = Lock()
loops = (randrange(2, 5) for x in range(randrange(3, 7)))
remaining = CleanOutputSet()


def loop(sec):
    # 获取当前执行的线程名，然后获取锁并保存线程名
    myname = current_thread().name
    lock.acquire()
    remaining.add(myname)
    print('[%s] Started %s' % (ctime(), myname))
    # 释放锁并睡眠随机秒
    lock.release()
    sleep(sec)
    # 重新获取锁，输出后再释放锁
    lock.acquire()
    remaining.remove(myname)
    print('[%s] Completed %s (%d sec)' % (ctime(), myname, sec))
    print('     (remaining: %s)' % (remaining or 'NONE'))
    lock.release()


def loop_with(sec):
    myname = current_thread().name
    with lock:
        remaining.add(myname)
        print('[%s] Started %s' % (ctime(), myname))
    sleep(sec)
    with lock:
        remaining.remove(myname)
        print('[%s] Completed %s (%d sec)' % (ctime(), myname, sec))
        print('     (remaining: %s)' % (remaining or 'NONE'))


def _main():
    for pause in loops:
        # Thread(target=loop, args=(pause,)).start()
        Thread(target=loop_with, args=(pause,)).start()



@register
def _atexit():
    print('all DONE at:', ctime())


if __name__ == '__main__':
    _main()

```
* 输出结果
loop方法
```
[Tue Feb 27 11:26:13 2018] Started Thread-1
[Tue Feb 27 11:26:13 2018] Started Thread-2
[Tue Feb 27 11:26:13 2018] Started Thread-3
[Tue Feb 27 11:26:13 2018] Started Thread-4
[Tue Feb 27 11:26:13 2018] Started Thread-5
[Tue Feb 27 11:26:13 2018] Started Thread-6
[Tue Feb 27 11:26:15 2018] Completed Thread-2 (2 sec)
     (remaining: Thread-3, Thread-4, Thread-1, Thread-5, Thread-6)
[Tue Feb 27 11:26:15 2018] Completed Thread-6 (2 sec)
     (remaining: Thread-3, Thread-4, Thread-1, Thread-5)
[Tue Feb 27 11:26:16 2018] Completed Thread-3 (3 sec)
     (remaining: Thread-4, Thread-1, Thread-5)
[Tue Feb 27 11:26:16 2018] Completed Thread-4 (3 sec)
     (remaining: Thread-1, Thread-5)
[Tue Feb 27 11:26:16 2018] Completed Thread-5 (3 sec)
     (remaining: Thread-1)
[Tue Feb 27 11:26:17 2018] Completed Thread-1 (4 sec)
     (remaining: NONE)
all DONE at: Tue Feb 27 11:26:17 2018
```
loop_with方法
```
[Tue Feb 27 11:43:15 2018] Started Thread-1
[Tue Feb 27 11:43:15 2018] Started Thread-2
[Tue Feb 27 11:43:15 2018] Started Thread-3
[Tue Feb 27 11:43:15 2018] Started Thread-4
[Tue Feb 27 11:43:15 2018] Started Thread-5
[Tue Feb 27 11:43:15 2018] Started Thread-6
[Tue Feb 27 11:43:17 2018] Completed Thread-3 (2 sec)
     (remaining: Thread-1, Thread-5, Thread-4, Thread-6, Thread-2)
[Tue Feb 27 11:43:17 2018] Completed Thread-6 (2 sec)
     (remaining: Thread-1, Thread-5, Thread-4, Thread-2)
[Tue Feb 27 11:43:17 2018] Completed Thread-5 (2 sec)
     (remaining: Thread-1, Thread-4, Thread-2)
[Tue Feb 27 11:43:18 2018] Completed Thread-1 (3 sec)
     (remaining: Thread-4, Thread-2)
[Tue Feb 27 11:43:18 2018] Completed Thread-4 (3 sec)
     (remaining: Thread-2)
[Tue Feb 27 11:43:18 2018] Completed Thread-2 (3 sec)
     (remaining: NONE)
all DONE at: Tue Feb 27 11:43:18 2018
```
