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
### 锁示例
锁有两种状态:**锁定** 和 **未锁定**。同时它也支持两个函数：**获得锁** 和 **释放锁**。当多线程争夺锁时，允许第一个获得锁的线程进入临界区，并执行。之后到达的线程被阻塞，直到第一个线程执行结束，退出临界区，并释放锁。其他等待的线程随机获得锁并进入临界区。

* 锁和更多的随机性

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

### 信号量示例
对于拥有有限资源的应用来说，可以使用信号量的方式来代替锁。**信号量** 是一个计数器，当资源消耗时递减，当资源释放时递增。信号量比锁更加灵活，因为可以有多个线程，每个线程拥有有限资源的一个实例。消耗资源使计数器递减的操作成为`P()`，当一个线程对一个资源完成操作时，该资源返回资源池的操作称为`V()`。

* 糖果机和信号量

>  这个特制的机器只有5个可用的槽来保持库存。如果所有槽都满了，糖果不能再加入这个机器中；如果每个槽都空了，想要购买的消费者无法买到糖果。使用信号量来跟踪这些有限的资源

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 导入相应的模块和信号量类
# BoundedSemaphore的额外功能是这个计数器的值永远不会超过它的初始值
# 它可以防范其中信号量释放次数多余获得次数的异常用例
from atexit import register
from random import randrange
from threading import BoundedSemaphore, Lock, Thread
from time import sleep, ctime

# 全局变量
# 锁
# 库存商品最大值的常量
# 糖果托盘
lock = Lock()
MAX = 5
candytray = BoundedSemaphore(MAX)


def refill():
    # 当虚构的糖果机所有者向库存中添加糖果时执行
    # 代码会输出用户的行动，并在某人添加的糖果超过最大库存是给予警告
    lock.acquire()
    print('Refilling candy...')
    try:
        candytray.release()
    except ValueError:
        print('full, skipping')
    else:
        print('OK')
    lock.release()


def buy():
    # 允许消费者获取一个单位的库存
    lock.acquire()
    print('Buying candy....')
    # 检测是否所有资源都已经消费完了
    # 通过传入非阻塞的标志False，让调用不再阻塞，而在应当阻塞的时候返回一个False
    # 指明没有更多资源
    if candytray.acquire(False):
        print('OK')
    else:
        print('Empty, skipping')
    lock.release()


def producer(loops):
    for i in range(loops):
        refill()
        sleep(randrange(3))


def consumer(loops):
    for i in range(loops):
        buy()
        sleep(randrange(3))


def _main():
    print('starting at:', ctime())
    nloops = randrange(2, 6)
    print('THE CANDY MACHINE (full with %d bars)' % MAX)
    Thread(target=consumer, args=(randrange(nloops, nloops+MAX+2),)).start()
    Thread(target=producer, args=(nloops,)).start()


@register
def _atexit():
    print('all DONE at:', ctime())


if __name__ == '__main__':
    _main()
```

* 输出结果

```
starting at: Tue Feb 27 14:48:31 2018
THE CANDY MACHINE (full with 5 bars)
Buying candy....
OK
Refilling candy...
OK
Refilling candy...
full, skipping
Buying candy....
OK
Refilling candy...
OK
Buying candy....
OK
Refilling candy...
OK
Refilling candy...
full, skipping
Buying candy....
OK
Buying candy....
OK
Buying candy....
OK
Buying candy....
OK
Buying candy....
OK
Buying candy....
Empty, skipping
all DONE at: Tue Feb 27 14:48:42 2018
```

## 生产者-消费者问题和queue模块
生产商品的时间是不确定的，消费生产者生产的商品的时间也是不确定的。在这个场景下将其放在类似队列的数据结构中。
[queue模块](https://docs.python.org/3/library/queue.html)来提供线程间通信的机制，从而让线程之间可以互相分享数据。具体而言就是创建一个队列，让生产者在其中放入新的商品，而消费者消费这些商品

### queue模块常用属性

|               属性                |                                                                          描述                                                                           |
|:---------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         Queue(maxsize=0)          |                              创建一个先入先出队列。如果给定最大值，则在队列没有空间时阻塞，否则(没有指定最大值),为无限队列                              |
|       LifoQueue(maxsize=0)        |                              创建一个后入先出队列。如果给定最大值，则在队列没有空间时阻塞，否则(没有指定最大值),为无限队列                              |
|      PriorityQueue(maxsize)       |                               创建一个优先级队列。如果给定最大值，则在队列没有空间时阻塞，否则(没有指定最大值),为无限队列                               |
|             queue异常             |                                                                                                                                                         |
|               Empty               |                                                           当对空队列调用get*()方法时抛出异常                                                            |
|               Full                |                                                         当对已满的队列调用put*()方法时抛出异常                                                          |
|           queue对象方法           |                                                                                                                                                         |
|              qsize()              |                                          返回队列大小(由于返回时队列大小可能被其他线程修改，所以改值为近似值)                                           |
|              empty()              |                                                        如果队列为空，则返回True；否则，返回False                                                        |
|              full()               |                                                        如果队列已满，则返回True；否则，返回False                                                        |
| put(item,block=True,timeout=None) | 将item放入队列。如果block为True(默认)且timeout为None，则在有可用空间之前阻塞；如果timeout为正值，则最多阻塞timeout秒；如果block为False，则抛出Empty异常 |
|           put_nowait()            |                                                                  和put(item,False)相同                                                                  |
|   get(block=True,timeout=None)    |                                          从队列中取得元素，如果给定了block(非0)，则一直阻塞到有可用的元素为止                                           |
|           get_nowait()            |                                                                    和get(False)相同                                                                     |
|            task_done()            |                                             用于标识队列中的某个元素已执行完成，该方法会被下面的join()使用                                              |
|              join()               |                                            在队列中所有元素执行完毕并调用上面的task_done()信号之前，保持阻塞                                            |

### 生产者消费者问题

使用了Queue对象，以及随机生产(消费)的商品的数量。生产者和消费者独立且并发地执行线程

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 使用queue.Queue对象和之前的myThread.MyThread线程类
from random import randint
from time import sleep
from queue import Queue
from myThread import MyThread


def writeQ(queue):
    # 将一个对象放入队列中
    print('producing object for Q...')
    queue.put('xxx', 1)
    print('size now', queue.qsize())


def readQ(queue):
    # 消费队列中的一个对象
    val = queue.get(1)
    print('consumed object from Q... size now', queue.qsize())


def writer(queue, loops):
    # 作为单个线程运行
    # 向队列中放入一个对象，等待片刻，然后重复上述步骤
    # 直至达到脚本执行时随机生成的次数没值
    for i in range(loops):
        writeQ(queue)
        # 睡眠的随机秒数比reader短是为了阻碍reader从空队列中获取对象
        sleep(randint(1, 3))


def reader(queue, loops):
    # 作为单个线程运行
    # 消耗队列中一个对象，等待片刻，然后重复上述步骤
    # 直至达到脚本执行时随机生成的次数没值
    for i in range(loops):
        readQ(queue)
        sleep(randint(2, 5))

# 设置派生和执行的线程总数
funcs = [writer, reader]
nfuncs = range(len(funcs))


def main():
    nloops = randint(2, 5)
    q = Queue(32)
    threads = []
    for i in nfuncs:
        t = MyThread(funcs[i], (q, nloops), funcs[i].__name__)
        threads.append(t)

    for i in nfuncs:
        threads[i].start()

    for i in nfuncs:
        threads[i].join()

    print('all DONE')

if __name__ == '__main__':
    main()

```

* 输出结果

```
starting at: Tue Feb 27 15:17:16 2018
producing object for Q...
size now 1
starting at: Tue Feb 27 15:17:16 2018
consumed object from Q... size now 0
producing object for Q...
size now 1
producing object for Q...
size now 2
done at: Tue Feb 27 15:17:20 2018
consumed object from Q... size now 1
consumed object from Q... size now 0
done at: Tue Feb 27 15:17:26 2018
all DONE
```

## 线程的替代方案
[subprocess模块](https://docs.python.org/3/library/subprocess.html?highlight=subprocess#module-subprocess)
[multiprocessing模块](https://docs.python.org/3/library/multiprocessing.html?highlight=multiprocessing#module-multiprocessing)
[concurrent.futures模块](https://docs.python.org/3/library/concurrent.futures.html?highlight=concurrent%20futures#module-concurrent.futures)
