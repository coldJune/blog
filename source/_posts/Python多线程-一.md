---
title: Python多线程(一)
date: 2018-02-24 16:13:37
categories: Python
copyright: true
tags:
    - Python
    - 多线程
description:
---
多线程编程对于以下编程任务是非常理想的：
* 本质上是异步的
* 需要多个并发活动
* 每个活动的处理顺序可能是不确定的(随机、不可预测的)
<!--Mare-->
使用多线程或者类似Queue的共享数据结构可以将一个串行程序规划成几个执行特定任务的线程
* UserRequestThread: 负责读取客户端输入。程序将创建多个线程，每个客户端一个，客户端的请求将会被放入队列中
* RequestProcessor: 该线程负责从队列中获取请求并进行处理，为第三个线程提供输出
* ReplyThread: 负责向用户输出，将结果传回给用户，或者把数据写到本地文件系统或者数据库中

## 线程和进程

* 进程

  > 计算机程序是储存在磁盘上的可执行二进制(或其他类型)的文件。**进程** （有时称为 **重量级进程**）则是一个执行中的程序。
    每一个进程都拥有自己的地址空间、内存、数据栈以及其他用于跟踪执行的辅助数据。操作系统管理其上的所有进程的执行，并为它们合理地分配时间。
  进程可以通过 **派生**(fork或spawn)新的进程来执行任务,而进程之间的通信只能通过 *进程间通信(IPC)* 的方式共享信息

* 线程

  > **线程**（有时称为 **轻量级进程**）共享相同的上下文。相当于在主进程中并行运行的一些“迷你进程”。当其他线程运行是，它可以被抢占（中断）
    和临时挂起（睡眠），这种做法叫 *让步(yielding)*。早单核CPU系统中，线程的实际规划是：每个线程运行一小会儿，然后让步给其他线程（再次排队
    等待更多的CPU时间）。在整个进程的执行当中，每个线程执行它自己特定的任务，在必要时和其他线程进行结果通信。

## 线程与Python

### 全局解释锁

  对Python虚拟机的访问是由 **全局解释锁(GIL)** 控制的。这个锁用来保证同时只能有一个线程运行。在多线程环境中，Python虚拟机将按照下面的方式执行。
  1. 设置GIL
  2. 切换进一个线程去运行
  3. 执行下面操作之一
      a. 指定数量的字节码指令
      b. 线程主动让出控制权(可以调用time.sleep(0)来完成)
  4. 把线程设置回睡眠状态(切换出线程)
  5. 解锁GIL
  6. 重复上述步骤

  当调用外部代码(即，任意C/C++扩展的内置函数)时，GIL会保持锁定，直至函数执行结束。

### 退出线程

  当一个线程完成函数的执行时，就会退出。还可以通过调用`thread.exit()`或者`sys.exit()`退出进程，或者抛出SystemExit异常，是线程退出。

## \_thread模块

  [\_thread模块](https://docs.python.org/3/library/_thread.html?highlight=_thread#module-_thread)提供了派生线程、基本的同步数据结构(*锁对象(lock object)*,也叫 *原语锁*、*简单锁*、*互斥锁*、*互斥* 和 *二进制信号量*)

### \_thread模和锁对象

|                    函数/方法                    |                             描述                             |
|:-----------------------------------------------:|:------------------------------------------------------------:|
|               \_thread模块的函数                |                                                              |
| start_new_thread(function, args, kwargs = None) | 派生一个新的线程，使用给定的args和可选的kwargs来执行function |
|                 allocate_lock()                 |                      分配LockType锁对象                      |
|                     exit()                      |                        给线程退出命令                        |
|              LockType锁对象的方法               |                                                              |
|              acquire(wait = None)               |                        尝试获取锁对象                        |
|                    locked()                     |         如果获取了锁对象则返回True，否则，返回False          |
|                    release()                    |                            释放锁                            |

#### 使用线程

* 一般方式

  - 程序

    ```Python
    #!usr/bin/python3
    # -*- coding:UTF-8 -*-

    import _thread
    from time import ctime, sleep


    def loop_0():
        print('start loop_0 at:', ctime())
        sleep(4)
        print('loop_0 done at:', ctime())


    def loop_1():
        print('start loop_1 at:', ctime())
        sleep(2)
        print('loop_1 done at:', ctime())


    def main():
        print('starting at:', ctime())
        # start_new_thread 方法即使要执行的
        # 函数不需要参数，也需要传递一个空元组
        _thread.start_new_thread(loop_0, ())
        _thread.start_new_thread(loop_1, ())
        # 阻止主线程的执行，保证其最后执行，
        # 后续去掉这种方式，引入锁的方式
        sleep(6)
        print('all done at', ctime())


    if __name__ == '__main__':
        main()

    ```
  - 执行结果

    在主线程中同时开启了两个线程，loop_1()由于只睡眠了2s，所以先执行完，其实执行完loo_0()，线程执行的总时间是最慢的那个线程(*loop_0()* )
    的运行时间
    ```
    starting at: Mon Feb 26 08:52:10 2018
    start loop_0 at: Mon Feb 26 08:52:10 2018
    start loop_1 at: Mon Feb 26 08:52:10 2018
    loop_1 done at: Mon Feb 26 08:52:12 2018
    loop_0 done at: Mon Feb 26 08:52:14 2018
    all done at Mon Feb 26 08:52:16 2018

    ```

* 使用锁对象

  - 程序
    ```Python
    #!usr/bin/python3
    # -*- coding:UTF-8 -*-

    import _thread
    from time import ctime, sleep

    loops = [4, 2]


    def loop(nloop, sec, lock):
        # nloop: 第几个线程
        # sec: 时间
        # lock: 分配的锁
        print('start loop', nloop, 'at:', ctime())
        sleep(sec)
        print('loop', nloop, 'done at:', ctime())
        # 当时间到了的时候释放锁
        lock.release()


    def main():
        print('starting at:', ctime())
        locks = []
        nloops = range(len(loops))

        for i in nloops:
            # 生成锁对象

            # 通过allocate_lock()函数得到锁对象
            # 通过acquire()取到每个锁
            # 添加进locks列表
            lock = _thread.allocate_lock()
            lock.acquire()
            locks.append(lock)

        for i in nloops:
            # 派生线程

            # 传递循环号，时间、锁对象
            _thread.start_new_thread(loop, (i, loops[i], locks[i]))

        for i in nloops:
            # 等待所有线程的锁都释放完了才执行主线程
            while locks[i].locked():
                pass

        print('all DONE at:', ctime())

    if __name__ == '__main__':
        main()

    ```
  - 执行结果

    未再设置时间等待所有线程执行结束，而是在线程全部结束后马上运行主线程代码

    ```
    starting at: Mon Feb 26 09:37:39 2018
    start loop 1 at: Mon Feb 26 09:37:39 2018
    start loop 0 at: Mon Feb 26 09:37:39 2018
    loop 1 done at: Mon Feb 26 09:37:41 2018
    loop 0 done at: Mon Feb 26 09:37:43 2018
    all DONE at: Mon Feb 26 09:37:43 2018
    ```
