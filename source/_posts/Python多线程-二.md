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
