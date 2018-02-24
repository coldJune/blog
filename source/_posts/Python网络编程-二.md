---
title: Python网络编程(二)
date: 2018-02-24 09:40:06
categories: Python
copyright: true
tags:
    - Python
    - 网络编程
description:
---
上篇对Python中的socket模块的简单应用做了描述和记录，下面便是对SocketServer模块和Twisted框架做一个简要的记录
<!--More-->
## socketserver模块
[socketserver](https://docs.python.org/3/library/socketserver.html?highlight=socketserver#module-socketserver)是标准库的一个高级模块，它的目标是简化很多样板代码，它们是创建网络客户端和服务器所必需的代码。

### socketserver模块类
|                     类                      |                                                              描述                                                              |
|:-------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|
|                 BaseServer                  |        包含核心服务器功能和mix-in类的钩子；仅用于推导，这样不会创建这个类的实例；可以用TCPServer或UDPServer创建类的实例        |
|             TCPServer/UDPServer             |                                                  基础的网络同步TCP/UDP服务器                                                   |
|     UnixStreamServer/UnixDatagramServer     |                                                基于文件的基础同步TCP/UDP服务器                                                 |
|         ForkingMixIn/ThreadingMixIn         |                    核心派出或线程功能；只用作mix-in类与一个服务器类配合实现一些异步性；不能直接实例化这个类                    |
|      ForkingTCPServer/ForkingUDPServer      |                                            ForkingMaxIn和TCPServer/UDPServer的组合                                             |
|    ThreadingTCPServer/ThreadingUDPServer    |                                           ThreadingMixIn和TCPServer/UDPServer的组合                                            |
|             BaseRequestHandler              | 包含处理服务请求的核心功能；仅用于推导，无法创建这个类的实例；可以使用StreamRequestHandler或DatagramRequestHandler创建类的实例 |
| StreamRequestHandler/DatagramRequestHandler |                                                 实现TCP/UDP服务器的服务处理器                                                  |

### socketserver TCP服务器/客户端
在原始服务器循环中，我们阻塞等待请求，当接收到请求时就对其提供服务，然后继续等待。在此处的服务器循环中，并非在服务器中创建代码，而是定义一个处理程序，当服务器接收到一个传入的请求时，服务器就可以调用

#### TCP服务器
  ```Python
    #!usr/bin/python3
    # -*- coding:UTF-8 -*-

    # 导入socketserver相关的类和time.ctime()的全部属性
    from socketserver import (TCPServer as TCP,
                              StreamRequestHandler as SRH)
    from time import ctime

    HOST = ''
    PORT = 12345
    ADDR = (HOST, PORT)


    class MyRequestHandler(SRH):
        # MyRequestHandler继承自StreamRequestHandler

        def handle(self):
            # 重写handle方法，当接收到一个客户端消息是，会调用handle()方法
            print('...connected from:', self.client_address)
            # StreamRequestHandler将输入和输出套接字看做类似文件的对象
            # 所以使用write()将字符串返回客户端，用readline()来获取客户端信息
            self.wfile.write(bytes('[%s] %s' % (
                ctime(), self.rfile.readline().decode('utf-8')), 'utf-8'))

    # 利用给定的主机信息和请求处理类创建了TCP服务器
    # 然后无限循环地等待并服务于客户端请求
    tcpServ = TCP(ADDR, MyRequestHandler)
    print('waiting for connection...')
    tcpServ.serve_forever()
  ```
#### TCP客户端
  ```Python
    #!usr/bin/python3
    # -*- coding:UTF-8 -*-

    from socket import *

    HOST = '127.0.0.1'
    PORT = 12345
    BUFSIZE = 1024
    ADDR = (HOST, PORT)

    while True:
        tcpSocket = socket(AF_INET, SOCK_STREAM)
        tcpSocket.connect(ADDR)
        data = input('> ')
        if not data:
            break
        # 因为处理程序类对待套接字通信像文件一样，所以必须发送行终止符。
        # 而服务器只是保留并重用这里发送的终止符
        tcpSocket.send(bytes('%s\r\n' % data, 'utf-8'))
        data = tcpSocket.recv(BUFSIZE)
        if not data:
            break
        # 得到服务器返回的消息时，用strip()函数对其进行处理并使用print()自动提供的换行符
        print(data.decode('utf-8').strip())
        tcpSocket.close()

  ```
#### socketserver TCP服务器和客户端运行结果
  在客户端启动的时候连接了一次服务器，而每一次发送一个请求连接一次，所以发送了三个请求连接了四次服务器
  * TCP服务器运行结果
      ````
      waiting for connection...
      ...connected from: ('127.0.0.1', 51835)
      ...connected from: ('127.0.0.1', 51877)
      ...connected from: ('127.0.0.1', 51893)
      ...connected from: ('127.0.0.1', 51901)

      ````

  * TCP客户端运行结果
      ```
      > hello
      [Sat Feb 24 10:29:28 2018] hello
      > hello
      [Sat Feb 24 10:29:44 2018] hello
      > hi
      [Sat Feb 24 10:29:50 2018] hi
      >
      ```

## Twisted框架的简单使用
  Twisted是一个完整的事件驱动的网络框架，利用它既能使用也能开发完整的异步网络应用程序和协议。它不是Python标准库的一部分，所以需要单独[下载](https://www.lfd.uci.edu/~gohlke/pythonlibs/#twisted)和安装它[^1]。
  ```
   pip3 install Twisted-17.9.0-cp36-cp36m-win_amd64.whl
  ```
  安装成功显示
  ```
  Processing e:\迅雷下载\twisted-17.9.0-cp36-cp36m-win_amd64.whl
  Requirement already satisfied: Automat>=0.3.0 in e:\python\python36\lib\site-packages (from Twisted==17.9.0)
  Requirement already satisfied: zope.interface>=4.0.2 in e:\python\python36\lib\site-packages (from Twisted==17.9.0)
  Requirement already satisfied: incremental>=16.10.1 in e:\python\python36\lib\site-packages (from Twisted==17.9.0)
  Requirement already satisfied: hyperlink>=17.1.1 in e:\python\python36\lib\site-packages (from Twisted==17.9.0)
  Requirement already satisfied: constantly>=15.1 in e:\python\python36\lib\site-packages (from Twisted==17.9.0)
  Requirement already satisfied: attrs in e:\python\python36\lib\site-packages (from Automat>=0.3.0->Twisted==17.9.0)
  Requirement already satisfied: six in e:\python\python36\lib\site-packages (from Automat>=0.3.0->Twisted==17.9.0)
  Requirement already satisfied: setuptools in e:\python\python36\lib\site-packages (from zope.interface>=4.0.2->Twisted==17.9.0)
  Installing collected packages: Twisted
  Successfully installed Twisted-17.9.0
  ```
### Twisted Reactor TCP 服务器/客户端
#### TCP服务器
  ```Python
  #!usr/bin/python3
  # -*- coding:UTF-8 -*-

  # 常用模块导入，特别是twisted.internet的protocol和reactor
  from twisted.internet import protocol, reactor
  from time import ctime

  # 设置端口号
  PORT = 12345


  class TWServProtocol(protocol.Protocol):
      # 继承Protocol类
      def connectionMade(self):
          # 重写connectionMade()方法
          # 当一个客户端连接到服务器是会执行这个方法
          client = self.client = self.transport.getPeer().host
          print('...connected from:', client)

      def dataReceived(self, data):
          # 重写dataReceived()方法
          # 当服务器接收到客户端通过网络发送的一些数据的时候会调用此方法
          self.transport.write(bytes('[%s] %s' % (
              ctime(), data.decode('utf-8')), 'utf-8'))

  # 创建一个协议工厂，每次得到一个接入连接是，制造协议的一个实例
  # 在reactor中安装一个TCP监听器，以此检查服务请求
  # 当接收到一个请求时，就是创建一个就是创建一个TWServProtocol实例来处理客户端事务
  factory = protocol.Factory()
  factory.protocol = TWServProtocol
  print('waiting for connection...')
  reactor.listenTCP(PORT, factory)
  reactor.run()

  ```
#### TCP客户端
  ```Python
    #!usr/bin/python
    # -*- coding:UTF-8 -*-

    from twisted.internet import  protocol, reactor

    HOST = '127.0.0.1'
    PORT = 12345


    class TWClientProtocol(protocol.Protocol):
        def sendData(self):
            # 需要发送数据时调用
            # 会在一个循环中继续，直到不输入任何内容来关闭连接
            data = input('> ')
            if data:
                print('...send %s...' % data)
                self.transport.write(bytes(data, 'utf-8'))
            else:
                self.transport.loseConnection()

        def connectionMade(self):
            #
            self.sendData()

        def dataReceived(self, data):
            print(data.decode('utf-8'))
            self.sendData()


    class TWClientFactory(protocol.ClientFactory):
        # 创建了一个客户端工厂
        protocol = TWClientProtocol
        clientConnectionLost = clientConnectionFailed = \
            lambda self, connector, reason: reactor.stop()
    # 创建了一个到服务器的连接并运行reactor，实例化了客户端工厂
    # 因为这里不是服务器，需要等待客户端与我们通信
    # 并且这个工厂为每一次连接都创建一个新的协议对象。
    # 客户端创建单个连接到服务器的协议对象，而服务器的工厂则创建一个来与客户端通信
    reactor.connectTCP(HOST, PORT, TWClientFactory())
    reactor.run()
  ```
#### TCP服务器和客户端运行结果
  * 服务器结果
  ```
  waiting for connection...
  ...connected from: 127.0.0.1

  ```
  * 客户端结果
  ```
  > hello
  ...send hello...
  [Sat Feb 24 11:19:49 2018] hello
  > hi
  ...send hi...
  [Sat Feb 24 11:20:02 2018] hi
  >
  ```

[^1]:需要安装python对应的版本和位数
