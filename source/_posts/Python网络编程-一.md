---
title: Python网络编程(一)
date: 2018-02-22 15:53:42
categories: Python
copyright: true
tags:
    - Python
    - 网络编程
description:
---
使用Python的一些模块来创建网络应用程序
<!--More-->
## socket()函数模块
要创建套接字，必须使用`socket.socket()`函数`socket(socket_family, socket_type, protocol = 0)`,其中`socket_family`是 *AF_UNIX*或 *AF_INET*,`socket_type`是 *SOCK_STREAM* 或 *SOCK_DGRAM*。[^1]`protocol`通常省略，默认为0。

* >创建TCP/IP套接字

    ```Python
    tcpSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ```

* >创建UDP/IP套接字

    ```Python
   udpSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ```

### 套接字对象内接方法
|         名称         |                                 描述                                  |
|:--------------------:|:---------------------------------------------------------------------:|
|   服务器套接字方法   |                                                                       |
|       s.bind()       |                将地址(主机名、端口号对)绑定到套接字上                 |
|      s.listen()      |                          设置并启动TCP监听器                          |
|      s.accept()      |           被动接受TCP客户端连接，一直等待知道连接到达(阻塞)           |
|   客户端套接字方法   |                                                                       |
|     s.connect()      |                         主动发起TCP服务器连接                         |
|    s.connect_ex()    | connect()的扩展版本，此时会以错误码的形式返回问题，而不是抛出一个异常 |
|   普通的套接字方法   |                                                                       |
|       s.recv()       |                              接受TCP消息                              |
|    s.recv_into()     |                       接受TCP消息到指定的缓冲区                       |
|       s.send()       |                              发送TCP消息                              |
|     s.sendall()      |                           完整地发送TCP消息                           |
|     s.recvfrom()     |                              接受UDP消息                              |
|  s.recvfrom_into()   |                       接受UDP消息到指定的缓冲区                       |
|      s.sendto()      |                              发送UDP消息                              |
|   s.getpeername()    |                      连接到套接字(TCP)的远程地址                      |
|   s.getsockname()    |                           当前套接字的地址                            |
|    s.getsockopt()    |                        返回给定套接字选项的值                         |
|    s.setsockopt()    |                        设置给定套接字选项的值                         |
|     s.shutdown()     |                               关闭连接                                |
|      s.close()       |                              关闭套接字                               |
|      s.detach()      |         在未关闭文件描述符的情况下关闭套接字，返回文件描述符          |
|      s.ioctl()       |                    控制套接字的模式(仅支持Windows)                    |
| 面向阻塞的套接字方法 |                                                                       |
|   s.setblocking()    |                     设置套接字的阻塞或非阻塞模式                      |
|    s.settimeout()    |                     设置阻塞套接字操作的超时时间                      |
|    s.gettimeout()    |                     获取阻塞套接字操作的超时时间                      |
| 面向文件的套接字方法 |                                                                       |
|      s.fileno()      |                          套接字的文件描述符                           |
|     s.makefile()     |                      创建与套接字关联的文件对象                       |
|       数据属性       |                                                                       |
|       s.family       |                              套接字家族                               |
|        s.type        |                              套接字类型                               |
|       s.proto        | 套接字协议                                                                      |

### socket模块属性
|                    属性名称                     |                                       描述                                       |
|:-----------------------------------------------:|:--------------------------------------------------------------------------------:|
|                    数据属性                     |                                                                                  |
| AF_UNIX、AF_INET、AF_INET6、AF_NETLINK、AF_TIPC |                           Python中支持的套接字地址家族                           |
|               SO_STREAM、SO_DGRAM               |                          套接字类型(TCP=流，UDP=数据报)                          |
|                    has_ipv6                     |                            指示是否支持IPv6的布尔标记                            |
|                      异常                       |                                                                                  |
|                      error                      |                                  套接字相关错误                                  |
|                     herror                      |                                主机和地址相关错误                                |
|                    gaierror                     |                                   地址相关错误                                   |
|                     timeout                     |                                     超时时间                                     |
|                      函数                       |                                                                                  |
|                    socket()                     |         以给定的地址家族、套接字类型和协议类型(可选) 创建一个套接字对象          |
|                  socketpair()                   |         以给定的地址家族、套接字类型和协议类型(可选) 创建一个套接字对象          |
|               create_connection()               |            常规函数，它接收一个地址(主机号，端口号)对，返回套接字对象            |
|                    fromfd()                     |                     以一个打开的文件描述符创建一个套接字对象                     |
|                      ssl()                      |                通过套接字启动一个安全套接字层连接；不执行证书验证                |
|                  getaddrinfo()                  |                         获取一个五元组序列形式的地址信息                         |
|                  getnameinfo()                  |                  给定一个套接字地址，返回(主机名，端口号)二元组                  |
|                    getfqdn()                    |                                  返回完整的域名                                  |
|                  gethostname()                  |                                  返回当前主机名                                  |
|                 gethostbyname()                 |                           将一个主机名映射到它的IP地址                           |
|               gethostbyname_ex()                |        gethostbyname()的扩展版本，它返回主机名、别名主机集合和IP地址列表         |
|                 gethostbyaddr()                 |         讲一个IP地址映射到DNS信息；返回与gethostbyname_ex()相同的三元组          |
|                getprotobyname()                 |                       将一个协议名(如‘TCP’)映射到一个数字                        |
|         getservbyname()/getservbyport()         | 将一个服务名映射到一个端口号，或者反过来；对于任何一个函数来说，协议名都是可选的 |
|                 ntohl()/ntohs()                 |                         将来自网络的整数装换为主机字节序                         |
|                 htonl()/htons()                 |                         将来自主机的整数转换为网络字节序                         |
|             inet_aton()/inet_ntoa()             |        将IP地址八进制字符串转换成32位的包格式，或者反过来(仅用于IPv4地址)        |
|             inet_pton()/inet_ntop()             |      将IP地址字符串转换成打包的二进制格式，或者反过来(同时适用于IPv4和IPv6)      |
|     getdefaulttimeout()/setdefaulttimeout()     | 以秒(浮点数)为单位返回默认套接字超时时间；以秒(浮点数)为单位设置默认套接字超时时间                                                                                 |

详情参阅[socket模块文档](https://docs.python.org/3/library/socket.html?highlight=socket#module-socket)
## 创建TCP服务器/客户端

### TCP服务器
* 下面是TCP服务器端的通用伪码，这是设计服务器的一种方式，可根据需求修改来操作服务器

  ```Python
  ss = socket()                 #创建服务器套接字
  ss.bind()                     #套接字与地址绑定
  ss.listen()                   #监听连接
  inf_loop:                     #服务器无线循环
      cs = ss.accept()          #接受客户端连接
      comm_loop:                #通信循环
          cs.recv()/cs.send()   #对话(接收/发送)
      cs.close()                #关闭客户端套接字
  ss.close()                    #关闭服务器套接字
  ```

* TCP时间戳服务器

  ```Python
  #!usr/bin/python3
  # -*- coding:UTF-8 -*-

  # 导入socket模块和time.ctime()的所有属性
  from socket import *
  from time import ctime

  # HOST变量是空白，这是对bind()方法的标识，标识它可以使用任何可用的地址
  # 选择一个随机的端口号
  # 缓冲区大小为1KB
  HOST = ''
  PORT = 12345
  BUFSIZE = 1024
  ADDR = (HOST, PORT)

  # 分配了TCP服务套接字
  # 将套接字绑定到服务器地址
  # 开启TCP的监听调用
  # listen()方法的参数是在连接被转接或拒绝之前，传入连接请求的最大数
  tcpSerSock = socket(AF_INET, SOCK_STREAM)
  tcpSerSock.bind(ADDR)
  tcpSerSock.listen(5)

  while True:
      # 服务器循环，等待客户端的连接的连接
      print('waiting for connection...')
      tcpCliSock, addr = tcpSerSock.accept()
      print('...connected from:', addr)

      while True:
          # 当一个连接请求出现时，进入对话循环，接收消息
          data = tcpCliSock.recv(BUFSIZE)
          if not data:
              # 当消息为空时，退出对话循环
              # 关闭客户端连接，等待下一个连接请求
              break
          tcpCliSock.send(bytes('[%s] %s' % (
              ctime(), data.decode('utf-8')), 'utf-8'))

      tcpCliSock.close()

  ```
### TCP客户端

* 下面是TCP客户端的通用伪码

  ```Python
  cs = socket()           #创建客户端套接字
  cs.connect()            #尝试连接服务器
  comm_loop:              #通信循环
      cs.send()/cs.recv   #对话(发送/接收)
  cs.close()              #关闭客户端套接字
  ```

* TCP时间戳客户端

  ```Python
  #!usr/bin/python3
  # -*- coding: UTF-8 -*-

  # 导入socket模块所有属性
  from socket import *

  # 服务器的主机名
  # 服务器的端口号,应与服务器设置的完全相同
  # 缓冲区大小为1KB
  HOST = '127.0.0.1'
  PORT = 12345
  BUFSIZE = 1024
  ADDR = (HOST, PORT)

  # 分配了TCP客户端套接字
  # 主动调用并连接到服务器
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)

  while True:
      # 无限循环，输入消息
      data = bytes(input('> '), 'utf-8')
      if not data:
          # 消息为空则退出循环
          break
      # 发送输入的信息
      # 接收服务器返回的信息，最后打印
      tcpCliSock.send(data)
      data = tcpCliSock.recv(BUFSIZE)
      if not data:
          # 消息为空则退出循环
          break
      print(data.decode('utf-8'))
  # 关闭客户端
  tcpCliSock.close()
  ```

### TCP服务器和客户端运行结果
  
  在运行程序时，必须 **首先运行服务器** 程序，然后再运行客户端程序。如果先运行客户端程序，将会报未连接到服务器的错误。
  按正确的顺序启动程序后，在客户端输入信息，将会接收到加上时间戳处理后的信息，如果直接输入回车，将会关闭客户端，而服务器将会等待下一个连接请求

* 服务器运行结果
 
  ```
  waiting for connection...
  ...connected from: ('127.0.0.1', 53220)
  waiting for connection...
  ```

* 客户端运行结果
 
  ```
  > hello
  [Fri Feb 23 14:22:58 2018] hello
  > hi
  [Fri Feb 23 14:23:02 2018] hi
  > hello world
  [Fri Feb 23 14:23:09 2018] hello world
  >
  Process finished with exit code 0
  ```

## 创建UDP服务器/客户端

### UDP服务器

* 下面是UDP服务器的伪码
  
  ```Python
  ss = socket()                           #创建服务器套接字
  ss.bind()                               #绑定服务器套接字
  inf_loop:                               #服务器无线循环
      cs = ss.recvfrom()/ss.sendto()      #关闭(接收/发送)
  ss.close()                              #关闭服务器套接字
  ```

* UDP时间戳服务器
  
  ```Python
  #!usr/bin/python3
  # -*- coding:UTF-8 -*-

  # 导入socket模块和time.ctime()的全部属性
  from socket import *
  from time import ctime

  # 与TCP相同，由于是无连接，所以没有调用监听传入连接
  HOST = ''
  PORT = 12345
  BUFSIZE = 1024
  ADDR = (HOST, PORT)

  udpSerSock = socket(AF_INET, SOCK_DGRAM)
  udpSerSock.bind(ADDR)

  while True:
      # 进入循环等待消息，一条消息到达时，处理并返回它，然后等待下一条消息
      print('waiting for message...')
      data, addr = udpSerSock.recvfrom(BUFSIZE)
      udpSerSock.sendto(bytes('[%s] %s' % (
          ctime(), data.decode('utf-8')), 'utf-8'), addr)
      print('...received from and returned to:', addr)

  ```

### UDP客户端

* 下面是客户端的伪码

  ```Python
  cs = socket()                         #创建客户端套接字
  comm_loop:                            #通信循环
      cs.sendto()/cs.recvfrom()         #对话(发送/接收)
  cs.close()                            #关闭客户端套接字
  ```

* UDP时间戳客户端

 ```Python
 #!usr/bin/python3
  # -*- coding:UTF-8 -*-

  from socket import *

  HOST = '127.0.0.1'
  PORT = 12345
  BUFSIZE = 1024
  ADDR = (HOST, PORT)

  udpClienSock = socket(AF_INET, SOCK_DGRAM)

  while True:
      data = bytes(input('>'), 'utf-8')
      if not data:
          break
      udpClienSock.sendto(data, ADDR)
      data, ADDR = udpClienSock.recvfrom(BUFSIZE)
      if not data:
          break
      print(data.decode('utf-8'))
  udpClienSock.close()

 ```

### UDP服务器和客户端运行结果

  因为UDP面向无连接的服务，所以程序的启动顺序没有要求。当服务器处理完一个数据报之后在等待下一个继续处理

* 服务器运行结果

  ```
  waiting for message...
  ...received from and returned to: ('127.0.0.1', 51434)
  waiting for message...
  ...received from and returned to: ('127.0.0.1', 51434)
  waiting for message...
  ```

* 客户端运行结果

  ```
  >hello
  [Fri Feb 23 15:23:57 2018] hello
  >hi
  [Fri Feb 23 15:24:03 2018] hi
  >

  Process finished with exit code 0
  ```

[^1]: *AF_UNIX* 是基于文件的套接字，代表 *地址家族(address family):UNIX*，*AF_INET* 是基于网络的套接字，代表 *地址家族：因特网*， *AF_INET6* 用于底6版因特网协议(IPv6)寻址。 *SOCK_STREAM* 表示面向连接的TCP套接字， *SOCK_DGRAM* 代表无连接的UDP套接字。
