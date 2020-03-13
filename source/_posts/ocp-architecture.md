---
title: ocp_architecture
date: 2020-03-11 14:07:59
categories: Oracle
copyright: true
tags:
    - OCP
    - Oracle
description: 学习Oracle OCP课程的体系结构
---

# 体系结构
## oracle服务端体系架构
* oracle服务端体系架构分为instance和database
* instance包含内存结构和后台进程，内存结构包括SGA(系统全局区)和PGA(进程全局区/程序全局区)，instance可由DBA开启或关闭
* database包含数据文件、控制文件、REDO日志、参数文件、密码/追踪等，database一经创建除非物理删除否则永久存在
![ORACLE_ARCHITECTURE](ocp-architecture/ORACLE_ARCHITECTURE.png)

## 内存结构
### SGA
#### 共享池(shared pool)
* 共享sql区：最近执行的SQL或PL/SQL程序
* 库高速缓存：由语句的hash值和执行计划树的方式保存SQL(LRU)
* 数据字典高速缓存：记录相关的用户、表等元数据信息
* 结果集缓存：缓存部分结果集
![oracle_sp](ocp-architecture/oracle_sp.png)

####  数据块(database buffer cache)
* 保存从数据文件中读取的数据(镜像)部分
* 读写最严重的地方
* 根据不同的数据文件大小创建不同的块
* 通过dbwr进程覆盖写回数据文件
![oracle_sp](ocp-architecture/oracle_dbbuffer.png)

#### Redo Log Buffer
* 保存有关数据所有改动信息，包含所有重做语句条目
* 通过lgwr后台进程写回REDO日志文件
* PGA告诉log buffer数据已经改变，重做条目是Oracle server process从用户数据(User session data)中拷贝过去的
  
![oracle_sp](ocp-architecture/oracle_logbuffer.png)

#### 大池(Large Pool)
* 数据库备份和恢复操作使用大池
* 共享服务器模式下UGA在大池保存

#### Java Pool
* 用于在JVM中存储所有会话运行的java代码数据

#### In-Memory Column Store
* 12.1.0.2的新特性
* 大表在任何列上的查询速度更快
* 使用扫描、连接和聚合
* 不使用索引（因为在内存中）

### PGA
* 对每一个连接分配一块独立的PGA空间
* 记录相关会话，记录相关语句，与SGA进行沟通，与Server Process连接

### 进程结构(Process Architecture)
* 用户进程结构、数据库进程结构、守护进程
#### 数据库进程结构
* 包括Server Process、后台进程
* DBWn(数据库写进程):异步、与检查点有关
* LGWR(日志写进程):写redo log buffer，只有一个进程
* CKPT(检查点进程):记录相关相关检查点信息，根据SN号的改变做记录(SN号在数据库是一直向推进的)，往控制文件中和每个数据文件的文件头写，具体写操作的控制单元
* SMON(系统监视器):清理临时段、启动时执行恢复
* PMON(进程监控):清理数据库数据缓存、监视空闲Session和超时Session
* RECO(恢复进程与):分布式数据库其它使用、自动连接其它分布式环境中的数据库，自动解决所有不确定的事务、删除与不确定事务的相关的所有行
* LREG(监听注册进程):监听注册进程，注册有关实例到监听程序上(动态监听)
* ARCn(归档进程):发生日志切换后将redo log文件存储到指定的存储设备上；收集事务重做的数据并传递数据到备用目标

## 数据库存储体系
* 数据块是Oracle中最小的逻辑单元，而系统块是操作系统中最小的存储物理操作单元
![oracle_sp](ocp-architecture/oracle_storage.png)

* 多个系统块构成一个数据块，多个数据块构成一个分区，多个分区构成一个段，多个分区组成一个数据文件，多个数据文件构成一个段，多个段构成一个表空间，多个表空间构成一个数据库；表空间和段有三种状态(持久、临时、undo)
![oracle_sp](ocp-architecture/oracle_storage2.png)

* SYSTEM和SYSAUX表空间在创建数据库时必须创建并且必须保证一直在线
* ASM(Oracle自动存储管理):经常在RAID上使用；移植性好；将裸设备按文件系统的方式进程管理，提升了可维护性；可以跨磁盘传输数据进行负载均衡
![oracle_sp](ocp-architecture/oracle_asm.png)






