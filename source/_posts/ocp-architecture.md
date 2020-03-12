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
* 通过lgwr后台进程协会REDO日志文件
* PGA告诉log buffer数据已经改变
  
![oracle_sp](ocp-architecture/oracle_logbuffer.png)

#### 大池(Large Pool)
* 数据库备份和恢复操作使用大池
* 共享服务器模式下UGA在大池保存

#### Java Pool
* 用于在JVM中存储所有会话运行的java代码数据





