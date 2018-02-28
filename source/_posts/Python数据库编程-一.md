---
title: Python数据库编程(一)
date: 2018-02-28 09:47:18
categories: Python
copyright: true
tags:
    - Python
    - 数据库编程
description:
---
Python和大多数语言一样，访问数据库包括直接通过数据库接口访问和使用ORM访问两种方式。其中ORM访问的方式不需要显式地给出SQL命令。在Python中数据库是通过**适配器**的方式进行访问的。适配器是一个Python模块，使用它可以与关系型数据库的客户端库接口相连。
<!--More-->

## Python的DB-API

> DB-API是阐明一系列所需对象和数据库访问机制的标准，它可以为不同的数据库适配器和底层数据库系统提供一致性访问

### 模块属性

#### DB-API模块属性

|     属性     |            描述            |
|:------------:|:--------------------------:|
|   apilevel   | 需要适配器兼容的DB-API版本 |
| threadsafety |    本模块的线程安全级别    |
|  paramstyle  |  本模块的SQL语句参数风格   |
|  connect()   |       Connect()函数        |
| (多种异常)             |                            |

#### 数据属性

* apilevel
> 该字符串指明了模块需要兼容的DB-API最高版本，默认值为1.0

* threadsafety
> 0: 不支持线程安全。线程间不能共享模块
  1: 最小化线程安全支持：线程间可以共享模块，但是不能共享连接
  2: 适度的线程安全支持：线程间可以共享模块和连接，但是不能共享游标
  3: 完整的线程安全支持：线程间可以共享模块、连接和游标

**如果有资源需要共享，那么就需要诸如自旋锁、信号量等同步原语达到原子锁定的目的**

#### 参数风格

* paramstyle

| 参数风格 |            描述            |        示例         |
|:--------:|:--------------------------:|:-------------------:|
| numeric  |        数值位置风格        |    WHERE name=:1    |
|  named   |          命名风格          |  WHERE name=:name   |
| pyformat | Python字典printf()格式转换 | WHERE name=%(name)s |
|  qmark   |          问号风格          |    WHERE name=?     |
|  format  |  ANSIC的printf()格式转换   |    WHERE name=%s    |

#### 函数属性
> connect()函数通过Connection对象访问数据库。兼容模块必须实现connect()函数。该函数创建并放回一个Connection对象

connect()函数使用例子：
`connect(dsn='myhost:MYDB', user='root', password='root')`

* connect()函数属性

|   参数   |   描述   |
|:--------:|:--------:|
|   user   |  用户名  |
| password |    面    |
|   host   |  主机名  |
| database | 数据库名 |
|   dsn    | 数据源名         |

使用ODBC或JDBC的API需要使用DSN；直接使用数据库，更倾向于使用独立的登录参数。

#### 异常

|       异常        |             描述             |
|:-----------------:|:----------------------------:|
|      Warning      |         警告异常基类         |
|       Error       |         错误异常基类         |
|  InterfaceError   |   数据库接口(非数据库)错误   |
|   DatabaseError   |          数据库错误          |
|     DataError     |      处理数据时出现错误      |
|  OperationError   | 数据库操作执行期间出现的错误 |
|  IntegrityError   |     数据库关系完整性错误     |
|   InternalError   |        数据库内部错误        |
| ProgrammingError  |       SQL命令执行失败        |
| NotSupportedError |       出现不支持的操作       |

### Connection对象
> 只有通过数据连接才能把命令传递到服务器，并得到返回的结果。当一个连接(或一个连接池)建立后，可以创建一个游标，向数据库发送请求，然后从数据库接收回应

#### Connection对象方法

|               方法名                |                     描述                     |
|:-----------------------------------:|:--------------------------------------------:|
|               close()               |                关闭数据库连接                |
|              commit()               |                 提交当前事务                 |
|             rollback()              |                 取消当前事务                 |
|              cursor()               | 使用该连接创建(并返回)一个游标或类游标的对象 |
| errorhandler(cxn,cur,errcls,errval) |         作为给定连接的游标的处理程序         |

* 当使用`close()`时，这个连接将不能再使用，否则会进入到异常处理中
* 如果数据库不支持事务处理或启用了自动提交功能，`commit()`方法都无法使用
* `rollback()`只能在支持事务处理的数据库中使用。发生异常时，`rollback()`会将数据库的状态恢复到事务处理开始时。
* 如果RDBMS(关系数据库管理系统)不支持游标，`cursor()`会返回一个尽可能模仿真实游标的对象

#### Cursor对象
> 游标可以让用户提交数据库命令，并获得查询的结果行。

|              对象属性              |                                                            描述                                                            |
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
|             arraysize              |                                     使用fetchmany()方法时，一次取出的结果行数，默认为1                                     |
|             connection             |                                                   创建此游标的连接(可选)                                                   |
|            description             | 返回游标活动状态(7项元组):(name,type_code,display_size,internal_size,precision,scale,null-ok)，只有name和type_code是必需的 |
|             lastrowid              |                                     上次修改行的行ID(可选，如果不支持行ID，则返回None)                                     |
|              rowcount              |                                             上次execute*()方法处理或影响的行数                                             |
|       callproc(func[,args])        |                                                        调用存储过程                                                        |
|              close()               |                                                          关闭游标                                                          |
|         execute(op[,args])         |                                                    执行数据库查询或命令                                                    |
|        executemany(op,args)        |                           类似execute()和map()的结合，为给定的所有参数准备并执行数据库查询或命令                           |
|             fetchone()             |                                                    获取查询结果的下一行                                                    |
| fetchmany([size=cursor,arraysize]) |                                                  获取查询结果的下面size行                                                  |
|             fetchall()             |                                                 获取查询结果的所有(剩余)行                                                 |
|             __iter__()             |                                           为游标创建迭代器对象(可选，参考nexi())                                           |
|              messages              |                                     游标执行后从数据库中获得的消息列表(元组集合，可选)                                     |
|               next()               |                           被迭代器用于获取查询结果的下一行(可选，类似fetchone(),参考__iter__())                            |
|             nextset()              |                                               移动到下一个结果集合(如果支持)                                               |
|             rownumber              |                                     当前结果集中游标的索引(以行为单位，从0开始，可选)                                      |
|        setinputsizes(sizes)        |                                      设置允许的最大输入大小(必须有，但是实现是可选的)                                      |
|     setoutputsize(size[,col])      |                                   设置大列获取的最大缓冲区大小(必须有，但是实现是可选的)                                   |
**游标对象最重要的属性是execute*()和fetch*()方法，所有针对数据库的服务请求都通过它们执行。当不需要是关闭游标**

#### 类型对象和构造函数
> 创建构造函数，从而构建可以简单地转换成适当数据库对象的特殊对象

|            类型对象            |                                 描述                                  |
|:------------------------------:|:---------------------------------------------------------------------:|
|         Date(yr,mo,dy)         |                              日期值对象                               |
|        Time(hr,min,sec)        |                              时间值对象                               |
| Timestamp(yr,mo,dy,hr,min,sec) |                             时间戳值对象                              |
|      DateFromTicks(ticks)      |  日期对象，给出从新纪元时间（1970 年1 月1 日00:00:00 UTC）以来的秒数  |
|      TimeFromTicks(ticks)      |  时间对象，给出从新纪元时间（1970 年1 月1 日00:00:00 UTC）以来的秒数  |
|   TimestampFromTicks(ticks)    | 时间戳对象，给出从新纪元时间（1970 年1 月1 日00:00:00 UTC）以来的秒数 |
|         Binary(string)         |                       对应二进制(长)字符串对象                        |
|             STRING             |                  表示基于字符串列的对象，比如VARCHAR                  |
|             BINARY             |                 表示(长)二进制列的对象，比如RAW、BLOB                 |
|             NUMBER             |                           表示数值列的对象                            |
|            DATETIME            |                         表示日期/时间列的对象                         |
|             ROWID              |                          表示“行ID”列的对象                           |
**SQL的NULL值对应于Python的NULL对象None**

#### 数据库适配器示例应用
