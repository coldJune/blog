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

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 导入必需的模块
import os
from random import randrange as rand

# 创建了全局变量
# 用于显示列的大小，以及支持的数据库种类
COLSIZ = 10
FIELDS = ('login', 'userid', 'projid')
RDBMSs = {
    's': 'sqlite',
    'm': 'mysql',
}
DBNAME = 'test'
DBUSER = 'root'
# 数据库异常变量，根据用户选择运行的数据库系统的不同来制定数据库异常模块
DB_EXC = None
NAMELEN = 16

# 格式化字符串以显示标题
# 全大写格式化函数，接收每个列名并使用str.upper()方法把它转换为头部的全大写形式
# 两个函数都将其输出左对齐，并限制为10个字符的宽度ljust(COLSIZ)
tformat = lambda s: str(s).title().ljust(COLSIZ)
cformat = lambda s: s.upper().ljust(COLSIZ)


def setup():
    return RDBMSs[input('''
        Choose a database system:
        (M)ySQL
        (S)QLite
        Enter choice:
    ''').strip().lower()[0]]


def connect(db):
    # 数据库一致性访问的核心
    # 在每部分的开始出尝试加载对应的数据库模块，如果没有找到合适的模块
    # 就返回None，表示无法支持数据库系统
    global DB_EXC
    dbDir = '%s_%s' % (db, DBNAME)

    if db == 'sqlite':
        try:
            # 尝试加载sqlite3模块
            import sqlite3
        except ImportError:
            return None
        DB_EXC = sqlite3
        # 当对SQLite调用connect()时，会使用已存在的目录
        # 如果没有，则创建一个新目录
        if not os.path.isdir(dbDir):
            os.mkdir(dbDir)
        cxn = sqlite3.connect(os.path.join(dbDir, DBNAME))
    elif db == 'mysql':
        try:
            # 由于MySQLdb不支持python3.6，所以导入pymysql
            import pymysql
            import pymysql.err as DB_EXC
            try:
                cxn = pymysql.connect(host="localhost",
                                      user="root",
                                      password="root",
                                      port=3306,
                                      db=DBNAME)
            except DB_EXC.InternalError:
                try:
                    cxn = pymysql.connect(host="localhost",
                                          user="root",
                                          password="root",
                                          port=3306)
                    cxn.query('CREATE DATABASE %s' % DBNAME)
                    cxn.commit()
                    cxn.close()
                    cxn = pymysql.connect(host="localhost",
                                          user="root",
                                          password="root",
                                          port=3306,
                                          db=DBNAME)
                except DB_EXC.InternalError:
                    return None
        except ImportError:
            return None
    else:
        return None
    return cxn


def create(cur):
    # 创建一个新表users
    try:
        cur.execute('''
            CREATE  TABLE  users(
                login VARCHAR(%d),
                userid INTEGER,
                projid INTEGER
            )
        ''' % NAMELEN)
    except DB_EXC.InternalError as e:
        # 如果发生错误，几乎总是这个表已经存在了
        # 删除该表，重新创建
        drop(cur)
        create(cur)

# 删除数据库表的函数
drop = lambda cur: cur.execute('DROP TABLE users')

# 由用户名和用户ID组成的常量
NAMES = (
    ('bob', 1234), ('angela', 4567), ('dave', 4523)
)


def randName():
    # 生成器
    pick = set(NAMES)
    while pick:
        yield pick.pop()


def insert(cur, db):
    # 插入函数
    # SQLite风格是qmark参数风格，而MySQL使用的是format参数风格
    # 对于每个用户名-用户ID对，都会被分配到一个项目卒中。
    # 项目ID从四个不同的组中随机选出的
    if db == 'sqlite':
        cur.executemany("INSERT INTO users VALUES(?,?,?)",
                        [(who, uid, rand(1, 5)) for who, uid in randName()])
    elif db == 'mysql':
        cur.executemany("INSERT INTO users VALUES(%s, %s, %s)",
                        [(who, uid, rand(1, 5)) for who, uid in randName()])

# 返回最后一次操作后影响的行数，如果游标对象不支持该属性，则返回-1
getRC = lambda cur: cur.rowcount if hasattr(cur, 'rowcount') else -1


# update()和delete()函数会随机选择项目组中的成员
# 更新操作会将其从当前组移动到另一个随机选择的组中
# 删除操作会将该组的成员全部删除
def update(cur):
    fr = rand(1, 5)
    to = rand(1, 5)
    cur.execute('UPDATE users SET projid=%d WHERE projid=%d' % (to, fr))
    return fr, to, getRC(cur)


def delete(cur):
    rm = rand(1, 5)
    cur.execute('DELETE FROM users WHERE projid=%d' % rm)
    return rm, getRC(cur)


def dbDump(cur):
    # 来去所有行，将其按照打印格式进行格式化，然后显示
    cur.execute('SELECT * FROM users')
    # 格式化标题
    print('%s' % ''.join(map(cformat, FIELDS)))
    for data in cur.fetchall():
        # 将数据(login,userid,projid)通过map()传递给tformat()，
        # 是数据转化为字符串，将其格式化为标题风格
        # 字符串按照COLSIZ的列宽度进行左对齐
        print(''.join(map(tformat, data)))


def main():
    # 主函数
    db = setup()
    print('*** Connect to %r database' % db)
    cxn = connect(db)
    if not cxn:
        print('ERROR: %r not supported or unreadable, exit' % db)
        return
    cur = cxn.cursor()
    print('***Creating users table')
    create(cur=cur)

    print('***Inserting names into table')
    insert(cur, db)
    dbDump(cur)

    print('\n***Randomly moving folks')
    fr, to, num = update(cur)
    print('(%d users moved) from (%d) to (%d)' % (num, fr, to))
    dbDump(cur)

    print('***Randomly choosing group')
    rm, num = delete(cur)
    print('\t(group #%d; %d users removed)' % (rm, num))
    dbDump(cur)

    print('\n***Droping users table')
    drop(cur)
    print('\n*** Close cxns')
    cur.close()
    cxn.commit()
    cxn.close()

if __name__ == '__main__':
    main()

```

* MySQL数据库访问结果

```
Choose a database system:
        (M)ySQL
        (S)QLite
Enter choice:
M
*** Connect to 'mysql' database
***Creating users table
***Inserting names into table
LOGIN     USERID    PROJID    
Dave      4523      2         
Bob       1234      3         
Angela    4567      3         

***Randomly moving folks
(2 users moved) from (3) to (1)
LOGIN     USERID    PROJID    
Dave      4523      2         
Bob       1234      1         
Angela    4567      1         
***Randomly choosing group
	(group #1; 2 users removed)
LOGIN     USERID    PROJID    
Dave      4523      2         

***Droping users table

*** Close cxns
```

* SQLite数据库访问结果

```
Choose a database system:
(M)ySQL
(S)QLite
Enter choice:
S
*** Connect to 'sqlite' database
***Creating users table
***Inserting names into table
LOGIN     USERID    PROJID    
Dave      4523      1         
Bob       1234      2         
Angela    4567      3         

***Randomly moving folks
(1 users moved) from (1) to (1)
LOGIN     USERID    PROJID    
Dave      4523      1         
Bob       1234      2         
Angela    4567      3         
***Randomly choosing group
(group #3; 1 users removed)
LOGIN     USERID    PROJID    
Dave      4523      1         
Bob       1234      2         

***Droping users table

*** Close cxns
```
