---
title: Python数据库编程(二)
date: 2018-02-28 18:15:57
categories: Python
copyright: true
tags:
    - Python
    - 数据库编程
description:
---
上一篇中主要对直接操作数据库做了一个比较详细的总结，这里将会对使用ORM框架进行简要的描述。
<!--More-->
## ORM
ORM系统的作者将纯SQL语句进行了抽象化处理，将其实现为Python中的对象，这样只操作这些对象就能完成与生成SQL语句相同的任务。

### python与ORM

[SQLAlchemy](http://www.sqlalchemy.org/)和[SQLObject](http://sqlobject.org/)是两种不同的Python ORM。这两种ORM并不在Python标准库中，所以需要安装。
* 安装SQLAlchemy
`pip3 install sqlalchemy`

* 安装SQLObject
`pip3 install -U SQLObject`

在这里将会通过两种ORM移植上一篇的[数据库适配器示例应用](http://coldjune.com/2018/02/28/Python数据库编程-一#数据库适配器示例应用)

#### SQLAlchemy
> SQLAlchemy相比于SQLObject的接口更加接近于SQL语句。SQLAlchemy中对象的抽象化十分完成，还可以以更好的灵活性提交原生的SQL语句

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 首先导入标准库中的模块(os.path、random)
# 然后是第三方或外部模块(sqlalchemy)
# 最后是应用的本地模块(ushuffleDB)
from os.path import dirname
from random import randrange as rand
from sqlalchemy import Column, Integer, \
    String, create_engine, exc, orm
from sqlalchemy.ext.declarative \
    import declarative_base
from ushuffleDB import DBNAME, NAMELEN, \
    randName, FIELDS, tformat, cformat, setup

# 数据库类型+数据库驱动名称://用户名:密码@地址:端口号/数据库名称
DSNs = {
    'mysql': 'mysql+pymysql://root:root@localhost:3306/%s' % DBNAME,
    'sqlite': 'sqlite:///:memory:',
}

# 使用SQLAlchemy的声明层
# 使用导入的sqlalchemy.ext.declarative.declarative_base
# 创建一个Base类
Base = declarative_base()


class Users(Base):
    # 数据子类
    # __tablename__定义了映射的数据库表名
    __tablename__ = 'users'
    # 列的属性，可以查阅文档来获取所有支持的数据类型
    login = Column(String(NAMELEN))
    userid = Column(Integer, primary_key=True)
    projid = Column(Integer)

    def __str__(self):
        # 用于返回易于阅读的数据行的字符串格式
        return ''.join(map(tformat, (self.login, self.userid, self.projid)))


class SQLAlchemyTest(object):
    def __init__(self, dsn):
        # 类的初始化执行了所有可能的操作以便得到一个可用的数据库，然后保存其连接
        # 通过设置echo参数查看ORM生成的SQL语句
        # create_engine('sqlite:///:memory:', echo=True)
        try:
            eng = create_engine(dsn)
        except ImportError:
            raise RuntimeError()

        try:
            eng.connect()
        except exc.OperationalError:
            # 此处连接失败是因为数据库不存在造成的
            # 使用dirname()来截取掉数据库名，并保留DSN中的剩余部分
            # 使数据库的连接可以正常运行
            # 这是一个典型的操作任务而不是面向应用的任务，所以使用原生SQL
            eng = create_engine(dirname(dsn))
            eng.execute('CREATE DATABASE %s' % DBNAME).close()
            eng = create_engine(dsn)
        # 创建一个会话对象，用于管理单独的事务对象
        # 当涉及一个或多个数据库操作时，可以保证所有要写入的数据都必须提交
        # 然后将这个会话对象保存，并将用户的表和引擎作为实例属性一同保存下来
        # 引擎和表的元数据进行了额外的绑定，使这张表的所有操作都会绑定到这个指定的引擎中
        Session = orm.sessionmaker(bind=eng)
        self.ses = Session()
        self.users = Users.__table__
        self.eng = self.users.metadata.bind = eng

    def insert(self):
        # session.add_all()使用迭代的方式产生一系列的插入操作
        self.ses.add_all(
            Users(login=who, userid=userid, projid=rand(1, 5))
            for who, userid in randName()
        )
        # 决定是提交还是回滚
        self.ses.commit()

    def update(self):
        fr = rand(1, 5)
        to = rand(1, 5)
        i = -1
        # 会话查询的功能，使用query.filter_by()方法进行查找
        users = self.ses.query(Users).filter_by(projid=fr).all()
        for i, user in enumerate(users):
            user.projid = to
        self.ses.commit()
        return fr, to, i+1

    def delete(self):
        rm = rand(1, 5)
        i = -1
        users = self.ses.query(Users).filter_by(projid=rm).all()
        for i, user in enumerate(users):
            self.ses.delete(user)
        self.ses.commit()
        return rm, i+1

    def dbDump(self):
        # 在屏幕上显示正确的输出
        print('\n%s' % ''.join(map(cformat, FIELDS)))
        users = self.ses.query(Users).all()
        for user in users:
            print(user)
        self.ses.commit()

    def __getattr__(self, attr):
        # __getattr__()可以避开创建drop()和create()方法
        # __getattr__()只有在属性查找失败时才会被调用
        # 当调用orm.drop()并发现没有这个方法时，就会调用getattr(orm, 'drop')
        # 此时调用__getattr__()，并且将属性名委托给self.users。结束期会发现
        # slef.users存在一个drop属性，然后传递这个方法调用到self.users.drop()中
        return getattr(self.users, attr)

    def finish(self):
        # 关闭连接
        self.ses.connection().close()


def main():
    # 入口函数
    print('\n***Connnect to %r database' % DBNAME)
    db = setup()
    if db not in DSNs:
        print('ERROR: %r not supported, exit' % db)
        return

    try:
        orm = SQLAlchemyTest(DSNs[db])
    except RuntimeError:
        print('ERROR: %r not supported, exit' % db)
        return

    print('\n*** Create users table(drop old one if appl.')
    orm.drop(checkfirst=True)
    orm.create()

    print('\n***Insert namse into table')
    orm.insert()
    orm.dbDump()

    print('\n***Move users to a random group')
    fr, to, num = orm.update()
    print('\t(%d users moved) from (%d) to (%d))' % (num, fr, to))
    orm.dbDump()

    print('\n***Randomly delete group')
    rm, num = orm.delete()
    print('\t(group #%d; %d users removed)' % (rm, num))
    orm.dbDump()

    print('\n***Drop users table')
    orm.drop()
    print('***Close cxns')
    orm.finish()

if __name__ == '__main__':
    main()
```

* mysql输出结果

```
***Connnect to 'test' database

Choose a database system:
    (M)ySQL
    (S)QLite
Enter choice:
M


*** Create users table(drop old one if appl.

***Insert namse into table

LOGIN     USERID    PROJID    
Bob       1234      1         
Dave      4523      1         
Angela    4567      3         

***Move users to a random group
	(2 users moved) from (1) to (4))

LOGIN     USERID    PROJID    
Bob       1234      4         
Dave      4523      4         
Angela    4567      3         

***Randomly delete group
	(group #2; 0 users removed)

LOGIN     USERID    PROJID    
Bob       1234      4         
Dave      4523      4         
Angela    4567      3         

***Drop users table
***Close cxns

```

* SQLite输出结果

```

***Connnect to 'test' database

Choose a database system:
        (M)ySQL
        (S)QLite
Enter choice:
S

*** Create users table(drop old one if appl.

***Insert namse into table

LOGIN     USERID    PROJID    
Bob       1234      2         
Dave      4523      1         
Angela    4567      2         

***Move users to a random group
	(2 users moved) from (2) to (2))

LOGIN     USERID    PROJID    
Bob       1234      2         
Dave      4523      1         
Angela    4567      2         

***Randomly delete group
	(group #1; 1 users removed)

LOGIN     USERID    PROJID    
Bob       1234      2         
Angela    4567      2         

***Drop users table
***Close cxns
```
#### SQLObject

SQLObject需要mysqldb支持，但是由于mysqldb不再支持python3，所以根据提示安装替代方案[Mysqlclient](https://www.lfd.uci.edu/~gohlke/pythonlibs/#Mysqlclient)，选择对应的版本进行下载后执行相应的命令：
`pip3 install mysqlclient-1.3.12-cp36-cp36m-win_amd64.whl`

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 使用SQLObject代替SQLAlchemy
# 其余和使用SQLAlchemy的相同
from os.path import dirname
from random import randrange as rand
from sqlobject import *
from ushuffleDB import  DBNAME, NAMELEN, \
    randName, FIELDS, tformat, cformat, setup

DSNs = {
    'mysql': 'mysql://root:root@127.0.0.1:3306/%s' % DBNAME,
    'sqlite': 'sqlite:///:memory:',
}


class Users(SQLObject):
    # 扩展了SQLObject.SQLObject类
    # 定义列
    login = StringCol(length=NAMELEN)
    userid = IntCol()
    projid = IntCol()

    def __str__(self):
        # 提供用于显示输出的方法
        return ''.join(map(tformat, (
            self.login, self.userid, self.projid)))


class SQLObjectTest(object):
    def __init__(self, dsn):
        # 确保得到一个可用的数据库，然后返回连接
        try:
            cxn = connectionForURI(dsn)
        except ImportError:
            raise RuntimeError()

        try:
            # 尝试对已存在的表建立连接
            # 规避RMBMS适配器不可用，服务器不在线及数据库不存在等异常
            cxn.releaseConnection(cxn.getConnection())
        except dberrors.OperationalError:
            # 出现异常则创建表
            cxn = connectionForURI(dirname(dsn))
            cxn.query('CREATE DATABASE %s' % DBNAME)
            cxn = connectionForURI(dsn)
        # 成功后在self.cxn中保存连接对象
        self.cxn = sqlhub.processConnection = cxn

    def insert(self):
        # 插入
        for who, userid in randName():
            Users(login=who, userid=userid, projid=rand(1, 5))

    def update(self):
        # 更新
        fr = rand(1, 5)
        to = rand(1, 5)
        i = -1
        users = Users.selectBy(projid=fr)
        for i, user in enumerate(users):
            user.projid = to
        return fr, to, i+1

    def delete(self):
        # 删除
        rm = rand(1, 5)
        users = Users.selectBy(projid=rm)
        i = -1
        for i, user in enumerate(users):
            user.destroySelf()
        return rm, i+1

    def dbDump(self):
        print('\n%s' % ''.join(map(cformat, FIELDS)))
        for user in Users.select():
            print(user)

    def finish(self):
        # 关闭连接
        self.cxn.close()


def main():
    print('***Connect to %r database' % DBNAME)
    db = setup()
    if db not in DSNs:
        print('\nError: %r not support' % db)
        return

    try:
        orm = SQLObjectTest(DSNs[db])
    except RuntimeError:
        print('\nError: %r not support' % db)
        return

    print('\n***Create users table(drop old one if appl.)')
    Users.dropTable(True)
    Users.createTable()

    print('\n*** Insert names into table')
    orm.insert()
    orm.dbDump()

    print('\n*** Move users to a random group')
    fr, to, num = orm.update()
    print('\t(%d users moved) from (%d) to (%d)' % (num, fr, to))
    orm.dbDump()

    print('\n*** Randomly delete group')
    rm, num = orm.delete()
    print('\t(group #%d;%d users removed)' % (rm, num))
    orm.dbDump()

    print('\n*** Drop users table')
    # 使用dropTable()方法
    Users.dropTable()
    print('\n***Close cxns')
    orm.finish()

if __name__ == '__main__':
    main()
```

* MySQL输出结果

```
Choose a database system:
(M)ySQL
(S)QLite
Enter choice:
M

***Create users table(drop old one if appl.)

*** Insert names into table

LOGIN     USERID    PROJID    
Bob       1234      4         
Dave      4523      3         
Angela    4567      1         

*** Move users to a random group
(0 users moved) from (2) to (4)

LOGIN     USERID    PROJID    
Bob       1234      4         
Dave      4523      3         
Angela    4567      1         

*** Randomly delete group
(group #3;1 users removed)

LOGIN     USERID    PROJID    
Bob       1234      4         
Angela    4567      1         

*** Drop users table

***Close cxns
```

* SQLite输出结果

```
Choose a database system:
(M)ySQL
(S)QLite
Enter choice:
S

***Create users table(drop old one if appl.)

*** Insert names into table

LOGIN     USERID    PROJID    
Bob       1234      2         
Angela    4567      4         
Dave      4523      3         

*** Move users to a random group
(1 users moved) from (3) to (1)

LOGIN     USERID    PROJID    
Bob       1234      2         
Angela    4567      4         
Dave      4523      1         

*** Randomly delete group
(group #2;1 users removed)

LOGIN     USERID    PROJID    
Angela    4567      4         
Dave      4523      1         

*** Drop users table

***Close cxns
```

## 非关系型数据库

Web和社交服务会产生大量的数据，并且数据的产生速率可能要比关系型数据库能够处理得更快。非关系数据库有对象数据库、键-值对存储、文档存储（或数据存储）、图形数据库、表格数据库、列/可扩展记录/宽列数据库、多值数据库等很多种类。

### MongoDB
[MongoDB](https://www.mongodb.com/)是非常流行的文档存储非关系数据库。
>文档存储(MongoDB、CouchDB/Amazon SimpleDB)与其他非关系数据库的区别在于它介于简单的键-值对存储(Redis、Voldemort)与列存储(HBase、Google Bigtable)之间。比基于列的存储更简单、约束更少。比普通的键-值对存储更加灵活。一般情况下其数据会另存为JSON对象、并且允许诸如字符串、数值、列表甚至嵌套等数据类型

MongoDB(以及NoSQL)要讨论的事文档、集合而不是关系数据库中的行和列。MongoDB将数据存储于特殊的JSON串(文档)中，由于它是一个二进制编码的序列化，通常也称其为BSON格式。它和JSON或者Python字典都很相似。

### PyMongo:MongoDB和Python
PyMongo是Python MongoDB驱动程序中最正式的一个。使用之前需要[安装MongoDB数据库](https://www.mongodb.com/download-center?jmp=nav#atlas)和PyMongo：
` pip3 install pymongo`
在windows下需要运行mongo.exe启动MongoDB，进入cmd到MongoDB的bin目录下，执行如下命令
` .mongod --dbpath E:\MongoDB\data`

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 主要导入的是MongoClient对象和及其包异常errors
from random import randrange as rand
from pymongo import MongoClient, errors
from ushuffleDB import DBNAME, randName, FIELDS, tformat, cformat

# 设置了集合(“表”)名
COLLECTION = 'users'


class MongoTest(object):
    def __init__(self):
        # 创建一个连接，如果服务器不可达，则抛出异常
        try:
            cxn = MongoClient()
        except errors.AutoReconnect:
            raise RuntimeError
        # 创建并复用数据库及“users”集合
        # 关系数据库中的表会对列的格式进行定义，
        # 然后使遵循这个列定义的每条记录成为一行
        # 非关系数据库中集合没有任何模式的需求，
        # 每条记录都有其特定的文档
        # 每条记录都定义了自己的模式，所以保存的任何记录都会写入集合中
        self.db = cxn[DBNAME]
        self.users = self.db[COLLECTION]

    def insert(self):
        # 向MongoDB的集合中添加值
        # 使用dict()工厂函数为每条记录创建一个文档
        # 然后将所有文档通过生成器表达式的方式传递给集合的insert()方法
        self.users.insert(
            dict(login=who, userid=uid, projid=rand(1, 5)
                 )for who, uid in randName()
        )

    def update(self):
        # 集合的update()方法可以给开发者相比于典型的数据库系统更多的选项
        fr = rand(1, 5)
        to = rand(1, 5)
        i = -1
        # 在更新前，首先查询系统中的项目ID(projid)与要更新的项目组相匹配的所有用户
        # 使用find()方法，并将查询条件传进去(类似SQL的SELECT语句)
        for i, user in enumerate(self.users.find({'projid': fr})):
            # 使用$set指令可以显式地修改已存在的值
            # 每条MongoDB指令都代表一个修改操作，使得修改操作更加高效、有用和便捷
            # 除了$set还有一些操作可以用于递增字段值、删除字段(键-值对)、对数组添加/删除值
            # update()方法可以用来修改多个文档(将multi标志设为True)
            self.users.update(user, {
                '$set': {'projid': to}
            })
        return fr, to, i+1

    def delete(self):
        # 当得到所有匹配查询的用户后，一次性对其执行remove()操作进行删除
        # 然后返回结果
        rm = rand(1, 5)
        i = -1
        for i, user in enumerate(self.users.find({'projid': rm})):
            self.users.remove(user)
        return rm, i+1

    def dbDump(self):
        # 没有天剑会返回集合中所有用户并对数据进行字符串格式化向用户显示
        print('%s' % ''.join(map(cformat, FIELDS)))
        for user in self.users.find():
            print(''.join(map(tformat, (
                user[k] for k in FIELDS))))


def main():
    print('***Connect to %r database' % DBNAME)
    try:
        mongo = MongoTest()
    except RuntimeError:
        print('\nERROR: MongoDB server unreadable, exit')
        return

    print('\n***Insert names into table')
    mongo.insert()
    mongo.dbDump()

    print('\n***Move users to a random group')
    fr, to, num = mongo.update()
    print('\t(%d users moved) from (%d) to (%d)' % (num, fr, to))
    mongo.dbDump()

    print('\n*** Randomly delete group')
    rm, num = mongo.delete()
    print('\tgroup #%d; %d users removed' % (rm, num))
    mongo.dbDump()

    print('\n***Drop users table')
    mongo.db.drop_collection(COLLECTION)

if __name__ == '__main__':
    main()
```

* 执行结果

```
***Connect to 'test' database

***Insert names into table
LOGIN     USERID    PROJID    
Dave      4523      4         
Bob       1234      4         
Angela    4567      2         

***Move users to a random group
	(0 users moved) from (1) to (2)
LOGIN     USERID    PROJID    
Dave      4523      4         
Bob       1234      4         
Angela    4567      2         

*** Randomly delete group
	group #2; 1 users removed
LOGIN     USERID    PROJID    
Dave      4523      4         
Bob       1234      4         

***Drop users table
```
