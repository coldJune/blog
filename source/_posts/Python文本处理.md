---
title: Python文本处理
date: 2018-03-13 14:48:15
categories: Python
copyright: true
tags:
    - Python
    - 文本处理
description: 无论什么类型的应用，都需要处理成可读的数据，而数据一般是文本。Python标准库有3个文本处理模块和包，它们分别可以处理csv、json、xml
---
## 逗号分割值(CSV)
### CSV简介
**逗号分割值(Comma-Spearated Value, CSV)** 通常用于在电子表格软件和纯文本之间交互数据。CSV文件内容仅仅是一些用逗号分隔的原始字符串值。CSV格式的文件需要专门用于解析和生成CSV的库，不能使用`str.splt(',')`来解析，因为会处理单个字段中含有逗号的情形。
```Python
#!/usr/bin/python3
# -*- coding:utf-8 -*-
import csv
#  创建需要导入的数据源
DATA = (
    (1, 'Web Clients and Servers', 'base64,urllib'),
    (2, 'Web program：CGI & WSGI', 'cgi, time, wsgiref'),
    (3, 'Web Services', 'urllib,twython')
)

print('*** WRITTING CSV DATA')
# 打开一个csv文件，使用utf-8编码，同时为了防止写入时附加多的空白行设置newline为空
with open('bookdata.csv', 'w', encoding='utf-8', newline='') as w:
    # csv.writer笑一个打开的文件(或类文件)对象，返回一个writer对象
    # 可以用来在打开的文件中逐行写入逗号分隔的数据。
    writer = csv.writer(w)
    for record in DATA:
        writer.writerow(record)


# writer对象提供一个writerow()方法

print('****REVIEW OF SAVED DATA')
with open('bookdata.csv', 'r', encoding='utf-8') as r:
    # csv.reader()用于返回一个可迭代对象，可以读取该对象并解析为CSV数据的每一行
    # csv.reader()使用一个已打开文件的句柄，返回一个reader对象
    # 当逐行迭代数据时，CSV数据会自动解析并返回给用户
    reader = csv.reader(r)
    for chap, title, modpkgs in reader:
        print('Chapter %s: %r (featureing %s)' % (chap, title, modpkgs))
```

* 输出结果

```
*** WRITTING CSV DATA
****REVIEW OF SAVED DATA
Chapter 1: 'Web Clients and Servers' (featureing base64,urllib)
Chapter 2: 'Web program：CGI & WSGI' (featureing cgi, time, wsgiref)
Chapter 3: 'Web Services' (featureing urllib,twython)
```
csv模块还提供csv.DictReader类和csv.DictWriter类，用于将CSV数据读进字典中(首先查找是否使用给定字段名，如果没有，就是用第一行作为键)，接着将字典字段写入CSV文件中。

## JSON
JSON是JavaScript的子集，专门用于指定结构化的数据。JSON是以人类更易读的方式传输结构化数据。
* JSON和Python类型之间的区别

|     JSON     |  Python3   |
|:------------:|:----------:|
|    object    |    dict    |
|    array     | list tuple |
|    string    |    str     |
| number(int)  |    int     |
| number(real) |   float    |
|     true     |    True    |
|    false     |   False    |
|     null     |    None    |

json提供了`dump()`/`load()`和`dumps()`/`loads()`。除了基本参数外，这些函数还包含许多仅用于JSON的选项。模块还包括encoder类和decoder类，用户既可以继承也可以直接使用。Python字典可以转化为JSON对象，Python列表和元组也可以转成对应的JSON数组。
```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

# 返回一个表示Python对象的字符串
# 用来美观地输出Python对象
from json import dumps
from pprint import pprint


# Python字典，使用字典是因为其可以构建具有结构化层次的属性。
# 在等价的JSON表示方法中，会移除所有额外的逗号
Books = {
    '0000001': {
        'title': 'Core',
        'edition': 2,
        'year': 2007,
    },
    '0000002': {
        'title': 'Python Programming',
        'edition': 3,
        'authors': ['Jack', 'Bob', 'Jerry'],
        'year': 2009,
    },
    '0000003': {
        'title': 'Programming',
        'year': 2009,
    }
}

# 显示转储的Python字典
print('***RAW DICT***')
print(Books)

# 使用更美观的方法输出
print('***PRETTY_PRINTED DICT***')
pprint(Books)

# 使用json.dumps()内置的美观的输出方式，传递缩进级别
print('***PRETTY_PRINTED JSON***')
print(dumps(Books, indent=4))
```
* 输出结果

```
***RAW DICT***
{'0000001': {'title': 'Core', 'edition': 2, 'year': 2007}, '0000002': {'title': 'Python Programming', 'edition': 3, 'authors': ['Jack', 'Bob', 'Jerry'], 'year': 2009}, '0000003': {'title': 'Programming', 'year': 2009}}
***PRETTY_PRINTED DICT***
{'0000001': {'edition': 2, 'title': 'Core', 'year': 2007},
 '0000002': {'authors': ['Jack', 'Bob', 'Jerry'],
             'edition': 3,
             'title': 'Python Programming',
             'year': 2009},
 '0000003': {'title': 'Programming', 'year': 2009}}
***PRETTY_PRINTED JSON***
{
    "0000001": {
        "title": "Core",
        "edition": 2,
        "year": 2007
    },
    "0000002": {
        "title": "Python Programming",
        "edition": 3,
        "authors": [
            "Jack",
            "Bob",
            "Jerry"
        ],
        "year": 2009
    },
    "0000003": {
        "title": "Programming",
        "year": 2009
    }
}
```

## XML
XML数据是纯文本数据，但是其可读性不高，所以需要使用解析器进行解析。
* 将字典转化为XML
```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

#
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


# Python字典，使用字典是因为其可以构建具有结构化层次的属性。
# 在等价的JSON表示方法中，会移除所有额外的逗号
Books = {
    '0000001': {
        'title': 'Core',
        'edition': 2,
        'year': 2007,
    },
    '0000002': {
        'title': 'Python Programming',
        'edition': 3,
        'authors': 'Jack:Bob:Jerry',
        'year': 2009,
    },
    '0000003': {
        'title': 'Programming',
        'year': 2009,
    }
}

# 创建顶层对象
# 将所有其他内容添加到该节点下
books = Element('books')
for isbn, info in Books.items():
    # 对于每一本书添加一个book子节点
    # 如果原字典没有提供作者和版本，则使用提供的默认值。
    book = SubElement(books, 'book')
    info.setdefault('authors', 'Bob')
    info.setdefault('edition', 1)
    for key, val in info.items():
        # 遍历所有键值对，将这些内容作为其他子节点添加到每个book中
        SubElement(book, key).text = ', '.join(str(val).split(':'))

xml = tostring(books)
print('*** RAW XML***')
print(xml)

print('***PRETTY-PRINTED XML***')
dom = parseString(xml)
print(dom.toprettyxml('     '))

print('***FLAT STRUCTURE***')
for elmt in books.iter():
    print(elmt.tag, '-', elmt.text)

print('\n***TITLE ONLY***')
for book in books.findall('.//title'):
    print(book.text)

```
* 输出结果
```
*** RAW XML***
b'<books><book><title>Core</title><edition>2</edition><year>2007</year><authors>Bob</authors></book><book><title>Python Programming</title><edition>3</edition><authors>Jack, Bob, Jerry</authors><year>2009</year></book><book><title>Programming</title><year>2009</year><authors>Bob</authors><edition>1</edition></book></books>'
***PRETTY-PRINTED XML***
<?xml version="1.0" ?>
<books>
     <book>
          <title>Core</title>
          <edition>2</edition>
          <year>2007</year>
          <authors>Bob</authors>
     </book>
     <book>
          <title>Python Programming</title>
          <edition>3</edition>
          <authors>Jack, Bob, Jerry</authors>
          <year>2009</year>
     </book>
     <book>
          <title>Programming</title>
          <year>2009</year>
          <authors>Bob</authors>
          <edition>1</edition>
     </book>
</books>

***FLAT STRUCTURE***
books - None
book - None
title - Core
edition - 2
year - 2007
authors - Bob
book - None
title - Python Programming
edition - 3
authors - Jack, Bob, Jerry
year - 2009
book - None
title - Programming
year - 2009
authors - Bob
edition - 1

***TITLE ONLY***
Core
Python Programming
Programming

```
