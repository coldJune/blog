---
title: IPython入门
date: 2018-03-15 10:05:42
categories: true
copyright: true
tags:
    - IPython
    - 数据分析
description: IPython有一个可以直接进行绘图的GUI控制台、一个基于Web的交互式笔记本，以及一个轻量级的快速并行计算引擎。
---
## IPython基础
[IPython](https://ipython.org/)的环境需要自行安装。如果已经安装了Python，可以通过执行`pip install ipython`安装。然后只需要在命令行输入`ipython`就能启动：
```
Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:54:40) [MSC v.1900 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]:
```
可以在IPython中执行任何Python语句，和使用Python解释器一样：
```
In [1]: import numpy as np

In [2]: from numpy.random import randn

In [3]: data = {i:randn() for i in range(10)}

In [4]: data
Out[4]:
{0: -0.24193324837938815,
 1: 0.22563840475528563,
 2: 0.14465306885873513,
 3: 0.5076262433687561,
 4: 0.9067731627966235,
 5: 0.23827518072962814,
 6: 0.3233586627456586,
 7: 0.0327013232275763,
 8: -0.357340429464286,
 9: -1.4105691657079547}

In [5]:
```
许多Python对象都被格式化为可读性更好的形式

### Tab键自动完成
在shell中输入表达式时，只要按下Tab键，当前命名空间中任何与已输入的字符串相匹配的变量(对象、函数等)就会被找出来：
```
In [5]: an_example1 = 15

In [6]: an_example2 = 20

In [7]: an<TAB>
           an_example1               AnalogCommonProxyStub.dll
           an_example2               and
           any()
```
也可以在任何对象后面输入一个句点以便自动完成方法和属性的输入：
```
In [7]: a = [1, 2, 3]

In [8]: a.<TAB>
           append()  count()   insert()  reverse()
           clear()   extend()  pop()     sort()
           copy()    index()   remove()
```
应用在模块上:
```
In [8]: import datetime

In [9]: datetime.
                  date()        MAXYEAR       timedelta
                  datetime      MINYEAR       timezone
                  datetime_CAPI time()        tzinfo()
```
IPython默认会隐藏那些以下划线开头的方法和属性。如果需要应Tab键自动完成，可以先输入一个下划线。也可以直接修改IPython配置文件中的相关设置。
Tab键还可以找出电脑文件系统中与之匹配的东西：
```
In [6]: ca<TAB>
           callable()
           %%capture
           catchLink/
```
其中 *catchLibk/* 为当前目录下的一个子目录。在使用补全目录的时候需要使用正斜杠(/)，文件夹或文件名中间不能有空格。

### 内省
在变量前面或者后面加上一个问号(**?**)就可以将有关该对象的一些通用信息显示:
```
In [2]: b = []

In [3]: b?
Type:        list
String form: []
Length:      0
Docstring:
list() -> new empty list
list(iterable) -> new list initialized from iterable's items
```

如果该对象是一个函数或实例方法，则其docstring也会被显示出来：
```
In [4]: def add_number(a,b):
   ...:     """
   ...:     Add two numbers together
   ...:     Returns
   ...:     -----------------------
   ...:     the sum: type of arguments
   ...:     """
   ...:     return a+b
   ...:
   ...:

In [5]: add_number?
Signature: add_number(a, b)
Docstring:
Add two numbers together
Returns
-----------------------
the sum: type of arguments
File:      d:\python\<ipython-input-4-7144b04645ed>
Type:      function


```
使用??还将显示源代码:
```
In [6]: add_number??
Signature: add_number(a, b)
Source:
def add_number(a,b):
    """
    Add two numbers together
    Returns
    -----------------------
    the sum: type of arguments
    """
    return a+b
File:      d:\python\<ipython-input-4-7144b04645ed>
Type:      function
```
?还可以搜索IPython的命名空间，一些字符再配以通配符(\*)即可显示出所有与该通配符表达式相匹配的名称:
```
In [7]: import numpy as np

In [8]: np.*load*?
np.__loader__
np.load
np.loads
np.loadtxt
np.pkgload
```
### %run命令
在IPython会话环境中，所有文件都可以通过 *%run* 命令当做Python程序来运行。现在在目录下有一个叫做ipython_script_test.py的脚本：
```Python
#!/usr/bin/python3
# -*- coding:utf-8 -*-


def f(x, y, z):
    return (x+y) /z

a = 1
b = 2
c = 3
result = f(a, b, c)

```
然后运行，并且运行成功后该文件中所定义的全部变量(import、函数和全局变量)都可以在IPython shell中访问:
```
In [9]: %run ipython_script_test.py

In [10]: result
Out[10]: 1.0

In [11]: a
Out[11]: 1

```

### 中断正在执行的代码
任何代码在执行时只要按下“Ctrl-C/control-C”,就会引发一个KeyboardInterrupt，除非Python代码已经调用某个已编译的扩展模块需要等待Python解释器重新获取控制权外，绝大部分Python程序将立即停止执行。

### 执行剪切板中的代码
使用`%paste`和`%cpaste`两个魔术函数粘贴代码在shell中以整体执行：
* %paste
```
In [12]: %paste
def f(x, y, z):
    return (x+y) /z

a = 1
b = 2
c = 3
result = f(a, b, c)
## -- End pasted text --
```
* %cpaste
相比于`%paste`，`%cpaste`多出了一个用于粘贴代码的特殊提示符,若果发现粘贴的代码有错，只需按下“Ctrl-C/control-C”即可终止%cpaste提示符：
```
In [16]: %cpaste
Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
:def f(x, y, z):
:    return (x+y) /z
:
:a = 1
:b = 2
:c = 3
:result = f(a, b, c)
:--
```
### 键盘快捷键
IPython提供了许多用于提示符导航和查阅历史shell命令的键盘快捷键(**C指代Ctrl或control**)：
|           命令            |                               说明                                |
|:-------------------------:|:-----------------------------------------------------------------:|
|        C-P或上箭头        |           后向搜索命令历史中以当前输入的文本开头的命令            |
|        C-N或下箭头        |           前向搜索命令历史中以当前输入的文本开头的命令            |
|            C-R            |                 按行读取的反向历史搜索(部分匹配)                  |
| C-Shift-V/Command-Shift-V |                         从剪切板粘贴文本                          |
|            C-C            |                      终止当前正在执行的代码                       |
|            C-A            |                         将光标移动到行首                          |
|            C-E            |                         将光标移动到行尾                          |
|            C-K            |                    删除从光标开始至行尾的文本                     |
|            C-U            | 清楚当前行的所有文本(只是和C-K相反，即删除从光标开始至行首的文本) |
|            C-F            |                      将光标向前移动一个字符                       |
|            C-b            |                      将光标向后移动一个字符                       |
|            C-L            |                               清屏                                |

### 异常和跟踪
如果%run某段脚本或执行某条语句是发生异常，IPython会默认输出整个调用栈跟踪，其中还会附上调用栈各点附近的几行代码作为上下文参考:
```
In [17]: %run ipython_bug.py
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
D:\Python\ipython\ipython_bug.py in <module>()
      5 b = 2
      6 c = 0
----> 7 result = f(a, b, c)

D:\Python\ipython\ipython_bug.py in f(x, y, z)
      1 def f(x, y, z):
----> 2     return (x+y) /z
      3
      4 a = 1
      5 b = 2

ZeroDivisionError: division by zero
```

### 魔术命令
IPython有一些特殊命令，它们有的为常见任务提供便利，有的则使控制IPython系统的行为更轻松。魔术命令以百分号 **%** 为前缀的命令。例如通过`%timeit`检测任何Python语句的执行时间:
```
In [41]: a = np.random.randn(100,100)

In [42]: %timeit np.dot(a,a)
237 µs ± 40 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
魔术命令可以看做运行于IPython系统中的命令行程序，使用`?`即可查看其选项:
```
In [44]: %reset?
Docstring:
Resets the namespace by removing all names defined by the user, if
called without arguments, or by removing some types of objects, such
as everything currently in IPython's In[] and Out[] containers (see
the parameters for details).

Parameters
----------
-f : force reset without asking for confirmation.

-s : 'Soft' reset: Only clears your namespace, leaving history intact.
    References to objects may be kept. By default (without this option),
    we do a 'hard' reset, giving you a new session and removing all
    references to objects from the current session.

in : reset input history

out : reset output history

dhist : reset directory history

array : reset only variables that are NumPy arrays

See Also
--------
reset_selective : invoked as ``%reset_selective``

Examples
--------
::

  In [6]: a = 1

  In [7]: a
  Out[7]: 1

  In [8]: 'a' in _ip.user_ns
  Out[8]: True
```
魔术命令可以不带百分号使用，只要没有定义与其同名的变量。
* 常用的魔术命令
|         命令         |                                   说明                                    |
|:--------------------:|:-------------------------------------------------------------------------:|
|      %quickref       |                           显示Python的快速参考                            |
|        %magic        |                        显示所有魔术命令的详细文档                         |
|        %debug        |                  从最新的异常跟踪的底部进入交互式调试器                   |
|        %hist         |                       打印命令的输入(可选输出)历史                        |
|         %pdb         |                        在异常发生后自动进入调试器                         |
|        %paste        |                         执行剪切板中的Python代码                          |
|       %cpaste        |             打开一个特殊提示符以便手工粘贴待执行的Python代码              |
|        %reset        |                  删除interactive命名空间的全部变量/名称                   |
|     %page OBJECT     |                         通过分页器打印输出OBJECT                          |
|    %run script.py    |                     在IPython中执行一个Python脚本文件                     |
|   %prun statement    |             通过cProfile执行statement，并打印分析器的输出结果             |
|   %time statement    |                          报告statement的执行时间                          |
|  %timeit statement   | 多次执行statement以计算系统平均执行时间。对那些执行时间非常小的代码很有用 |
| %who、%who_is、%whos |         显示interactive命名空间中定义的变量，信息级别/冗余度可变          |
|    %xdel variable    |        删除variable，并参加过时清除其在IPython中的对象上的一切引用        |
