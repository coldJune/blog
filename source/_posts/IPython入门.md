---
title: IPython入门
date: 2018-03-15 10:05:42
categories: 数据分析
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
使用`??`还将显示源代码:
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
`?`还可以搜索IPython的命名空间，一些字符再配以通配符(\*)即可显示出所有与该通配符表达式相匹配的名称:
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
在IPython会话环境中，所有文件都可以通过`%run`命令当做Python程序来运行。现在在目录下有一个叫做ipython_script_test.py的脚本：
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
如果`%run`某段脚本或执行某条语句是发生异常，IPython会默认输出整个调用栈跟踪，其中还会附上调用栈各点附近的几行代码作为上下文参考:
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
IPython有一些特殊命令，它们有的为常见任务提供便利，有的则使控制IPython系统的行为更轻松。魔术命令以百分号 `%` 为前缀的命令。例如通过`%timeit`检测任何Python语句的执行时间:
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

### matplotlib集成与pylab模式
启动IPython时加上`--pylab`标记来集成matplotlib`ipython --pylab`。这样IPython会默认GUI后台集成，就可以创建matplotlib绘图了。并且NumPy和matplotlib的大部分功能会被引入到最顶层的interactive命名空间以产生一个交互式的计算环境。也可以通过`%gui`对此进行手工设置。
```
Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:54:40) [MSC v.1900 64 bit (AMD64)]
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
Using matplotlib backend: TkAgg

In [1]:

```

## 使用命令历史
IPython维护着一个位于硬盘上的小型数据库，其中含有执行过的每条命令的文本：

1. 只需很少的按键次数即可搜索、自动完成并执行之前已经执行过的命令
2. 在会话间持久化命令历史
3. 将输入/输出历史记录到日志文件

### 搜索并重用命令历史
如果需要输入之前执行过的相同的命令，只需要按照上面的快捷键表操作，就可以搜索出命令历史中第一个与输入的字符相匹配的命令。既可以后向搜索也可以前向搜索。
### 输入和输出变量
IPython会将输入(输入的文本)和输出(返回的对象)的引用保存在一些特殊变量中。最近的两个输出结果分别保存在 `_`(一个下划线)和 `__`(两个下划线)变量中：
```
In [6]: 1+1
Out[6]: 2

In [7]: _
Out[7]: 2

In [8]: _+1
Out[8]: 3

In [9]: 3+1
Out[9]: 4

In [10]: __
Out[10]: 3
```
输入的文本保存在名为`_ix`的变量中，其中 **X** 是输入行的行号。每个输入变量都有一个对应的输出变量`_x`:
```
In [11]: _i6
Out[11]: '1+1'

In [12]: _6
Out[12]: 2
```
由于输入变量是字符串，因此可以用Python的`exec()`方法重新执行:
```
In [18]: exec(_i6)

In [19]: _
Out[19]: '1+1'
```
有几个魔术命令可以用于控制输入和输出历史。`%hist`用于打印全部或部分输入历史，可以选择是否带行号。`%reset`用于清空interactive命名空间，并可选择是否清空输入和输出缓存。`%xdel`用于从IPython系统中移除特定对象的一切引用。
### 记录输入和输出
IPython能够记录整个控制台会话，包括输入和输出。执行`%logstart`即可开始记录日志：
```
In [20]: %logstart
Activating auto-logging. Current session state plus future input saved.
Filename       : ipython_log.py
Mode           : rotate
Output logging : False
Raw input log  : False
Timestamping   : False
State          : active
```
IPython的日志功能可以在任何时刻开启。还有与`%logstart`配套的`%logoff`、`%logon`、`%logstate`和`%logstop`，可以参考其文档。
### 与操作系统交互
可以在IPython中实现标准的Windows或UNIX命令行活动，将命令的执行结果保存在Python对象中

* 跟系统相关的IPython魔术命令

|         命令          |               说明               |
|:---------------------:|:--------------------------------:|
|         !cmd          |       在系统shell中执行cmd       |
|   output=!cmd args    | 执行cmd，并将stout存放在output中 |
| %alias alias_name cmd |     为系统shell命令定义别名      |
|       %bookmark       |    使用IPython的目录书签系统     |
|     %cd directory     |  将系统工作目录更改为directory   |
|         %pwd          |      返回系统的当前工作目录      |
|   %pushd directory    |  将当前目录入栈，并转向目标目录  |
|         %popd         |    弹出栈顶目录，并转向该目录    |
|         %dirs         |   返回一个含有当前目录栈的列表   |
|        %dhist         |         打印目录访问历史         |
|         %env          |    以dict形式返回系统环境变量    |

#### shell命令和别名
在IPython中，以感叹号(!)开头的命令行表示其后的所有内容需要在系统shell中执行:
```
In [23]: !python
Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:54:40) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
还可以将shell命令的控制台输出存放到变量中，只需将 `!` 开头的表达式赋值给变量:
```
In [152]: ip_info = !ls

In [153]: ip_info
Out[153]: ['experiment.py', 'ipython_bug.py', 'ipython_script_test.py']
```

## 软件开发工具
IPython集成并加强了Python内置的pdb调试器，同时提供了一些简单易用的代码运行时间及性能分析工具。
### 交互式调试器
IPython的调试器增强了pdb，如Tab键自动完成、语法高亮、为异常跟踪的每条信息添加上下文参考。`%debug`命令(在发生异常之后马上输入)将会调用那个“事后”调试器，并直接跳转到引发异常的那个栈帧：
```
In [45]: %run ipython_bug.py
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

In [46]: %debug
> d:\python\ipython\ipython_bug.py(2)f()
      1 def f(x, y, z):
----> 2     return (x+y) /z
      3
      4 a = 1
      5 b = 2
```
在这个调试器中，可以执行任意Python代码并查看各个栈帧中的一切对象和数据。默认是从最低级开始(即错误发生的地方)。输入`u`(或up)和`d`(或down)即可在栈跟踪的各级别之间切换:
```
ipdb> u
> d:\python\ipython\ipython_bug.py(7)<module>()
      3
      4 a = 1
      5 b = 2
      6 c = 0
----> 7 result = f(a, b, c)

ipdb> d
> d:\python\ipython\ipython_bug.py(2)f()
      1 def f(x, y, z):
----> 2     return (x+y) /z
      3
      4 a = 1
      5 b = 2
```
执行`%pdp`命令可以让IPython在出现异常之后自动调用调试器。
如果需要设置断点或对函数/脚本进行单步调试以查看各条语句的执行情况时，可以使用带有`-d`选项的`%run`命令，这会在执行脚本文件中的代码之前打开调试器，然后输入`s`(或step)步进才能进入脚本:
```
In [50]: %run -d ipython_bug.py
Breakpoint 1 at d:\python\ipython\ipython_bug.py:1
NOTE: Enter 'c' at the ipdb>  prompt to continue execution.
> d:\python\ipython\ipython_bug.py(1)<module>()
1---> 1 def f(x, y, z):
      2     return (x+y) /z
      3
      4 a = 1
      5 b = 2

ipdb> s
> d:\python\ipython\ipython_bug.py(4)<module>()
      2     return (x+y) /z
      3
----> 4 a = 1
      5 b = 2
      6 c = 0

ipdb> s
> d:\python\ipython\ipython_bug.py(5)<module>()
      3
      4 a = 1
----> 5 b = 2
      6 c = 0
      7 result = f(a, b, c)

ipdb> s
> d:\python\ipython\ipython_bug.py(6)<module>()
      3
      4 a = 1
      5 b = 2
----> 6 c = 0
      7 result = f(a, b, c)
```
通过`b num`在num行出设置断点，输入`c`(或continue)使脚本一直运行下去直到该断点时为止,然后输入`n`(或next)直到执行下一行(即step over):
```
In [53]: %run -d ipython_bug.py
Breakpoint 1 at d:\python\ipython\ipython_bug.py:1
NOTE: Enter 'c' at the ipdb>  prompt to continue execution.
> d:\python\ipython\ipython_bug.py(1)<module>()
1---> 1 def f(x, y, z):
      2     return (x+y) /z
      3
      4 a = 1
      5 b = 2

ipdb> b 7
Breakpoint 2 at d:\python\ipython\ipython_bug.py:7
ipdb> c
> d:\python\ipython\ipython_bug.py(7)<module>()
      3
      4 a = 1
      5 b = 2
      6 c = 0
2---> 7 result = f(a, b, c)

ipdb> n
ZeroDivisionError: division by zero
> d:\python\ipython\ipython_bug.py(7)<module>()
      3
      4 a = 1
      5 b = 2
      6 c = 0
2---> 7 result = f(a, b, c)

ipdb> n
--Return--
None
> d:\python\ipython\ipython_bug.py(7)<module>()
      3
      4 a = 1
      5 b = 2
      6 c = 0
2---> 7 result = f(a, b, c)
```
* IPython调试器命令

|           命令           |                     功能                     |
|:------------------------:|:--------------------------------------------:|
|          h(elp)          |                 显示命令列表                 |
|       help command       |              显示command的文档               |
|        c(ontinue)        |                恢复程序的执行                |
|          q(uit)          |         退出调试器，不再执行任何代码         |
|     b(readk) number      |      在当前文件的第number行设置一个断点      |
| b path/to/file.py:number |      在指定文件的第number行设置一个断点      |
|          s(tep)          |               单步进入函数调用               |
|          n(ext)          |     执行当前行，并前进到当前级别的下一行     |
|       u(p)/d(own)        |         在函数调用栈中向上或向下移动         |
|          a(rgs)          |              显示当前函数的参数              |
|     debug statement      |    在新的(递归)调试器中调用语句statement     |
|    l(ist)  statement     |  显示当前行，以及当前栈级别的上下文参考代码  |
|         w(here)          | 打印当前位置的完整栈跟踪(包括上下文参考代码) |

### 测试代码的执行时间:%time和%timeit
`%time`一次执行一条语句，然后报告总体执行时间
```
In [56]: strings = ['foo','bar','abc','foobar','python','Guide Peple']*100000

In [57]: %time method1 = [x for x  in strings if x.startswith('foo')]
Wall time: 102 ms

In [58]: %time method2 = [x for x in strings if x[:3] == 'foo']
Wall time: 59.2 ms
```

`%timeit`对于任意语句，它会自动多次执行以产生一个非常精确的平均执行时间
```
In [59]: %timeit method1 = [x for x  in strings if x.startswith('foo')]
100 ms ± 5.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [60]: %timeit method2 = [x for x in strings if x[:3] == 'foo']
57 ms ± 7.12 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
### 基本性能分析：%prun和%run -p
代码的性能分析跟代码执行时间密切相关，只不过它关注的事耗费时间的位置，主要的Python性能分析工具是cProfile模块。CProfile在执行一个程序或代码块时，会记录各函数所耗费的时间。CProfile一般在命令行上使用，它将执行整个程序然后输出各函数的执行时间。`%prun`分析的是Python语句而不是整个.py文件：
```
In [141]: %cpaste
Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
:def run_experiment(niter=100):
    k = 100
    results = []
    for _ in range(niter):
        mat = np.random.randn(k, k)
        max_eigenvalue = np.abs(eigvals(mat)).max()
        results.append(max_eigenvalue)
    return results:::::::
:
:--

In [142]: %prun -l 7 -s cumulative run_experiment()
         3804 function calls in 0.901 seconds

   Ordered by: cumulative time
   List reduced from 31 to 7 due to restriction <7>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.901    0.901 {built-in method builtins.exec}
        1    0.000    0.000    0.901    0.901 <string>:1(<module>)
        1    0.002    0.002    0.901    0.901 <ipython-input-141-78ef833ef08b>:1(run_experiment)
      100    0.814    0.008    0.838    0.008 linalg.py:834(eigvals)
      100    0.060    0.001    0.060    0.001 {method 'randn' of 'mtrand.RandomState' objects}
      100    0.012    0.000    0.018    0.000 linalg.py:213(_assertFinite)
      300    0.008    0.000    0.008    0.000 {method 'reduce' of 'numpy.ufunc' objects}

```
执行`%run -p -s cumulative experiment.py`也能达到以上的效果，无需退出IPython:
```
In [75]: %run -p -l 7 -s cumulative experiment.py
Largest one we saw:11.9165340849
         3888 function calls (3887 primitive calls) in 0.467 seconds

   Ordered by: cumulative time
   List reduced from 77 to 7 due to restriction <7>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000    0.467    0.467 {built-in method builtins.exec}
        1    0.000    0.000    0.467    0.467 <string>:1(<module>)
        1    0.000    0.000    0.467    0.467 interactiveshell.py:2445(safe_execfile)
        1    0.000    0.000    0.467    0.467 py3compat.py:182(execfile)
        1    0.000    0.000    0.467    0.467 experiment.py:1(<module>)
        1    0.001    0.001    0.466    0.466 experiment.py:5(run_experiment)
      100    0.431    0.004    0.436    0.004 linalg.py:819(eigvals)
```
## ipython html notebook
需要安装 *jupyter* 来使用该功能:
```
pip3 install jupyter
```
这是一个基于Web的交互式计算文档格式。它有一种基于JSON的文档格式.upynb，可以轻松分享代码、输出结果以及图片等内容。执行如下命令启动：
```
jupyter notebook
```
这是运行于命令行上的轻量级服务器进程，Web浏览器会自动打开Notebook的仪表盘。
