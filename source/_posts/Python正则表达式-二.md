---
title: Python正则表达式(二)
date: 2018-02-10 21:28:04
categories: Python
copyright: true
tags:
    - Python
    - 正则表达式
description:
---
正则表达式的匹配规则基本已经在上一篇博文中全部罗列出来了，下面便是结合到具体语言进行学习和练习了。
由于个人兴趣和想要专研的方向，在这里将会使用Python [^1] 语言进行描述。
<!--More-->

## 正则表达式和Python语言

### re模块：核心函数和方法
| 函数方法                                         | 描述                                                                                                                                        |
| :-----:                                          | :----:                                                                                                                                      |
| 仅仅是re函数模块                                 |                                                                                                                                             |
| compile(pattern, flags=0)                        | 使用任何可选的标记来编译正则表达式的模式，然后返回一个正则表达式对象                                                                        |
| re模块函数和正则表达式对象的方法                 |                                                                                                                                             |
| match(pattern, string, flags=0)                  | 尝试使用带有可选的标记的正则表达式的模式来匹配字符串，如果匹配成功，就返回匹配对象；如果失败，就返回None                                    |
| search(pattern, string, flags=0)                 | 使用可选标记搜索字符串中第一次出现的正则表达式模式。如果匹配成功，则返回匹配对象；如果匹配失败，怎返回None                                  |
| findall(pattern, string [,flags])                | 查找字符串中所有(非重复)出现的正则表达式模式，并返回一个匹配列表                                                                            |
| finditer(pattern, string[,flags])                | 与findall()函数相同，但返回的不是一个列表，而是一个迭代器。对于每一次匹配，迭代器都返回一个匹配对象                                         |
| split(pattern, string, max=0)                     | 根据正则表达式的模式分隔符，split函数将字符串分割为列表，然后返回成功的列表，分割最多操作max次(默认分割所有匹配成功的位置)                  |
| sub(pattern, repl, string, count=0)              | 使用repl替换所有正则表达式的模式在字符串中出现的位置，除非定义count，否则就讲替换所有出现的位置（另见subn()函数，该函数返回替换操作的数目） |
| purge()                                          | 清除隐式编译的正则表达式模式                                                                                                                |
| 常见的匹配对象方法                               |                                                                                                                                             |
| group(num=0)                                     | 返回整个匹配对象，或者编号为num的特定子组                                                                                                   |
| groups(default=None)                             | 返回一个包含所有匹配子组的元组(如果没有成功匹配，则返回一个空元组)                                                                          |
| groupdict(default=None)                          | 返回一个包含所有匹配的命名子组的字典，所有的子组名称作为字典的键(如果没有成功匹配，则返回一个空字典)                                        |
| 常用的模块属性（用于大多数正则表达式函数的标记） |                                                                                                                                             |
| re.I,re.IGNORECASE                               | 不去分大小写的匹配                                                                                                                          |
| re.L,re.LOCALE                                   | 根据所使用的本地语言环境通过\w、\w、\b、\B、\s、\S实现匹配                                                                                  |
| re.M,re.MULTILINE                                | ^和$分别匹配目标字符串中行的起始和结尾，而不是严格匹配整个字符串本身的起始和结尾                                                            |
| re.S,re.DOTALL                                   | "."(点号)通常匹配除了\n(换行符)之外的所有单个字符：该标记表示"."(点号)能匹配全部字符                                                        |
| re.X,re.VERBOSE                                  | 通过反斜线转移，否则所有空格加上#(以及在该行中后续文字)都被忽略，除非在一个字符类中或者允许注释并且提高可读性                                                                                                                                            |

### 部分方法总结

- *compile(pattern, flags=0)[^2]*
  >使用预编译使用推荐的方式，但不是必须的，可以通过设置标志位(上表已罗列出使用频繁的标记，详情可以[查阅文档](https://docs.python.org/3/library/re.html?highlight=re#module-re)),标志位通过 （|）合并

- *group(num=0)* 和 *groups(default=None)*
  >匹配对象[^3]的两个主要方法。 *group()* 要么返回整个匹配对象，要么按要求返回特定子组。 *groups()* 仅返回一个包含唯一或全部子组的元组。如果没有子组的要求，*group()* 返回整个匹配，*groups()* 返回一个空元组。

- *match(pattern, string, flags=0)*
   > *match()* 方法试图从字符串的**起始部分**对模式进行匹配。如果匹配成功，返回一个匹配对象；如果失败就返回None
   ``` python
   #匹配成功
   m = re.match('foo', 'foo') #模式匹配字符串
   if m is not None:         #如果匹配成功，就输出匹配内容
       m.group()

  'foo'                       #输出结果

  #匹配失败
  m  = re.match('foo', 'Bfoo') #模式匹配字符串
  if m is not None:           #如果匹配成功，就输出匹配内容
      m.group()

                              #因为起始字符为'B',所以匹配不成功，无任何输出
    ```

- *search(pattern, string, flags=0)*
  > *search()* 的工作方式和 *match()* 相同，不同之处在于 *search()* 会用它的字符串参数在**任意位置**对给定正则表达式模式搜索**第一次**出现的匹配情况。如果搜索到成功的匹配，就返回一个匹配对象；否则，就返回None。
  ```python
  #将上面使用match()方法匹配的串改用search()匹配
  m = re.search('foo', 'Bfoo') #模式匹配字符串
  if m is not None:            #如果匹配成功，就输出匹配内容
      m.group()

  'foo'                        #可以看到就算起始位置未能匹配，也能匹配成功
  ```

- *findall(pattern, string[,flags])* 和 *finditer(pattern, string[,flags])*
  > *findall()* 总是返回一个列表，如果没有找到匹配对象，返回一个空列表  
    *finditer()* 是一个与 *findall()* 类似但更节省内存的变体，*finditer()* 在匹配对象中迭代[^4]
  ```Python
  #findall()匹配
  re.findall('car', 'carry the barcardi to the car') #模式匹配字符串

  ['car', 'car', 'car']                              #返回结果

  #finditer()匹配
  iter = re.finditer('car', 'carry the barcardi to the car') #模式匹配字符串
  for i in iter:                                            #遍历迭代器
      print(i.group())

  #输出结果
  car
  car
  car
  ```

- *sub(pattern, repl, string, count=0)* 和 *subn(pattern, repl, string, count=0)*
  > *sub()* 和 *subn()* 用于实现搜索和替换功能。两者都是将某字符串中所有匹配正则表达式的部分进行某种形式的替换。和 *sub()* 不同的是，*subn()* 返回一个表示替换的总数，替换后的字符串和表示替换总数的数字一起作为一个拥有两个元素的元组返回
  ```Python
  #sub()
  re.sub('car', 'cat', 'My car is not only a car.') #模式匹配字符串


  'My cat is not only a cat.'                         #输出结果

  #subn()
  re.subn('car', 'cat', 'My car is not only a car.') #模式匹配字符串

  ('My cat is not only a cat.', 2)                   #输出结果
  ```

- *split(pattern, string, max=0)*
  > 正则表达式对象的 *split()* 方法和字符串的工作方式类似，但它是基于正则表达式的模式分割字符串。
  ```Python
  re.split(':', 'str1:str2:str3')               #模式匹配字符串

  ['str1', 'str2', 'str3']                      #输出结果，与'str1:str2:str3'.split(':')相同

  #split()复杂用法
  #使用split()基于逗号分割字符串，如果空格紧跟在5个数字或者两个大写字母之后，就用split()分割该空格
  #使用(?=)正向前视断言，不适用输入字符串 而是使用后面的空格作为分割字符串
  import re
  DATA = (
    'Mountain View, CA 94040',
    'Sunnyvale, CA',
    'Los Altos, 94023',
    'Cupertino 95014',
    'Palo Alto CA',
  )
  for datum in DATA:
      print(re.split(', |(?= (?:\d{5}|[A-Z]{2})) ', datum))

  #输出结果
  ['Mountain View', 'CA', '94040']
  ['Sunnyvale', 'CA']
  ['Los Altos', '94023']
  ['Cupertino', '95014']
  ['Palo Alto', 'CA']
  ```
### 符号的使用
#### `|` 与 `.` 和 `[]`
  > 包括择一匹配符号`|`、点号`.`，点号不匹配非字符或换行付\n（即空字符）
    字符集`[]`中的字符只取其一

#### 重复、特殊字符[^5]以及分组
  > `?`操作符表示前面的模式出现零次或一次
  > `+`操作符表示前面的模式出现至少一次
  > `*`操作符表示前面的模式出现任意次(包括0次)
  > 分组从左起第一个括号开始算第一个分组
  ```Python
  m  = re.match('(\w(\w\w))-(\d\d\d)','abc-123')
  m.group()                           #完整匹配
  'abc-123'                           #输出结果

  m.group(1)                          #第一组
  'abc'                               #输出结果    

  m.group(2)                          #第二组
  'bc'                                #输出结果

  m.group(3)                          #第三组
  '123'                               #输出结果

  m.groups()                          #全部子组
  ('abc', 'bc', '123')                #输出结果
  ```

[^1]:这里Python指代的是Python3.6.4

[^2]:预编译可以提升执行效率，而 `re.compile()` 方法提供了这个功能。模块函数会对已编译的对象进行缓存，所以无论使用 `match()` 和 `search()` 在执行时编译的正则表达式,还是使用 `compile()` 编译的表达式,在再次使用时都会查询缓存。但使用 `compile()` 同样可以节省查询缓存的时间

[^3]:除了正则表达式对象之外，还有另外一个对象类型：**匹配对象**。这些是成功调用 `match()` 和 `search()` 返回的对象。

[^4]:如果遇到无法调用 `next()`方法，可以使用 `\_\_next\_\_()`方法代替。

[^5]:特殊字符的详情可以参考[上一篇博文](http://coldjune.com/2018/02/09/Python%E6%AD%A3%E5%88%99%E8%A1%A8%E8%BE%BE%E5%BC%8F-%E4%B8%80/)
