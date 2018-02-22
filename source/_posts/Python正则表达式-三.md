---
title: Python正则表达式(三)
date: 2018-02-12 09:55:17
categories: Python
copyright: true
tags:
    - Python
    - 正则表达式
---
在之前的两篇博文中，已经对正则表达式基本及核心的知识点进行了罗列和总结。而对于正则表达式的使用却缺乏实践。本文将基于《Python核心编程(第三版)》的练习题进行一些练习。
<!--More-->
## 正则表达式
1. > 识别后续的字符串：“bat”、“bit”、“but”、“hat”、“hit”或者“hut”。
   ```Python
   import re
   mode = re.compile(r'bat|bit|but|hat|hit|hut')
   #mode  = re.compile(r'[bh][iau]t')
   strs = ['bat', 'bit', 'but', 'hat', 'hit', 'hut']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   'bat'
   'bit'
   'but'
   'hat'
   'hit'
   'hut'
   ```

2. > 匹配由单个空格分隔的任意单词对，也就是姓和名。
   ```Python
   import re
   mode  = re.compile(r'^[A-Za-z]+ [A-Za-z]+$')
   strs = ['david Bob', 'D.Jone Steven', 'Lucy D May']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   'david Bob'
   ```

3. > 匹配由单个逗号和单个空白符分隔的任何单词和单个字母，如姓氏的首字母。
   ```Python
   import re
   mode = re.compile(r'[A-Za-z]+,\s[A-Za-z]+')
   strs = ['david, Bob', 'D.Jone, Steven', 'Lucy, D, May']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   'david, Bob'
   'Lucy, D'
   ```

4. > 匹配所有有效Python 标识符[^1]的集合。
   ```Python
   import re
   mode = re.compile(r'[^0-9][\w_]+')#用in排除关键字
   strs = ['1var', 'v_ar', '_var', 'var', 'var_9', 'var_']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   'v_ar'
   '_var'
   'var'
   'var_9'
   'var_'
   ```
5. > 根据读者当地的格式，匹配街道地址（使你的正则表达式足够通用，来匹配任意数
量的街道单词，包括类型名称）。例如，美国街道地址使用如下格式：1180 Bordeaux
Drive。使你的正则表达式足够灵活，以支持多单词的街道名称，如3120 De la Cruz
Boulevard。
    ```Python
    import re
    mode = re.compile(r'^\d{4}( [A-Z][a-z]+)+$')
    strs = ['1221 Bordeaux Drive', '54565 Bordeaux Drive', 'Bordeaux Drive', '1221 Bordeaux Drive Drive']
    for s in strs:
        if mode.match(s) is not None:mode.match(s).group()

    #输出结果
    '1221 Bordeaux Drive'
    '1221 Bordeaux Drive Drive'
    ```
6. > 匹配以“www”起始且以“.com”结尾的简单Web 域名；例如，www://www. yahoo.com/。
选做题：你的正则表达式也可以支持其他高级域名，如.edu、.net 等（例如，
http://www.foothill.edu）。
   ```Python
   import re
   mode = re.compile(r'^(http[s]?://)?www\.(\w+\.)+(com|net|edu)$')
   strs=['https://www.baidu.com', 'http://www.bilibili.com', 'www.baidu.com', 'baidu.com', 'www.cqupt.edu']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   'https://www.baidu.com'
   'http://www.bilibili.com'
   'www.baidu.com'
   'www.cqupt.edu'
   ```
7. > 匹配所有能够表示Python 整数的字符串集。
   ```Python
   import re
   mode = re.compile(r'^\d+[lL]?$')
   strs = ['123', '123l', '12312L']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   '123'
   '123l'
   '12312L'
   ```
8. > 匹配所有能够表示Python 长整数的字符串集。
   ```Python
   import re
   mode = re.compile(r'^\d+[lL]$')
   strs = ['123', '123l', '12312L']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()

   #输出结果
   '123l'
   '12312L'
   ```

9. > 匹配所有能够表示Python 浮点数的字符串集。
   ```Python
   import re
   mode = re.compile(r'(0|[1-9]\d*)(\.\d+)?$')
   strs = ['00.10', '0.123', '12.23', '12', '12.36l']
   for s in strs:
       if mode.match(s) is not None:mode.match(s).group()   

   #输出结果
   '0.123'
   '12.23'
   '12'
   ```
10. > 匹配所有能够表示Python 复数的字符串集。
    ```Python
    import re
    mode = re.compile(r'^((0|[1-9]\d*)(\.\d+)?\+)?((0|[1-9]\d*)(\.\d+)?j)?$')
    strs = ['12.3+1.2j', '1+2j', '4j']
    for s in strs:
        if mode.match(s) is not None:mode.match(s).group()   

    #输出结果
    '12.3+1.2j'
    '1+2j'
    '4j'
    ```
11. > 匹配所有能够表示有效电子邮件地址的集合（从一个宽松的正则表达式开始，然
后尝试使它尽可能严谨，不过要保持正确的功能）。
    ```Python
    import re
    mode = re.compile(r'^\w+@(\w+\.)+(com|com\.cn|net)$')
    strs = ['12345@qq.com', 'sina@163.com', 'qq@sina.com.cn', 'net@21cn.com', 'new123@163.sina.com']
    for s in strs:
        if mode.match(s) is not None:mode.match(s).group()   

    #输出结果
    '12345@qq.com'
    'sina@163.com'
    'qq@sina.com.cn'
    'net@21cn.com'
    'new123@163.sina.com'
    ```
12. > type()。内置函数type()返回一个类型对象，如下所示，该对象将表示为一个Pythonic
类型的字符串。
    ```Python
    import re
    mode = re.compile(r'<type \'(.*)\'>')
    strs = ['<type \'int\'>', '<type \'float\'>', '<type \'builtin_function_or_method\'>']
    for s in strs:
        if mode.match(s) is not None:mode.match(s).group(1)

    #输出结果
    'int'
    'float'
    'builtin_function_or_method'
    ```
13. > 处理日期。1.2 节提供了来匹配单个或者两个数字字符串的正则表达式模式，来表示1～
9 的月份(0?[1-9])。创建一个正则表达式来表示标准日历中剩余三个月的数字。
    ```Python
    import re
    mode = re.compile(r'1[0-2]')
    strs = ['10', '11', '12']
    for s in strs:
         if mode.match(s) is not None:mode.match(s).group()

    #输出结果
    '10'
    '11'
    '12'
    ```
14. > 创建一个允许使用连字符的正则表达式，但是仅能用于正确的位置。例如，15 位的信用卡号
码使用4-6-5 的模式，表明4 个数字-连字符-6 个数字-连字符-5 个数字；16 位的信用卡号码使用4-4-4-4 的模式。
    ```Python
    import re
    mode = re.compile(r'\d{4}-((\d{6}-\d{5})|(\d{4}-\d{4}-\d{4}))')
    strs = ['1234-567890-12345', '1234-5678-8012-3456']
    for s in strs:
        if mode.match(s) is not None:mode.match(s).group()

    #输出结果
    '1234-567890-12345'
    '1234-5678-8012-3456'
    ```
[^1]:标识符有字母、数字、下划线组成，但不能由数字开头
