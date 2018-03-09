---
title: CGI和WSGI
date: 2018-03-08 09:08:11
categories: Python
copyright: true
tags:
    - Web编程
    - CGI
    - WSGI
description: 对Python Web编程的广泛概述，从Web浏览到创建用户反馈表单，从识别URL到生成动态Web页面。本文先介绍通用网关接口CGI然后是Web服务器网关接口WSGI。
---
## CGI
这里将会主要介绍CGI的含义、与Web服务器的工作方式，使用Python创建CGI应用
### CGI简介
* **通用网关接口(Common Gateway Interface CGI)** 在Web服务器和应用之间充当了交互作用
  1. Web服务器从客户端接收到请求(GET或POST)，并调用相应的应用程序
  2. Web服务器和客户端等待HTML页面
  3. 应用程序处理完成后将会生成动态的HTML页面返回服务器，服务器将这个结果返回给用户
  4. 表单处理过程，服务器与外部应用程序交互，收到并生成的HTML页面通过CGI返回客户端
 含有需要用户输入项(文本框、单选按钮等)、Submit按钮、图片的Web页面，都会涉及某种CGI活动。创建HTML的CGI应用程序通常是高级语言来实现的，可以接受、处理用户数据，向服务器端返回HTML页面。*CGI有明显的局限性，以及限制Web服务器同时处理客户端的数量。(CGI被抛弃的原因)*

* CGI应用程序和和相关模块
  1. CGI应用程序
  CGI 应用程序和典型的应用程序主要区别在于输入、输出以及用户和程序的交互方面。当一个CGI脚本启动后，需要获得用户提供的表单数据，但这些数据必须从Web客户端才可以获得，这就是 *请求(request)*。与标准输出不同，这些输出将会发送回连接的Web客户端，而不是发送到屏幕、GUI窗口或者硬盘上。这些返回的数据必须是具有一系列有效头文件的HTML标签数据。**用户和脚本之间没有任何交互，所有交互都发生在Web客户端(基于用户的行为)、Web服务器端和CGI应用程序间**。

  2. cgi模块
  cgi模块有一个主要类 *FieldStorage* 完成了所有的工作。Python CGI脚本启动会实例化这个类，通过Web服务器从Web客户端读出相关的用户信息。在实例化完成后，其中会包含一个类似字典的对象，它具有一系列键值对。键就是通过表单传入的表单条目的名字，而值则包含响应的数据。
  这些值有三个对象：*FieldStorage* 对象；*MiniFieldStorage* 对象用在没有文件上传或mulitple-part格式数据的情况下，*MiniFieldStorage* 实例只包含名称和数据的键值对；当表单中的某个字段有多个输入值时，还可以是这些对象的列表。

  3. cgitb模块
  cgitb模块用于在浏览器中看到Web应用程序的回溯信息，而不是“内部服务器错误”。

### CGI应用程序
>再启动服务器的目录下创建一个cgi-bin目录，放入Python CGI脚本。将一些HTML文件放到启动服务器的目录中。确保启动服务器目录中有个cgi-bin目录，同时确保其中有相应的.py文件。否则服务器将会把Python文件作为静态文本返回而不是执行它们

* CGI服务器

```Python
#!/usr/bin/python3
# -*- coding:UTF-8 -*-

from cgi import FieldStorage
from os import environ
from io import StringIO
from urllib.parse import quote, unquote


class AdvCGI(object):
    # 创建header和url静态类变量，在显示不同页面的方法中会用到这些变量
    header = 'Content-Type:text/html\n\n'
    url = '/cgi-bin/advcgi.py'
    # HTML静态文本表单，其中含有程序语言设置和每种语言的HTML元素
    formhtml = '''
        <HTML>
            <HEAD>
                <TITLE>Advanced CGI Demo</TITLE>
            </HEAD>
            <BODY>
                <H2>Advanced CGI Demo</H2>
                <FORM METHOD=post ACTION='%s' ENCTYPE='multipart/form-data'>
                    <H3>My Cookie Setting</H3>
                    <LI>
                        <CODE><B>CPPuser = %s</B></CODE>
                        <H3>Enter cookie value<BR>
                            <INPUT NAME=cookie value='%s'/>(<I>optional</I>)
                        </H3>
                        <H3>Enter your name<BR>
                            <INPUT NAME=person VALUE='%s'/>(<I>required</I>)
                        </H3>
                        <H3>What languages can you program in ?
                        (<I>at least one required</I>)  
                        </H3>
                        %s
                        <H3>Enter file to upload<SMALL>(max size 4k)</SMALL></H3>
                        <INPUT TYPE=file NAME=upfile VALUE='%s' SIZE=45>
                        <P><INPUT TYPE=submit />
                    </LI>
                </FORM>
            </BODY>
        </HTML>
    '''
    langset = ('Python', 'Java', 'C++', 'C', 'JavaScript')

    langItem = '<INPUT TYPE=checkbox NAME=lang VALUE="%s"%s> %s\n'


    def get_cpp_cookies(self):
        """
        当浏览器对应用进行连续调用时，将相同的cookie通过HTTP头发送回服务器
        :return:
        """
        # 通过HTTP_COOKIE访问这些值
        if 'HTTP_COOKIE' in environ:
            cookies = [x.strip() for x in environ['HTTP_COOKIE'].split(';')]
            for eachCookie in cookies:
                # 寻找以CPP开头的字符串
                # 只查找，名为“CPPuser”和“CPPinfo”的cookie值
                if len(eachCookie) > 6 and eachCookie[:3] == 'CPP':
                    # 去除索引8处的值进行计算，计算结果保存到Python对象中
                    tag = eachCookie[3:7]
                    try:
                        # 查看cookie负载，对于非法的Python对象，仅仅保存相应的字符串值。
                        self.cookies[tag] = eval(unquote(eachCookie[8:]))
                    except (NameError, SyntaxError):
                        self.cookies[tag] = unquote(eachCookie[8:])
            # 如果这个cookie丢失，就给他指定一个空字符串
            if 'info' not in self.cookies:
                self.cookies['info'] = ''
            if 'user' not in self.cookies:
                self.cookies['user'] = ''
        else:
            self.cookies['info'] = self.cookies['user'] = ''

        if self.cookies['info'] != '':
            self.who, langstr, self.fn = self.cookies['info'].split(';')
            self.langs = langstr.split(',')
        else:
            self.who = self.fn = ''
            self.langs = ['Python']

    def show_form(self):
        """
        将表单显示给用户
        :return:
        """
        # 从之前的请求中(如果有)获取cookie，并适当地调整表单的格式
        self.get_cpp_cookies()

        langstr = []
        for eachLang in AdvCGI.langset:
            langstr.append(AdvCGI.langItem % (
                eachLang, ' CHECKED' if eachLang in self.langs else '', eachLang))

        if not ('user' in self.cookies and self.cookies['user']):
            cookstatus = '<I>(cookie has not been set yet)</I>'
            usercook = ''
        else:
            usercook = cookstatus = self.cookies['user']

        print('%s%s' % (AdvCGI.header, AdvCGI.formhtml % (
            AdvCGI.url, cookstatus, usercook, self.who,
            ''.join(langstr), self.fn)))

    errhtml = '''
            <HTML>
                <HEAD>
                    <TITLE>Advanced CGI Demo</TITLE>
                </HEAD>
                <BODY>
                    <H3>ERROR</H3>
                    <B>%s</B>
                    <P>
                    <FORM>
                        <INPUT TYPE= button VALUE=Back ONCLICK="window.history.back()"></INPUT>
                    </FORM>
                </BODY>
            </HTML>
    '''

    def show_error(self):
        """
        生成错误页面
        :return:
        """
        print('%s%s' % (AdvCGI.header, AdvCGI.errhtml % (self.error)))

    reshtml = '''
    <HTML>
        <HEAD>
            <TITLE>Advanced CGI Demo</TITLE>
        </HEAD>
        <BODY>
            <H2>Your Uploaded Data</H2>
            <H3>Your cookie value is: <B>%s</B></H3>
            <H3>Your name is: <B>%s</B></H3>
            <H3>You can program in the following languages:</H3>
            <UL>%s</UL>
            <H3>Your uploaded file...<BR>
                Name: <I>%s</I><BR>
                Contents:
            </H3>
            <PRE>%s</PRE>
            Click <A HREF="%s"><B>here</B></A> to return to form.
        </BODY>
    </HTML>'''

    def set_cpp_cookies(self):
        """
        应用程序调用这个方法来发送cookie（从Web服务器）到浏览器，并存储在浏览器中
        :return:
        """
        for eachCookie in self.cookies:
            print('Set-Cookie: CPP%s=%s; path=/' % (
                eachCookie, quote(self.cookies[eachCookie])))

    def doResult(self):
        """
        生成结果页面
        :return:
        """
        MAXBYTES = 4096
        langlist = ''.join('<LI>%s<BR>' % eachLang for eachLang in self.langs)
        filedata = self.fp.read(MAXBYTES)
        if len(filedata) == MAXBYTES and f.read():
            filedata = '%s%s' % (filedata, '...<B><I>(file truncated due to size)</I></B>')
        self.fp.close()

        if filedata == '':
            filedata = '<B><I>(file not give or upload error)</I></B>'
        filename = self.fn

        if not ('user' in self.cookies and self.cookies['user']):
            cookstatus = '<I>(cookie has not been set yet)</I>'
            usercook = ''
        else:
            usercook = cookstatus = self.cookies['user']

        self.cookies['info'] = ':'.join((self.who, ','.join(self.langs), filename))
        self.set_cpp_cookies()

        print('%s%s' % (
            AdvCGI.header, AdvCGI.reshtml % (cookstatus, self.who, langlist, filename, filedata, AdvCGI.url)))

    def go(self):
        self.cookies = {}
        self.error = ''
        form = FieldStorage()
        if not list(form.keys()):
            self.show_form()
            return

        if 'person' in form:
            print(form.keys())
            self.who = form['person'].value.strip().title()
            if self.who == '':
                self.error = 'Your name is required.(blank)'
        else:
            self.error = 'Your name is required.(missing)'

        self.cookies['user'] = unquote(form['cookie'].value.strip())
                                                      if 'cookie' in form else ''

        if 'lang' in form:
            lang_data = form['lang']
            if isinstance(lang_data, list):
                self.langs = [eachLang.value for eachLang in lang_data]
            else:
                self.langs = [lang_data.value]
        else:
            self.error = 'At least one language required'

        if 'upfile' in form:
            upfile = form['upfile']
            self.fn = upfile.filename or ''
            if upfile.file:
                self.fp = upfile.file
            else:
                self.fp = StringIO('(no data)')
        else:
            self.fp = StringIO('(no file)')
            self.fn = ''

        if not self.error:
            self.doResult()
        else:
            self.show_error()

if __name__ == '__main__':
    page = AdvCGI()
    page.go()

```
* 启动程序
>将启动程序放在启动目录中，然后执行。

```Python
#!/usr/bin/python
# -*- coding:UTF-8 -*-

from http.server import CGIHTTPRequestHandler, test

if __name__ == '__main__':
    test(CGIHTTPRequestHandler)
```
[源代码](https://github.com/coldJune/Python/tree/master/web)
## WSGI
*WSGI*[^1] 是为了替代CGI而出现的。

### 服务器集成和外部进程
* 服务器集成
服务器集成也叫 **服务器API**，其针对CGI性能的解决方案是将网关集成进服务器，不是讲服务器切分成多个语言解释器来分别处理请求，而是生成函数调用，运行应用程序代码，在运行过程中进行响应。服务器根据对应的API通过一组预先创建的进程或线程处理工作。
服务器API的会使含有bug的代码影响服务器执行效率，不同语言的实现无法兼容，应用程序必须线程安全。
* 外部进程
外部进程让CGI应用在服务器外部运行。当有请求进入时，服务器将这个请求传递到外部进程中。外部进程存在时间长，不是处理完单个请求后就终止，所以其扩展性比纯CGI好。
因为使用了不同的调用机制，所以造成开发者的负担，不仅要开发应用本省，还要决定于服务器的集成。

### WSGI简介
WSGI只是定义的一个接口，其目标是在Web服务器和Web框架层之间提供一个通用的API标准，减少之间的会操作性并形成统一的调用方式。
根据WSGI定义，其应用是可调用对象，其参数固定为：含有服务器环境变量的字典；可调用对象，该对象使用HTTP状态码和返回给客户端的HTTP头来初始化响应。

### WSGI服务器
在服务器端，必须调用应用，传入环境变量和start_response()这个可调用对象，接着等待应用执行完毕。在执行完成后，必须获得返回的可迭代对象，将这些数据返回给客户端。

[^1]:WSGI只是做一个简单的了解，可以结合框架一起看。
