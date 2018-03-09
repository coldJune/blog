---
title: 'Web框架:Django'
date: 2018-03-09 15:39:07
categories: Python
copyright: true
tags:
    - Web框架
    - Django
description: Web框架可以用于提供Web应用的所有相关服务，如Web服务器、数据库ORM、模板和所有需要的中间件hook
---
## Django简介
* 安装
在使用[Django](https://www.djangoproject.com/)开发之前，必须安装必需的组件，包括依赖组件和Django本身
```
 pip3 install django
```
* 项目和应用
**项目** 是指的一系列文件，用来创建并运行一个完整的Web站点。在项目文件夹下，有一个或多个子文件夹，每个文件夹有特定的功能，称为 **应用**。应用不一定要位于项目文件夹中。应用可以专注于项目某一方面的功能，或可以作为通用组件，用于不同的项目。应用是一个具有特定功能的子模块，这些子模块组合起来就能完成Web站点的功能。
1. 在Django中创建项目
Django自带有一个名为`django-admin.py`/`django-admin.exe`的工具，它可以简.
化任务。在POSIX平台上，一般在`/usr/local/bin`、`/usr/bin`这样的目录中。使用Windows系统会安装在Python包下的Scripts目录下，如`E:\Python\Python36\Scripts`。两种系统都应该确保文件位于PATH环境变量中。
在项目文件加下执行命令创建项目:
```
django-admin.py startproject mysite
```
2. Django项目文件
|   文件名    |        描述/用途         |
|:-----------:|:------------------------:|
| __init__.py | 告诉Python这是一个软件包 |
|   urls.py   |  全局URL配置("URLconf")  |
| setting.py  |      项目相关的配置      |
|  manage.py  |     应用的命令行接口     |
* 运行开发服务器
Django内置Web服务器，该服务器运行在本地，专门用于开发阶段，仅用于开发用途。使用开发服务器有以下几个优点：
1. 可以直接运行与测试项目和应用，无需完整的生产环境
2. 当改动Python源码文件并重新载入模块时，开发服务器会自动检测，无须每次编辑代码后手动重启
3. 开发服务器知道如何为Django管理应用程序寻找和显示静态媒体文件，所以无须立即了解管理方面的内容
>启动服务器
```
python manage.py runserver
```
