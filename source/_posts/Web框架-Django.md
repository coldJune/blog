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

## 应用
### 创建应用
在项目目录下使用如下命令创建一个应用：
```
python3 ./manage.py startapp blog
```
这样就建立了一个blog目录，其中有如下内容：
|  文件名   |                               描述/目的                               |
| :-------: | :-------------------------------------------------------------------: |
| __init.py |                         告诉Python这是一个包                          |
|  urls.py  | 应用的URL配置文件("URLconf")，这个文件并不像项目的URLconf那样自动创建 |
| models.py |                                数据模型                               |
|  views.py |                       视图函数(即MVC中的控制器)                       |
|  tests.py |                                单元测试                               |
与项目类似，应用也是一个Python包。本地应用的URLconf需要手动创建，接着使用URLconf里的include()指令将请求分配给应用的URLconf。为了让Django知道这个新应用是项目的一部分，需要编辑 *settings.py*，将应用名称(**blog**)添加到元组的末尾。Django使用 **INSTALLED_APPS**来配置系统的不同部分，包括自动管理应用程序和测试框架。
```Python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog'
]
```
### 创建模型添加数据库服务
#### 创建模型
models.py将定义博客的数据结构，首先创建一个基本类型。数据模型表示将会存储在数据库每条记录的数据类型。Django提供了许多[字段类型](https://docs.djangoproject.com/en/2.0/ref/models/fields/)，用来将数据映射到应用中。
```Python
from django.db import models

# Create your models here.


class BlogPost(models.Model):
    """
    django.db.models.Model的子类Model是Django中用于数据模型的标准基类。
    BlogPost中的字段像普通类属性那样定义，
    每个都是特定字段类的实例，每个实例对应数据库中的一条记录。
    """
    title = models.CharField(max_length=150)
    body = models.TextField()
    timestamp = models.DateTimeField()
```
#### 创建数据库
在项目的*setting.py*文件中设置数据库。关于数据库，有6个相关设置(有时只需要两个):**ENGINE**、**NAME**、**HOST**、**PORT**、**USER**、**PASSWORD**。只需要在相关设置选项后面添上需要让Django使用的数据库服务器中合适的📄即可。
* 使用MySQL
```Python
DATABASES = {
    # 使用mysql
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'django_test',
        'USER': 'root',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```
*使用SQLite
SQLite一般用于测试，它没有主机、端口、用户、密码信息。因为其使用本地文件存储信息，本地文件系统的访问权限就是数据库的访问控制。SQLite不仅可以使用本地文件，还可以使用纯内存数据库。使用实际的Web服务器(如Apache)来使用SQLite时，需要确保拥有Web服务器进程的账户同时拥有数据库文件本身和含有数据库文件目录的写入权限。
```Python
DATABASES = {
    # 使用sqlite
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```
#### 创建表
使用 *makemigrations* 参数创建映射文件，当执行命令时Django会查找INSTALLED_APPS中列出的ing用的models.py文件。对于每个找到的模型，都会创建一个映射表。
```
python3 ./manage.py makemigrations
```
使用 *migrate*映射到数据库
```
python3 ./manage.py migrate
```

### Python应用Shell
#### 在Django中使用Python shell
即使没有模版(view)或视图(controller)，也可以通过添加一些BlogPost项来测试数据模型。如果应用由RDBMS支持，则可以为每个blog项的表添加一个数据记录。如果使用的是NoSQL数据库，则需要向数据库中添加其他对象、文档或实体。通过以下命令启动shell(使用对应版本)：
```
python3 ./manage.py shell

Python 3.6.4 (default, Jan  6 2018, 11:51:59)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
In [1]:
```
[Django shell](https://docs.djangoproject.com/en/dev/intro/tutorial01/#playing-with-the-api)和标准的shell相比更专注于Django项目的环境，可以与视图函数和数据模型交互，这个shell会自动设置环境变量，包括sys.path，它可以访问Django与自己项目中的模块和包，否则需要手动配置。除了标准shell之外，还有其他的交互式解释器可供选择。Django更倾向于使用功能丰富的shell，如IPython和bpython，这些shell在普通的解释器基础上提供及其强大的功能。运行shell命令时，Django首先查找含有扩展功能的shell，如果没有回返回标准解释器。这里使用的是IPython。也可以使用 *-i*来强制使用普通解释器。
```
python3 ./manage.py shell -i python

Python 3.6.4 (default, Jan  6 2018, 11:51:59)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39.2)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>>
```
#### 测试数据模型
在启动Python shell之后输入一些Python或IPython命令来测试应用及其数据模型。
```
In [1]: from datetime import datetime

In [2]: from blog.models import BlogPost

In [3]: BlogPost.objects.all()
Out[3]: <QuerySet [<BlogPost: BlogPost object (1)>, <BlogPost: BlogPost object (2)>, <BlogPost: BlogPost object (3)>]>

In [4]: bp = BlogPost(title='my blog', body='''
   ...: my 1st blog...
   ...: yoooo!''',
   ...: timestamp=datetime.now())

In [5]: bp
Out[5]: <BlogPost: BlogPost object (None)>

In [6]: bp.save()

In [7]: BlogPost.objects.count()
Out[7]: 4


In [9]: bp = BlogPost.objects.all()[0]


In [11]: print(bp.title)
test shell


In [13]: print(bp.body)

my 1st blog post...
yo!

In [14]: bp.timestamp.ctime()
Out[14]: 'Sun Mar 11 08:13:31 2018'
```
前两行命令导入相应的对象，第3步查询数据库中BlogPost对象，第4步是实例化一个BlogPost对象来向数据库中添加BlogPost对象，向其中传入对应属性的值(title、body和timestamp)。创建完对象后，需要通过BlogPost.save()方法将其写入到数据库中。完成创建和写入后，使用BlogPost.objects.count()方法确认数据库中对象的个数。然后获取BlogPost对象列表的第一个元素并获取对应属性的值。
**在使用时间生成器datetime.now()之前先在setting.py文件中设置USE_TZ = False以使用当前时区**
### Django管理应用
admin应用让开发者在完成完整的UI之前验证处理数据的代码。
#### 设置admin
在 *setting.py*的`INSTALLED_APP`中添加`'django.contrib.admin',`，然后运行`python3 ./manage.py makemigrations`和`python3 ./manage.py migrate`两条命令来创建其对应的表。在admin设置完之后于 *urls.py*中设置url路径：
```
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```
最后应用程序需要告诉Django哪个模型需要在admin页面中显示并编辑，这时候就需要在应用的 *admin.py*中注册BlogPost：
```
from django.contrib import admin
from blog import models
# Register your models here.
admin.site.register(models.BlogPost)
```
#### 使用admin
使用命令`python3 ./manage.py runserver`启动服务，然后在浏览器中输入 *http://localhost:8000/admin* 访问admin页面。在访问之前使用`python3 manage.py createsuperuser`创建的超级用户的用户名和密码用于登录管理页面。
