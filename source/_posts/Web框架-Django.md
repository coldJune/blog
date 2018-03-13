---
title: 'Web框架:Django'
date: 2018-03-12 16:46:21
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
1. **在Django中创建项目**
Django自带有一个名为`django-admin.py`/`django-admin.exe`的工具，它可以简.
化任务。在POSIX平台上，一般在`/usr/local/bin`、`/usr/bin`这样的目录中。使用Windows系统会安装在Python包下的Scripts目录下，如`E:\Python\Python36\Scripts`。两种系统都应该确保文件位于PATH环境变量中。
在项目文件加下执行命令创建项目:
```
django-admin.py startproject mysite
```
2. **Django项目文件**

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
与项目类似，应用也是一个Python包。本地应用的URLconf需要手动创建，接着使用URLconf里的include()指令将请求分配给应用的URLconf。为了让Django知道这个新应用是项目的一部分，需要编辑 *settings.py*，将应用名称(**blog**)添加到元组的末尾。Django使用 **INSTALLED_APPS** 来配置系统的不同部分，包括自动管理应用程序和测试框架。
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
*models.py* 将定义博客的数据结构，首先创建一个基本类型。数据模型表示将会存储在数据库每条记录的数据类型。Django提供了许多[字段类型](https://docs.djangoproject.com/en/2.0/ref/models/fields/)，用来将数据映射到应用中。
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
在项目的*setting.py*文件中设置数据库。关于数据库，有6个相关设置(有时只需要两个):**ENGINE**、**NAME**、**HOST**、**PORT**、**USER**、**PASSWORD**。只需要在相关设置选项后面添上需要让Django使用的数据库服务器中合适的值即可。
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
* 使用SQLite
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
使用 *makemigrations* 参数创建映射文件，当执行命令时Django会查找INSTALLED_APPS中列出的应用的models.py文件。对于每个找到的模型，都会创建一个映射表。
```
python3 ./manage.py makemigrations
```
使用 *migrate* 映射到数据库
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
[Django shell](https://docs.djangoproject.com/en/dev/intro/tutorial01/#playing-with-the-api)和标准的shell相比更专注于Django项目的环境，可以与视图函数和数据模型交互，这个shell会自动设置环境变量，包括sys.path，它可以访问Django与自己项目中的模块和包，否则需要手动配置。除了标准shell之外，还有其他的交互式解释器可供选择。Django更倾向于使用功能丰富的shell，如IPython和bpython，这些shell在普通的解释器基础上提供及其强大的功能。运行shell命令时，Django首先查找含有扩展功能的shell，如果没有回返回标准解释器。这里使用的是IPython。也可以使用 *-i* 来强制使用普通解释器。
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
设置时区:
```Python
LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

USE_TZ = False

```
### Django管理应用
admin应用让开发者在完成完整的UI之前验证处理数据的代码。
#### 设置admin
在 *setting.py* 的`INSTALLED_APP`中添加`'django.contrib.admin',`，然后运行`python3 ./manage.py makemigrations`和`python3 ./manage.py migrate`两条命令来创建其对应的表。在admin设置完之后于 *urls.py* 中设置url路径：
```Python
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```
最后应用程序需要告诉Django哪个模型需要在admin页面中显示并编辑，这时候就需要在应用的 *admin.py* 中注册BlogPost：
```Python
from django.contrib import admin
from blog import models
# Register your models here.
admin.site.register(models.BlogPost)
```
#### 使用admin
使用命令`python3 ./manage.py runserver`启动服务，然后在浏览器中输入 *http://localhost:8000/admin* 访问admin页面。在访问之前使用`python3 manage.py createsuperuser`创建的超级用户的用户名和密码用于登录管理页面。（账号：*root*，密码：*Aa123456*）
为了更好地显示博文列表，更新blog/admin.py文件，使用新的BlogPostAdmin类：
```Python
from django.contrib import admin
from blog import models
# Register your models here.


class BlogPostAdmin(admin.ModelAdmin):
    list_display = ('title', 'timestamp')


admin.site.register(models.BlogPost, BlogPostAdmin)

```
### 创建博客的用户界面
Django shell和admin是针对于开发者的工具，而现在需要构建用户的界面。Web页面应该有以下几个经典组建：
1. **模板**，用于显示通过Python类字典对象传入的信息
2. **视图函数**，用于执行针对请求的核心逻辑。视图会从数据库中获取信息，并格式化显示结果
3. **模式**，将传入的请求映射到对应的视图中，同时也可以将参数传递给视图

Django是自底向上处理请求，它首先查找匹配的URL模式，接着调用对应的视图函数，最后将渲染好的数据通过模板展现给用户。构建应用可以按照如下顺序：
1. 因为需要一些可观察对象，所以先创建基本的模板
2. 设计一个简单的URL模式，让Django可以立刻访问应用
3. 开发出一个视图函数原型，然后在此基础上迭代开发
在构建应用过程中模板和URL模式不会发生太大的变化，而应用的核心是视图。这非常符合 *测试驱动模型(TDD)* 的开发模式。

#### [创建模板](https://docs.djangoproject.com/en/2.0/topics/templates/#tags)
* *变量标签*
**变量标签** 是由 *花括号({{……}})* 括起来的内容，花括号内用于显示对象的内容。在变量标签中，可以使用Python风格的 *点分割标识* 访问这些变量的属性。这些值可以是纯数据，也可以是可调用对象，如果是后者，会自动调用这些对象而无需添加圆括号"()"来表示这个函数或方法可调用。

* *过滤器*
**过滤器** 是在变量标签中使用的特殊函数，它能在标签中立即对变量进行处理。方法是在变量右边插入一个 *管道符号("|")*，接着跟上过滤器名称。`<h2> { { post.title | title } } </h2>`

* *上下文*
**上下文** 是一种特殊的Python字典，是传递给模板的变量。假设通过上下文传入的BlogPost对象称为"post"。通过上下文传入所有的博文，这样可以通过循环显示所有文章。

* *块标签*
**块标签** 通过花括号和百分号来表示：&#123;%…%&#125;，它们用于向HTML模版中插入如循环或判断这样的逻辑。

将HTML模版代码保存到一个简单的模版文件中，命名为archive.html，放置在应用文件夹下的 **templates** 目录下，模版名称任取，但模版目录一定是 *templates*
```Html
{%for post in posts%}
    <h2>{{post.title}}</h2>
    <h2>{{post.timestamp}}</h2>
    <h2>{{post.body}}</h2>
{% endfor%}
```
#### 创建URL模式
* 项目的URLconf
服务器通过WSGI的功能，最终会将请求传递给Django。接受请求的类型(GET、POST等)和路径(URL中除了协议、主机、端口之外的内容)并传递到项目的URLconf文件(mysite/urls.py)。为了符合代码重用、DRY、在一处调试相同的代码等准则，需要应用能负责自己的URL。在项目的urls.py(这里时mysite/urls.py)中添加url配置项，让其指向应用的URLconf。
```Python
from django.contrib import admin
from django.urls import path
from django.urls import include
urlpatterns = [
    path('admin/', admin.site.urls),
    # include函数将动作推迟到其他URLconf
    # 这里将以blog/开通的请求缓存起来，并传递给mysite/blog/urls.py
    path('blog/', include('blog.urls'))
]
```
*include()* 会移除当前的URL路径头，路径中剩下的部分传递给下游URLconf中的path()函数。（*当输入'http://localhost:8080/blog/foo/bar' 这个URL时，项目的URLconf接收到的是blog/foo/bar，匹配blog找到一个include()函数，然后将foo/bar传递给mysite/blog/urls.py*）。上述代码中使用include()和未使用include()的区别在于使用include()传递的是 **字符串**，未使用include传递的是 **对象**。

* 应用的URLconf
在项目的URLconf中通过include()包含blog.urls，让匹配blog应用的URL将剩余的部分传递到blog应用中处理。在mysite/blog/urls.py(没有就创建),添加以下代码：
```Python
from django.urls import *
import blog.views
urlpatterns = [
    # 第一个参数是路径，第二个参数是视图函数，在调用到这个URL时用于处理信息
    path('', blog.views.archive)
]
```
请求URL的头部分(blog/)匹配到的是根URLconf已经被去除。添加新的视图在列表中添加一行代码即可。
#### 创建视图函数
一个简单的视图函数会从数据库获取所有博文，并使用模板显示给用户：
1. 向数据库查询所有博客条目
2. 载入模板文件
3. 为模板创建上下文字典
4. 将模板渲染到HTML中
5. 通过HTTP响应返回HTML
在应用的views.py中添加如下代码:
```Python
from django.shortcuts import render
from blog.models import BlogPost
from django.template import loader, Context
from django.shortcuts import render_to_response
# Create your views here.


def archive(request):
    posts = BlogPost.objects.all()
    return render_to_response('archive.html', {'posts': posts})

```
### 改进输出
现在得到了一个可以工作的应用，有了可以工作的简单博客，可以响应客户端的请求，从数据库提取信息，向用户显示博文。现在更改查询方式，让博文按时间逆序显示，并且限制每页显示的数目。
> BlogPOST是数据模型类。Objects属性是模型的Manager类，其中含有all()方法来获取QuerySet。QuerySet执行“惰性迭代”，在求值时才会真正查询数据库。

实现排序只需调用order_by()方法时提供一个排序参数即可(views.py)：
```Python
def archive(request):
    # 在timestamp前面加上减号(-)指定按时间逆序排列。正常的升序只需要移除减号
    posts = BlogPost.objects.all().order_by('-timestamp')
    return render_to_response('archive.html', {'posts': posts})
```
为了测试限制显示数目，先启动Django shell添加数据：
```
python ./manage.py shell
Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:54:40) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> from datetime import datetime
>>> from blog.models import BlogPost
>>> for i in range(10):
...     bp = BlogPost(title='post $%d' % i ,body='body of post $%d' %d, timestamp=datetime.now())
...     bp.save()
...
```
然后使用切片的方式获取最新的10篇(views.py)：
```Python
def archive(request):
    # 在timestamp前面加上减号(-)指定按时间逆序排列。正常的升序只需要移除减号
    posts = BlogPost.objects.all().order_by('-timestamp')[:10]
    return render_to_response('archive.html', {'posts': posts})
```

* 设置模型的默认排序方式

如果在模型中设置首选的排序方式，其他基于Django的应用或访问这个数据的项目也会使用这个顺序。为了给模型设置默认顺序，需要创建一个名为 **Meta** 的内部类，在其中设置一个名为 **ordering** 的属性(models.py):
```Python
class BlogPost(models.Model):
    """
    django.db.models.Model的子类Model是Django中用于数据模型的标准基类。
    BlogPost中的字段像普通类属性那样定义，
    每个都是特定字段类的实例，每个实例对应数据库中的一条记录。
    """
    title = models.CharField(max_length=150)
    body = models.TextField()
    timestamp = models.DateTimeField()

    class Meta:
        ordering = ('-timestamp',)
```
取消视图函数中的排序(views.py):
```Python
def archive(request):
    # 在timestamp前面加上减号(-)指定按时间逆序排列。正常的升序只需要移除减号
    posts = BlogPost.objects.all()[:10]
    return render_to_response('archive.html', {'posts': posts})
```

### 处理用户输入
1. 添加一个HTML表单，让用户可以输入数据(archive.html),为了防止
```Html
<form action="/blog/create/" method="post">
    Title:
    <input type="text" name="title"><br>
    Body:
    <textarea name="body" rows="3" cols="60"></textarea><br>
    <input type="submit">
</form>
<hr>
{%for post in posts%}
    <h2>{{post.title}}</h2>
    <p>{{post.timestamp}}</p>
    <p>{{post.body}}</p>
<hr>
{% endfor %}

```

2. 插入(URL，视图)这样的URLConf项
使用前面的HTML，需要用到/blog/create/的路径，所以需要将其关联到一个视图函数中，该函数用于把内容保存到数据库中，这个函数命名为create_blogpost()，在应用的urls.py中添加：
```Python
from django.urls import *
import blog.views
urlpatterns = [
    # 第一个参数是路径，第二个参数是视图函数，在调用到这个URL时用于处理信息
    path('', blog.views.archive),
    path(r'create/', blog.views.create_blogpost)
]
```

3. 创建视图来处理用户输入
在应用的views.py中添加上面定义的处理方法
```Python
def create_blogpost(request):
    if request.method == 'POST':
        # 检查POST请求
        # 创建新的BlogPost项，获取表单数据，并用当前时间建立时间戳。
        BlogPost(
            title=request.POST.get('title'),
            body=request.POST.get('body'),
            timestamp=datetime.now()
        ).save()
    # 重定向会/blog
    return HttpResponseRedirect('/blog')
```

* 在完成上面的步骤之后，会发现创建表单的调用会被拦截报403的错误。这是因为Django有数据保留特性，不允许不安全的POST通过 *跨站点请求伪造（Cross-site Request Forgery,CSRF）* 来进行攻击。需要在HTML表单添加CSRF标记(&#123;% csrf_token %&#125;):
```Html
<form action="/blog/create/" method="post">{%csrf_token%}
    Title:
    <input type="text" name="title"><br>
    Body:
    <textarea name="body" rows="3" cols="60"></textarea><br>
    <input type="submit">
</form>
<hr>
    {%for post in posts%}

    <h2>{{post.title}}</h2>
    <p>{{post.timestamp}}</p>
    <p>{{post.body}}</p>
<hr>
{% endfor %}
```
通过模板发送向这些标记请求的上下文实例，这里将`archive()`方法调用的`render_to_response()`改为`render`:
```Python
def archive(request):
    # 在timestamp前面加上减号(-)指定按时间逆序排列。正常的升序只需要移除减号
    posts = BlogPost.objects.all()[:10]
    return render(request, 'archive.html', {'posts': posts})
```

### 表单和模型表单
* 如果表单字段完全匹配一个数据模型，则通过Django ModelForm能更好的完成任务(models.py):
```Python
class BlogPostForm(forms.ModelForm):
    class Meta:
        # 定义一个Meta类，他表示表单基于哪个数据模型。当生成HTML表单时，会含有对应数据模型中的所有属性字段。
        # 不信赖用户输入正确的时间戳可以通过添加exclude属性来设置。
        model = BlogPost
        exclude = ('timestamp',)
```
* 使用ModelForm来生成HTML表单(archive.html):
```Python
<form action="/blog/create/" method="post">{%csrf_token%}
  <table>{{form}}</table>
    <input type="submit">
</form>
<hr>
    {%for post in posts%}

    <h2>{{post.title}}</h2>
    <p>{{post.timestamp}}</p>
    <p>{{post.body}}</p>
<hr>
{% endfor %}
```
* 因为数据已经存在于数据模型中，便不用去通过请求获取单个字段，而由于timestamp不能从表单获取，所以修改后的views.py中`create_blogpost()`方法如下:
```Python
def create_blogpost(request):
    if request.method == 'POST':
        # 检查POST请求
        # 创建新的BlogPost项，获取表单数据，并用当前时间建立时间戳。
        # BlogPost(
        #     title=request.POST.get('title'),
        #     body=request.POST.get('body'),
        #     timestamp=datetime.now()
        # ).save()
        form = BlogPostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.timestamp = datetime.now()
            post.save()
    # 重定向会/blog
    return HttpResponseRedirect('/blog')
```

### 添加测试
Django通过扩展Python自带的单元测试模块来提供测试功能。Django还可以测试文档字符串(即docstring)，这称为 *文档测试(doctest)*
>应用的tests.py

```Python
from django.test import TestCase
from datetime import datetime
from django.test.client import Client
from blog.models import BlogPost
# Create your tests here.


class BlogPostTest(TestCase):
    # 测试方法必须以“test_”开头，方法名后面的部分随意。
    def test_obj_create(self):
        # 这里仅仅通过测试确保对象成功创建，并验证标题内容
        BlogPost.objects.create(
            title='raw title', body='raw body', timestamp=datetime.now())
        # 如果两个参数相等则测试成功，否则该测试失败
        # 这里验证对象的数目和标题
        self.assertEqual(1, BlogPost.objects.count())
        self.assertEqual('raw title', BlogPost.objects.get(id=1).title)

    def test_home(self):
        # 在'/blog/'中调用应用的主页面，确保收到200这个HTTP返回码
        response = self.client.get('/blog/')
        self.assertIn(response.status_code, (200, ))

    def test_slash(self):
        # 测试确认重定向
        response = self.client.get('/')
        self.assertIn(response.status_code, (301, 302))

    def test_empty_create(self):
        # 测试'/blog/create/'生成的视图，测试在没有任何数据就错误地生成GET请求，
        # 代码应该忽略掉这个请求，然后重定向到'/blog'
        response = self.client.get('/blog/create/')
        self.assertIn(response.status_code, (301, 302))

    def test_post_create(self):
        # 模拟真实用户请求通过POST发送真实数据，创建博客项，让后将用户重定向到"/blog"
        response = self.client.post('/blog/create/', {
            'title': 'post title',
            'body': 'post body'
        })
        self.assertIn(response.status_code, (301, 302))
        self.assertEqual(1, BlogPost.objects.count())
        self.assertEqual('post title', BlogPost.objects.get(id=1).title)
```
[源代码](https://github.com/coldJune/Python/tree/master/mysite)
