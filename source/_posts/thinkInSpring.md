---
title: springInAction
date: 2020-07-05 17:26:15
categories: 架构
copyright: true
tags:
    - Spring
    - Java
description:  Spring相关知识点总结
---

# [IoC](https://en.wikipedia.org/wiki/Inversion_of_control)
## 实现策略
* 依赖查找
* 依赖注入

### 依赖查找
* 服务定位模式：通常通过JBNDI获取Java EE的组件
* 上下文依赖查询：例如Java Beans里面的beancontext既能传输bean也能管理bean的层次性
* 使用模版方法：例如JdbcTemplate中的Statement回调，它让使用者不用关心callback从何处来，从而达到控制反转的效果
* 策略模式(?)

### 依赖注入
* 构造器注入
* 参数注入
* setter注入
* 接口注入

### 优劣对比

|类型|依赖处理|实现便利性|代码侵入性|API依赖性|可读性|
|:--:|:--:|:--:|:--:|:--:|:--:|
|依赖查找|主动获取(拉)|相对繁琐|侵入业务逻辑|依赖容器API|良好|
|依赖注入|被动提供(推)|相对便利|低侵入性|不依赖容器API|一般|

### 构造器注入和Setter注入
#### 构造器注入
##### 优点
* 保证组件不可变
* 管理的对象状态一致
* 确保需要的依赖不为空
* 减少代码

##### 缺点
* 参数过多使代码可读性不好(重构代码，别把太多的功能放在同一个类中)
* 构造器参数无名称提供给外部进行内省，即可读性不高
* 对IDE的支持比Setter少
* 过长的参数列表和过于臃肿的构造方法会变得不可控
* 对可选属性提供更少的支持
* 单元测试更加困难

#### setter注入
##### 优点
* 让对象变得更可配
* JavaBean属性对IDE的支持比较好
* JavaBean属性是一个自文档的，即不需要多余的文档就能明白这个属性的意思
* 在必要时可以使用JavaBean property-editor进行类型转换
* 大量已经存在的JavaBean可以继续使用而不用进行修改
* 能够覆盖初始值，意味着不是所有属性都需要在运行时马上赋值

##### 缺点
* 注入对象可能为空
* setter的顺序没法确定
* 不是所有setter都必须要调用的


## IoC容器的职责
### 依赖处理
> 解决依赖的来源以及怎么把它返回给客户端，同时处理从来到区的中间缓解的相关转化工作

* 依赖查找：主动的方式
* 依赖注入：有主动但大多数是容器完成

### 生命周期管理
* 容器：启动、暂停、回滚等容器的生命周期
* 托管的资源(Java Beans或其他资源)：比如POJO或事件的监听器

### 配置
* 容器：控制容器的行为
* 外部化配置：属性配置、XML配置等
* 托管的资源(Java Beans或其他资源)：Bean、外部容器、线程池等

## IoC容器的实现
### Java EE
* Java Beans
* Java ServiceLoader SPI
* JNDI (Java Naming and Directory Interface)

### Java EE
* EJB(Enterprise Java Beans)
* Servlet

### 开源
* ~~[Apache Avalon](http://avalon.apache.org/closed.html)~~
* [PicoContainer](http://picocontainer.com/)
* [Google Guice](https://github.com/google/guice)
* [Spring Framework](https://spring.io/projects/spring-framework)

## [Java Beans](https://github.com/coldJune/spring/tree/master/springInAction/JavaBeansDemo)作为IoC容器
Java beans 是一种综合需求的基础，它包含 Bean 自省（Bean 内部描述），Bean 时间，Bean 的内容修改（编辑）等等，并且由 BeanContext 统一托管 Bean 示例，简单地说，Spring 有部分思想是借鉴了 Java Beans

## 特性
* 依赖查找
* 生命周期管理
* 配置元信息
* 事件
* 自定义（方法、属性、类型转换）
* 资源管理
* 持久化

### Spring中的使用
`PropetyDescriptor`在Spring中应用广泛,`PropetyDescriptor`允许添加属性编辑器`PropertyEditor`，在Spring3.0之前大量的实践是基于`PropertyEditorSupport`来进行操作，既能满足元数据或元信息编程，也是一些类型配置和转换的根据

## 轻量级IoC容器
### 具备特征
* 能管理应用代码的运行(即生命周期)
* 能够快速启动(能否快速其实和依赖资源的多少有关)
* 不需要特殊配置
* 能达到轻量级的内存占用和最小化的API依赖
* 提供管控渠道，可以管理和部署一些细粒度的对象或粗粒度的组件

### 好处
* 释放掉一些聚式或者单体容器
* 最大化代码复用
* 能更大程度地面向对象
* 更大化地产品化(即关注效率)
* 更好的可测试性

## Spring作为IoC容器的优势
* 典型的IoC管理：依赖查找和依赖注入
* AOP抽象
* 事务抽象
* 事件机制：基于Java标准事件`EventObject`类和`EventListener`接口
* SPI扩展：`BeanPostProcessor`Bean的扩展、`BeanFactoryPostProcessor`IoC容器(BeanFactory)的扩展、工厂扩展机制(Spring Factories)
* 强大的第三方整合：ORM、JDBC、Spring Data等
* 易测试性：spring test
* 更好的面向对象

# Spring IoC
## [Spring IoC的依赖查找](https://github.com/coldJune/spring/tree/master/thinkInSpring/ioc-container-overview/src/main/java/com/jun/ioc/overview/dependency/lookup)
* 根据Bean名称查找
  * 实时查找
  * 延迟查找
* 根据Bean类型查找
  * 单个Bean对象
  * 集合Bean对象
* 根据Bean名称+类型查找
* 根据Java注解查找
  * 单个Bean对象d
  * 集合Bean对象

## [Spring IoC依赖注入](https://github.com/coldJune/spring/tree/master/thinkInSpring/ioc-container-overview/src/main/java/com/jun/ioc/overview/dependency/injection)
* 根据Bean名称注入
* 根据Bean类型注入
  * 单个Bean对象
  * 集合Bean对象
* 注入容器内建Bean对象
* 注入非Bean对象
* 注入类型
  * 实时注入
  * 延迟注入

## Spring IoC依赖来源
* 自定义Bean
* 容器内建Bean对象
* 容器内建依赖


## Spring IoC配置元信息
* Bean定义配置
  * 基于XML配置
  * 基于Properties配置
  * 基于Java注解
  * 基于Java API
* IoC容器配置
  * 基于XML配置
  * 基于Java注解
  * 基于Java API
* 外部化属性配置
  * 基于Java注解

## Spring IoC容器
>BeanFacotry和ApplicationContext谁才是IoC容器

* `BeanFactory`是底层的IoC容器，`ApplicationContext`在之上增加了一些特性，`ApplicationContext`是`BeanFactory`的一个超集，它在底层是通过组合引入了一个`beanFactory`，当要获取`beanFactory`时需要调`getBeanFactory`方法获取真正的`beanFactory`

## Spring应用上下文
ApplicationContext除了IoC容器还提供：
* 面向切面(AOP)
* 配置元信息(Configuration Metadata)
* 资源管理(Resource)
* 事件(Events)
* 国际化(i18n)
* 注解(Annotation)
* Environment抽象(Environment Abstraction)

# Spring Bean
## Spring Bean 的定义
BeanDefinition是Spring Framework中定义Bean的配置元信息接口，包含：
* Bean的类名
* Bean行为配置元素，如作用域、自动绑定的模式、生命周期回调
* 其他Bean引用(合作者*Collaborators*或者依赖*Dependencies*)
* 配置设置，比如Bean属性(Properties)

## BeanDefinition元信息
### BeanDefinition元信息

|属性(Property)|说明|
|:--:|:--:|
|Class|Bean全类名，必须是具体类，不能是抽象类或接口|
|Name|Bean的名称或者ID|
|Scope|Bean的作用域(如:sigleton、prototype等)|
|Constructor arguments|Bean构造器参数(用于依赖注入)|
|Properties|Bean属性设计(用于依赖注入)|
|Autowriting mode|Bean自动绑定模式(如:通过名称byName)|
|Lazy initialization mode|Bean延迟初始化模式(延迟和非延迟)|
|Initialization method|Bean初始化回调方法名称|
|Destruction method|Bean销毁回调方法名称|

### BeanDefintion构建
* 通过`BeanDefinitionBuilder`
* 通过`AbstractBeanDefinition`以及派生类

## Bean命名
* 在Bean**所在的容器**必须是唯一的
* xml中使用`id`或`name`属性指定
* 为空容器会自动生成

### Bean名称生成器(BeanNameGenerator)
* `DefaultBeanNameGenerator`:默认通用的BeanNameGenerator实现
* `AnnotationBeanNameGenerator`:基于注解扫描的BeanNameGenerator实现

### Bean别名(Alias)
#### 价值
* 复用现有的BeanDefinition
* 更具有场景化的命名方法

## 注册Spring Bean
* XMl配置元信息
`<bean name="" .../>`
* Java注解配置元信息
  * `@Bean`
  * `@Component`
  * `@Import`
* Java API配置元信息
  * 命名方式:`BeanDefinitionRegistry#registerBeanDefinition(String,BeanDefinition)`
  * 非命名方式:`BeanDefinitionReaderUtils#registerWithGeneratedName(AbstractBeanDefinition,BeanDefinitionRegistry)`
  * 配置类信息:`AnnotatedBeanDefinitionReader#register(Class)`

## 实例化Spring Bean
### 常规方式
* 通过构造器（配置元信息：XML、Java注解和Java API）
* 通过静态工厂方法（配置元信息：XML和Java API）
* 通过Bean工厂方法（配置元信息：XML和Java API）
* 通过`FactoryBean`（配置元信息：XML、Java注解和Java API）

### 特殊方式
* 通过`ServiceLoaderFactoryBean`（配置元信息：XML、Java注解和Java API）
* 通过`AutowireCapableBeanFactory#createBean(java.lang.Class,int,boolean)`
* 通过`BeanDefinitionRegistry#registerBeanDefinition(String,BeanDefinition)`

## Bean初始化
* `@PostContruct`标注方法
* 实现`InitializingBean`接口的`afterPropertiesSet()`
* 自定义初始化方法
  * XML配置:`<bean init-method="init" .../>`
  * Java注解:`@Bean(initMethod="init")`
  * Java API:`AbstractBeanDefinition#setInitMethodName(String)`
> 同时定义，优先级顺序为`@PostContruct`->`afterPropertiesSet`->自定义初始化方法

### 延迟初始化
* XML配置:`<bean lazy-init="true" ...>`
* Java注解:`@Lazy(value=true)`
>延迟初始化和非延迟初始化的区别在于延迟初始化使用时才进行初始化，非延迟初始化在容器启动时就进行初始化

## Bean 销毁
* `@PreDestroy`标注方法
* 实现`DisposableBean`接口的`destroy()`方法
* 自定义销毁方法
  * XML配置:`<bean destroy-method="destroy" .../>`
  * Java注解:`@Bean(destroyMethod="destroy")`
  * Java API:`AbstractBeanDefinition#setDestroyMethodName(String)`
> 同时定义，优先级顺序为`@PreDestroy`->`destroy()`->自定义初始化方法

##  Bean的垃圾回收(GC)
1. 关闭Spring容器(上下文)
2. 执行GC
3. Spring Bean覆盖的`finalize()`被回调

# Spring IoC依赖查找
## 单一类型依赖查找(`BeanFactory`)
### 根据Bean名称查找
* `getBean(String)`
* Spring 2.5覆盖默认参数：`getBean(String, Object ...)`

### 根据Bean类型查找
#### Bean实时查找
* Spring 3.0 `getBean(Class)`
* Spring 4.1 覆盖默认参数：`getBean(Class, Object...)`

#### Spring 5.1 Bean 延迟查找
* `getBeanPeovider(Class)`
* `getBeanProvider(ResolvableType)`

### 根据Bean名称+类型查找
* `getBean(String,Class)`

## 集合类型依赖查找(`ListableBeanFactory`)
### 根据Bean类型查找
* 获取同类型Bean名称列表
  * `getBeanNamesForType(Class)`
  * Spring 4.2 `getBeanNamesForType(ResolvableType)`
* 获取同类型Bean实例列表
  * `getBeansOfType(Class)`以及重载方法

### 通过注解类型查找
* Spring 3.0 获取标注类型Bean名称列表
  * `getBeanNamesForAnnotation(Class<? extends Annotation>)`
* Spring 3.0 获取标注类型Bean实例列表
  * `getBeanWithAnnotation(Class <? extends Annotation>)`
* Spring 3.0 获取指定名称 + 标注类型Bean实例
  * `findAnnotationOnBean(String, Class<? extends Annotation>)`

## 层次性依赖查找(`HierarchicalBeanFactory`)
* 双亲 BeanFactory：`getParentBeanFactory()`

### 层次性查找
* 根据Bean名称查找
  * 基于`containsLocalBean`方法实现
* 根据Bean类型查找实例列表
  * 单一类型：`BeanFactoryUtils#beanOfType`
  * 集合类型：`BeanFactoryUtils#beansOfTypeIncludingAncestors`
* 根据Java注解查找名称列表
  * `BeanFactoryUtils#beanNamesForTypeIncludingAncestors`

## 延迟依赖查找
* Bean 延迟依赖查找接口
  * `org.springframework.beans.factory.ObjectFactory`
  * `org.springframework.beans.factory.ObjectProvider`
    * Spring 5 对Java 8特性扩展
      * `getIfAvailable(Supplier)`
      * `ifAvailable(Consumer)`
    * Stream扩展-`stream()`

## 安全依赖查找

|依赖查找类型|代表实现|是否安全|
|:--:|:--:|:--:|
|单一类型查找|BeanFactory#getBean|否|
|单一类型查找|ObjectFactory#getObjedc|否|
|单一类型查找|ObjectProvider#getIfAvailable|是|
||||
|集合类型查找|ListableBeanFactory#getBeansOfType|是|
|集合类型查找| ObjectProvider#Stream|是|

**层次性依赖查找的安全性取决于其扩展的单一或集合类型的BeanFactory接口**

## 内建可查找的依赖
### `AbstractApplicationContext`内建可查找的依赖

|Bean名称|Bean实例|使用场景|
|:--:|:--:|:--:|
|environment|Environment 对象|外部化配置以及Profiles| 
|systemProperties|java.util.Properties 对象|Java系统属性|
|systemEnvironment|java.util.Map 对象|操作系统环境变量|
|messageSource|MessageSource 对象|国际化文案|
|lifecycleProcessor|LifecycleProcessor 对象|Lifecycle Bean 处理器|
|applicationEventMulticaster|ApplicationEventMulticaster 对象|Spring事件广播器| 

### 注解驱动Spring应用上下文内建可查找的依赖

|Bean名称|Bean实例|使用场景|
|:--:|:--:|:--:|
|org.springframework.context.annotation.internalConfigurationAnnotationProcessor|ConfigurationClassPostProcessor 对象|处理Spring配置类|
|org.springframework.context.annotation.internalAutowiredAnnotationProcessor|AutowiredAnnotationBeanPostProcessor 对象|处理@Autowired和@Value注解|
|org.springframework.context.annotation.internalCommonAnnotationProcessor|CommonAnnotationBeanPostProcessor 对象|(条件激活)处理JSR-250注解，如@PostConstruct等|
|org.springframework.context.annotation.internalEventListenerAnnotationProcessor|EventListenerMethodProcessor 对象|处理标注@EventListener的Spring事件监听方法|
|org.springframework.context.annotation.internalEventListenerFactory|DefaultEventListenerFactory 对象|@EventListener的Spring事件监听方法适配为ApplicationListener|
|org.springframework.context.annotation.internalPersistenceAnnotationProcessor|PersistenceAnnotationBeanPostProcessor 对象|(条件激活)处理JPA注解|

## 依赖查找中的经典异常
|异常类型|触发条件(举例)|场景举例|
|:--:|:--:|:--:|
|NoSuchBeanDefinitionException|当查找Bean不存在与IoC容器时|BeanFactory#getBean,ObjectFactory#getObject|
|NoUniqueBeanDefinitionException|类型依赖查找时，IoC容器存在多个Bean实例|BeanFactory#getBean(Class)|
|BeanInstantiationException|当Bean所对应的类型非具体类时|BeanFactory#getBean|
|BeanCreationException|当Bean初始化过程中|Bean初始化方法执行异常时|
|BeanDefinitionStoreException|当BeanDefinition配置元信息非法时|XML配置资源无法打开时|

# Spring IoC依赖注入
## 依赖注入的模式和类型
### 模式
* 手动模式 - 配置或者编程的方式，提前安排注入规则
  * XML资源配置元信息
  * Java注解配置元信息
  * API配置元信息
* 自动模式 - 实现方提供依赖自动关联的方式，按照内建的注入规则
  * Autowiring(自动绑定)

### 类型
|依赖注入类型|配置元数据举例|
|:--:|:--:|
|Setter方法|<property name="user" ref="userBean"/>|
|构造器|<construct-arg name="user" ref="userBean"/>|
|字段|@Autowired User user;|
|方法|@Autowired public void user(User user){...}|
|接口回调|class MyBean implements BeanFactoryAware{...}|

## 自动绑定(Autowiring)
### 模式

|模式|说明|
|:--:|:--:|
|no|默认值，未激活Autowiring,需要手动指定依赖注入对象|
|byName|根据被注入属性的名称作为Bean名称进行依赖查找，并将对象设置到该属性|
|byType|根据被注入属性的类型作为依赖类型进行查找，并将对象设置到该属性|
|constructor|特殊byType类型，用于构造器参数|
参考枚举:`org.springframework.beans.factory.annotation.Autowire`

### 限制和不足
1. 精确依赖(`property`和`construct-arg`)会覆盖Autowiring
2. 不能绑定简单类型，如原生类型、`String`和`Classes`
3. 是一种猜测性的，缺乏精确性
4. 绑定的信息无法在一些工具上进行呈现
5. 上下文存在多个Bean，会产生歧义

## Setter方法注入
### 实现方法
* 手动模式
  * XML 资源配置元信息
  * Java 注解配置元信息
  * API 配置元信息
* 自动模式
  * byName
  * byType