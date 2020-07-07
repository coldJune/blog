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