---
title: ocp_sql
date: 2020-03-09 10:19:49
categories: Oracle
copyright: true
tags:
    - OCP
    - Oracle
description: 学习Oracle OCP课程
---

# Select
## Select的功能
* `SELECT`定义查询的列
* `SELECT *`查询所有数据
* `SELECT COl`查询指定列名
* 先用聚合函数`count`查看表的数据量，避免直接`select *`数据过多卡死数据库
* SQL不区分大小写，字面量区分
* SQL可被拆分成多行
* 关键字不能分离
* 表达式尽量写在当行(非必须，但是为了保持好的风格)
* SQL developer多条语句使用分号结尾
* sqlplus必须使用分号
## Select算数运算符计算与空值
* `+`、`-`、`*`、`\`
* `SELECT salary+100`
* `NULL`进行运算为`NULL`
* `NVL`函数把空值替换为另一个值
## 列别名
* 改变列头
* 用在表达式
* 表达式后面使用`AS`定义(`SELECT SALARY*12 AS YEARSALARY FROM EMPLOYEE`)
* 别名区分大小写或有空格需要加上双引号(`SELECT SALARY*12 AS "YEAR SALARY" FROM EMPLOYEE`)
## 字符串拼接
* `||`连接字符串和列
## 重复列
* `distinct`去重(`SELECT DISTINCT MIN_SALARY FROM JOBS`)

# where和排序
* `where`限制查询条件
* `to_date`函数定义时间格式
* `BETWEEN... AND ...`是闭区间
* `LIKE`大小写敏感，使用`%`和`_`占位模糊匹配
* `ORDER BY`排序，默认升序排序`ASC`，降序排序关键字为`DESC`
* 可以使用排序进行简单聚类
* 12c新特性row limiting clause限制行输出(`SELECT * FROM JOBS FETCH FIRST 5 ROWS ONLY;`输出前5行，`SELECT * FROM JOBS OFFSET 5 ROWS FETCH NEXT 5 ROWS ONLY;`获取第5行后的5行,`OFFSET`表示偏移量)
* `&`表示替换变量(`SELECT * FROM EMPLOYEES WHERE EMPLOYEE_ID=&employee_num`，会提示输入`&employee_num`的值，也可以使用`DEFINE &employee_num=`提前定义变量值)

# 函数
* `nvl(exp1, exp2)`，`exp1`为空则返回为`exp2`
* `nvl2(exp1, exp2, exp3)`，`exp1`不为空返回`exp2`，否则返回`exp3`
* `nullif(exp1, exp2)`，`exp1`和`exp2`相等则返回空，否则返回`exp1`

# 分组与聚合
* `count`不将`NULL`计算在内(聚合函数都不对`NULL`进行处理，如果有`NULL`值则数据总量缩小)
* `listagg`先分组后排序
* `stddev` 标准差
* `variance`方差
* `group by`聚合条件和查询字段相同，分组之前考虑好以哪个字段进行分组
* `having` 对分组以后的结果进行条件筛选，先分组产生结果后再进行筛选