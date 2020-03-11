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

# 多表查询
* 笛卡尔积($m*n$)
* 等值链接(`join...on...`，`inner join...on...`,`,`)，可以控制笛卡尔积的产生
* 不等值连接(`join...between...and...`)
* 自然连接：基于两个表中名称相同的所有列；两个表相同的列数据类型不同无法成立；两个表在所有匹配列中具有相等值的行
* 自连接：条件出现在同一个表中
* 外连接：左连接(`left [out] join`)左表为驱动表，右表为匹配表，右连接(`right [out] join`)右表为驱动表，左表为匹配表，而谁为驱动表就显示谁的所有数据，所以左连接显示左表的全部数据，右连接显示右表的全部数据；全连接(`full [out] join`)两侧表都为驱动表，所以显示两侧表的全部数据

# 子查询
* 单行子查询
* 多行子查询(`IN`,`ANY`,`ALL`,`SOME`)
* 子查询会出现在`where`后、`where`和`from`之间、`select`和`from`之间

# 集合操作
* 并集(`UNION`、`UNION ALL`)：`UNINO`合并输出会去重，`UNION ALL`不去重
* 交集(`INTERSECT`)：只显示两个或多个结果集中的重复部分
* 差集(`MINUS`)：减去两个结果集中相同的部分，剩余显示被减结果集中的数据

# DML
* CTAS语句：
  * `CREATE TABLE XXX AS  SELECT * FROM 表名 WHERE 1=1;`复制表结构和数据
  * `CREATE TABLE XXX AS  SELECT * FROM 表名 WHERE 1=2;`复制表结构
* `DELETE`是DML，删除表中所有内容，是逻辑删除，将表中所有数据进行标记，标记为不可用；`TRUNCATE`是DDL，删除表中所有内容，是物理删除，彻底删除表中的数据，释放表空间
* 提交方式分为显示提交和隐式提交，显示提交触发`commit`或`rollback`，使用DDL语句隐式触发`commit`;
* `savepoint`设置回滚点
* `for update`锁定选定的行

# DDL
* `LONG`类型不建议使用