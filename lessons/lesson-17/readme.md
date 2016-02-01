---
title: Databases
duration: "3:00"
creator:
    name: Arun Ahuja
    city: NYC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 
Week # | Lesson 17

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Understanding of the uses and differences of databases
- Accessing databases from Pandas

### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- There will be multiple ways to run the exercises:
    - Using Postgres Exercises
        - For this students should install Postgres (more details) below
        - If `brew` is installed, this should be as simple as `brew install postgres`
    - Setting up local Postgres
    - Using Wagon
        - For this setup, students should create an account at https://www.wagonhq.com/ and download the software

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- There will be multiple ways to run the exercises:
    - Using Postgres Exercises
        - For this students should install Postgres (more details) below
        - If `brew` is installed, this should be as simple as `brew install postgres`
        - You can setup a local demo of the postgres exercises as well from https://pgexercises.com/gettingstarted.html
    - Using Wagon
        - For this setup, you can should create an account at https://www.wagonhq.com/ and download the software


### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  | Lesson Objectives  |
| 35 min  | [Introduction](#introduction)   | Intro to Databases |
| 20 min  | [Demo](#demo1)  | Demo: Accessing Databases from Pandas |
| 60 min  | [Demo](#demo2)  | Demo/Codealong: SQL Syntax: SELECT, WHERE, GROUP BY, JOIN |
| 40 min  | [Independent Practice](#ind-practice-pandas)  | Pandas and SQL Exercises |
| 15 min  | [Demo](#demo3)  | Installing Postgres, CREATE TABLE |
| 5 min  | [Conclusion](#conclusion)  |   |

---
<a name="opening"></a>
## Opening (5 min)
Today's lesson will be on databases and the SQL query language. Databases are the standard solution for data storage and are much more robust than how we've stored data in this class so far - text and CSV files. Databases come in many flavors, but we will explore the most common, relational databases. Relational databases themselves come in different varieties, but almost all use the SQL language as a base for querying. Almost all analyses typically involve pulling data from a resource and in most industry settings that resource will be a database.

<a name="introduction"></a>
## Intro to Databases (35 mins)

Databases are computer systems that manage storage and querying of datasets. Databases provide a way to organize the data on disk and efficient methods to retrieve information. Typically, retrieval is performed using a query language, a mini programming language with a few basic operators for data transformation, the most common of which is **SQL**.

Databases, additionally, allow users to create rules that ensure proper data management and verification.

A _relational database_ is a database based on links between data entities or concepts. Typically, a relational database is organized into _tables_. Each table should correspond to one entity or concept. Each _table_ is similar to a single CSV file or Pandas dataframe.

For example, let's take a sample application like Twitter. Our two main entities are Users and Tweets. For each of these we would have a table.

A table is made up rows and columns, similar to an Pandas dataframe or Excels spreadsheet. A table also has a _schema_ which is a set of rules for what goes in each table, similar to the header in a CSV file. These specify what columns are contained in the table and what _type_ those columns are (text, integers, floats, etc.). The additional _type_ information make this constraint stronger than a CSV file. For this reason, and many others, databases allow for stronger consistency of the data and often are a better solution for data storage.

Each table typically has a _primary key_ column. This column is a unique value per row and serves as the identifier for the row.

A table can have many _foreign keys_ as well. A _foreign key_ is a column that contains values to link the table to the other tables. For example, the tweets table may have as columns:
    - tweet_id, the primary key tweet identifier
    - the tweet text
    - the user id of the member, a foreign key to the users table

These keys that link the table together define the relational database.

MySQL and Postgres are popular variants of relational databases and are widely used. Both of these are open-source so are available for free. Alternatively, many companies will have Oracle or Microsoft SQL databases. While these all offer many of the same features, and use SQL as a query language, the latter two offer some maintenance features that some large companies find useful. However, mysql and postgres are very much in use at large companies as well.

### Normalized vs Denormalized data

Once we start organizing our data into tables we start to separate into _normalized_ or _denormalized_ setups.

_Normalized_ structures have a single table per entity, and use many foreign keys or link table to connect the entities.

_Denormalized tables_ have fewer tables and may for example place all of the tweets and the information on the users in one table.

Each style has advantages and disadvantages. _Denormalized_ structure table, duplicates a lot of information. For example, in our combined tweets/users table, we may store the address of each user. Now instead of storing this once per user, we are storing this once per tweet!  However, this makes the data easy to access if we ever need the tweet along with the user's location.
_Normalized_ tables would save the storage space by separating the information. However, if we ever needed to access those two pieces of information, we would need join the two tables, which can be a fairly slow operation.

#### Alternative databases
While relational tables are most popular database type, and most broadly useful, specific applications may require different data organization. While it's not important to know all varieties, it is good to know the overall themes

#### Key-value stores
Some databases are nothing more than very-large (and very-fast) hash-maps or dictionaries that can be larger than memory. These are useful for storing key based data, i.e. store the last access or visit time per user, count things per user or customer.

Every entry in these databases is two values, a key and value, and we can retrieve any value based on it's key. This is exactly like a python dictionary, but can be much larger than your memory allows and uses smart caching algorithms to ensure frequently or recently accessed items are quickly accessible. 

Popular key-value stores include Cassandra or memcachedb.


#### NoSQL or Document databases
What are commonly referred to to "NoSQL" databases are those that don't rely on a traditional table setup and are more flexible in their data organization. Typically they do actually have SQL querying abilities, but model their data differently.

Many still organize the data on an entity level, but more often have denormalized and nested data setups. For example, for each user, we may store their metadata, along with a collection of tweets, each of which has it's own metadata. This nesting can continue down encapsulating entities.

This data layout is similar to that in JSON documents.

Popular databases of this variety include mongodb and couchdb.

**Check:** Go through a few examples of data storage and talk about what might be a better storage solution. I think for many of these the answers may be personal preference, but the students should understand some of the differences.
- Database for an application with user profiles:
    - (As always) Probably relational DB:
    - Perhaps 'noSQL' database if user profile fields are commonly changing updating or we want to make it easy to do complex queries on user profile elements
- Database for an online store
    - Probably relational DB: Fast storage querying, standard table structures, transactional capabilities
- Storing last visit date of a user
    - A fast key-value store might be useful for caching interesting statistics about a user (last visit, total visits)

<a name="demo1"></a>
## Demo: Accessing Databases from Pandas (20 min)

While database provide many analytical capabilities, often times it's useful to pull the data back into Python for more flexible programming. Large, fixed operations would e more efficient in a database, but Pandas allow for interactive processing.

```python
import pandas as pd
from pandas.io import sql
```

Pandas can be used to connect to most relational databases. In this demonstration, we will create and connect to a SQLite database. SQLite creates portable SQL databases saved in a single file. These databases are stored in a very efficient manner and allow fast querying making them ideal for small databases or databases that need to be moved across machines.

If you are looking to start using a database without the setup of `mysql` or `postgres`, SQLite is a good start.

We can create a SQLite database as follows:
```python
import sqlite3

conn = sqlite3.connect('dat-test.db')
```

This creates a file `dat-test.db` which will act as a relational/SQL database.

### Writing data into a database

Data in Pandas can be loaded into a relational database. For the most part, Pandas can use it's column information to infer the schema for the table it creates.

Let's return to the Rossmann sales data and load that into the database.

```python
import pandas as pd

data = pd.read_csv('../../../lesson-15/assets/dataset/rossmann.csv', low_memory=False)
data.head()
```

Data is moved to the database through the `to_sql` command, similar to the `to_csv` command.

`to_sql` takes as arugments:
    - `name`, the table name to create
    - `con`, a connection to a database
    - `index`, whether to input the index column
    - `schema`, if we want to write a custom schema for the new table
    - `if_exists`, what to do if the table already exists. We can overwrite it, add to it, or fail

```python
data.to_sql('rossmann_sales',
            con=conn,
            if_exists='replace',
            index=False)
```

### Reading data from a database

If we already have data in the database, or we input as we just did we can use Pandas to query our database.

Querying is done through the `read_sql` command in the `sql` module.

```python
import pandas as pd
from pandas.io import sql

sql.read_sql('select * from rossmann_sales limit 10', con=conn)
```

This runs the query passed in and returns a dataframe with the results

```python
sql.read_sql('select Store, SUM(Customers) from rossmann_sales group by Store', con=conn)
```

**Check:** Load the Rossmann Store metadata in `rossmann-stores.csv` and create a table into the database from it

```python
rossmann_stores = pd.read_csv('../../assets/dataset/rossmann-stores.csv')
rossmann_stores.to_sql('rossmann_stores', con=conn)
```

<a name="demo2"></a>
## Demo/Codealong: SQL Syntax (60 mins)

## SQL Operators
> NOTE: For each of the following, they can be demo-ed in pandas using the data we've setup above. I've written a demo and check for each, but it is up to the instructor whether to do the simultaneously or go through all the operators first and then the checks or mix them in between.

### SELECT 
Every query should start with `SELECT`.  `SELECT` is followed by the names of the columns in the output.

`SELECT` is always paired with `FROM`, and `FROM` identifies the table to retrieve data from.

```sql
SELECT
<columns>
FROM
<table>
```

`SELECT *` denotes returns *all* of the columns.

Rossmann Stores example:
```sql
SELECT 
Store, Sales 
FROM rossmann_sales;
```

**Check:** Have the students write a query that returns the `Store`, `Date` and `Customers`

```sql
SELECT 
Store, Date, Customers 
FROM rossmann_sales;
```

### WHERE 
`WHERE` is used to filter table to a specific criteria and follows the `FROM` clause.

```sql
SELECT
<columns>
FROM
<table>
WHERE
<condition>
```

Rossman Stores example:
```sql
SELECT 
Store, Sales 
FROM rossmann_sales
WHERE Store = 1;
```

```sql
SELECT 
Store, Sales 
FROM rossmann_sales
WHERE Store = 1 and Open = 1;
```

The condition is some filter applied to the rows, where rows that match the condition will be in the output.

**Check:** Have the students write a query that returns the `Store`, `Date` and `Customers` for when the stores were open and running a promotion.

```sql
SELECT 
Store, Date, Customers 
FROM rossmann_sales
WHERE Open = 1 and Promo = 1;
```

### GROUP BY

`GROUP BY` allows us to aggregate over any field in the table. We identify some key with which want to segment the rows. Then, we roll-up or compute some statistic over all of the rows that match that key.

`GROUP BY` *must be* paired with an aggregate function - the statistic we want to compute in the rows - in the `SELECT` statement.

`COUNT(*)` denotes counting up all of the rows.  Other aggregate functions commonly available are `AVG` - average, `MAX`, `MIN`, and `SUM`.

If we want an aggregate over the entire table - without results specific to any key, we can use an aggregate function in the `SELECT` statement and ignore the `GROUP BY` clause.

Rossman Stores example:
```sql
SELECT 
Store, SUM(Sales), AVG(Customers)
FROM rossmann_sales
GROUP BY Store
WHERE Open = 1;
```

**Check:** Have the students write a query that returns the total sales on promotion days.

```sql
SELECT
Promo, Store, SUM(Sales)
FROM rossmann_sales
GROUP BY Promo
```

### ORDER BY
Results of a query can be sorted by `ORDER BY`.

If we want to order the states by the number of polls, we would add an `ORDER BY` clause at the end. 

Rossman Stores example:
```sql
SELECT 
Store, SUM(Sales) as total_sales, AVG(Customers)
FROM rossmann_sales
GROUP BY Store
WHERE Open = 1;
ORDER BY total_sales desc;
```

`COUNT(*) as cnt` renames the `COUNT(*)` value as `cnt` so we can refer to it later in the `ORDER BY` clause.

`desc` tells the result to be in _descending_ order as opposed to increasing.


### JOIN
`JOIN` allows us to access data across many tables. We will need to specify how a row in one table links to a row in another.  

```sql
SELECT a.Store, a.Sales, s.CompetitionDistance
FROM 
rossmann_sales a
JOIN
rossmann_stores s
ON a.Store = s.Store
```

Here `ON` denotes an *Inner Join*.

#### Inner Join
By default, most joins are an *Inner Join*, which means only when there is a match in both tables, does a row appear in the result.

#### Outer Join
If we want to keep the rows of one table *even if there is no matching counterpart* we can perform an *Outer Join*. Outer joins can be `LEFT`, `RIGHT`, or `FULL` meaning keep all the left rows, all the right rows or *all* the rows, respectively.

<a name="ind-practice"></a>
## Independent Practice: Pandas and SQL (40 mins)

1. Load the Walmart sales and store features data
1. Create a table for each of those datasets
1. Select the store, date and fuel price on days it was over 90 degrees
1. Select the store, date and weekly sales and temperature
1. What were average sales on holiday vs. non-holiday sales
1. What were average sales on holiday vs. non-holiday sales when the temperature was below 32 degrees

> NOTE: Solutions in notebook


<a name="independent-practice-sql"></a>
## Independent Practice: SQL (Untimed - Extra)

> NOTE: I've put multiple options of practice here, as some may want more practice for this than others. This can be done in addition to the above exercises.



#### PG-Exercises

> NOTE: pgexercises.com is a very good site for Postgres exercises, as it has a nice table setup to explore as well as an interactive question/answering. The one issue is that for a large class, the website can start to hang to responses. This just means that the website will not properly grade the answers, but the student can still complete the exercises. Alternatively, the exercises can also be set up locally if the student have installed Postgres.  The instructions are available at the bottom this site: https://pgexercises.com/gettingstarted.html.

From Postgres Exercises:
> The dataset for these exercises is for a newly created country club, with a set of members, facilities such as tennis courts, and booking history for those facilities. Amongst other things, the club wants to understand how they can use their information to analyze facility usage/demand. 

For each student, I would suggest doing 3-5 questions in each of:
1. [Simple SQL Queries](https://pgexercises.com/questions/basic/)
1. [Aggregation](https://pgexercises.com/questions/aggregates/)
1. [Joins and Subqueries](https://pgexercises.com/questions/joins/)

There are many more questions in each of these for students who want more practice as well as advanced sections.

#### Wagon
> NOTE: This requires signing up for the Wagon service, and downloading their application, but it gives access to some sample databases.

The following exercises use the `music_store` database in the Wagon application. We will work 3 tables from it:
- `music_store.artists`
- `music_store.albums`
- `music_store.tracks`


1. Display all tracks on which Jimmy Page was the composer.
1. Who were the top five composers by number of tracks?
1. Who were the top five composers by length of tracks?
1. Select all of the albums from Led Zeppelin
1. Count the number of albums per artist, and display the top 10.
1. Display the track name and album name from all Led Zeppelin albums
1. Compute how many songs and how long (in minutes) each Led Zeppelin album was.

Solutions: 
> Display all tracks on which Jimmy Page was the composer.

```sql
select *
from 
music_store.tracks
where composer = 'Jimmy Page'
```

> Who were the top five composers by number of tracks?

```
select composer, count(name) as cnt
from 
music_store.tracks
group by composer
order by cnt desc
limit 5
```

> Who were the top five composers by length of tracks?

```
select composer, sum(milliseconds) as ms
from 
music_store.tracks
group by composer
order by ms desc
limit 5
```

> Select all of the albums from Led Zeppelin

```
select 
*
from 
music_store.artists as a
join
music_store.albums as b
on a.id = b.artist_id
where a.name = 'Led Zeppelin'
```

> Count the number of albums per artist, and display the top 10.

```
select a.id, a.name, count(*) as cnt
from 
music_store.artists as a
join
music_store.albums as b
on a.id = b.artist_id
group by a.id
order by cnt desc
limit 10;
```

> Display the track name and album name from all Led Zeppelin albums

```
select 
b.title, t.name
from 
music_store.artists as a
join
music_store.albums as b
on a.id = b.artist_id
join
music_store.tracks as t
on b.id = t.album_id
```

> Compute how many songs and how long (in minutes) each Led Zeppelin album was.

```
select 
b.title, count(*), sum(t.milliseconds) / 60000 as time_in_min
from 
music_store.artists as a
join
music_store.albums as b
on a.id = b.artist_id
join
music_store.tracks as t
on b.id = t.album_id
where a.name = 'Led Zeppelin'
group by b.title
```

## Demo: Installing Postgres (15 min)
> NOTE: This could be a good option for students that want to use a database in their projects.

On a Mac:
```sh
brew install postgres
```

`brew` will provide a few command to make sure postgres runs on startup.

If this is done, you can demonstrate the `postgres` command line.

#### Demo of Postgres Shell
- Starting postgres
```sh
psql
```

- Listing tables
```sh
\dt
```

- Creating a table
```sh
create table example(
    id int,
    name varchar(140),
    value float
);
```

- Inserting a row
```sql
insert into example values(1, 'general assembly', 3.14);
```

- Querying it back
```sql
select * from example;
```

<a name="conclusion"></a>
## Conclusion (5 mins)
- While this was a brief introduction to databases, they are often at the core of any data analysis. Most data acquisition would start with retrieving data from a database. SQL is a key language that any data scientist should understand, and luckily there are only a few key instructions to remember:
    - SELECT: Used in every query to define the resulting rows
    - WHERE: To filter rows based a value in a certain column
    - GROUP BY: Equivalent to the Pandas group by
    - JOIN: Equivalent to the Pandas join

- Pandas can be used to access data from databases as well. The result of the queries will end up in a Pandas dataframe.
- There is much more to learn about query optimization if one dives further, but we haven't covered that here.

***

### BEFORE NEXT CLASS
|   |   |
|---|---|

### ADDITIONAL RESOURCES
- [Pandas: Comparison with SQL](http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html)
- [PostgresSQL Exercises](https://pgexercises.com/)
- [SQL Zoo](http://sqlzoo.net/)
