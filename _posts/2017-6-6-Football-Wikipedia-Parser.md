## Introduction
After playing numerous knowledge-based football games, it appeared to me that extensive knowledge on footballers previous club history is required. Unfortunately, it is extremely time consuming to find this information for each individual player.

### Modules used

* BeautifulSoup
* Wikipedia
* csv

### Code Walkthrough

The first thing to do is to create a blank csv file

```python
with open("playerhistory.csv", "w") as fp:
    a = csv.writer(fp)
    a.writerow("")
```
And also create a blank list, in which the data will be added to.

```python
footballer_names = []
footballer_data = []
```

