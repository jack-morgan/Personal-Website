## Introduction
After playing numerous knowledge-based football games, it appeared to me that extensive knowledge on footballers previous club history is required. Unfortunately, it is extremely time consuming to find this information for each individual player.

### Libraries used

* BeautifulSoup
* Wikipedia
* csv

### Code Walkthrough

The first thing that needs to be done is to create a blank csv file

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
#### Automating the input to the program

It is not very efficient to manually have to type in each players name to extract information. Therefore, in order to automate this process, a list of footballers names (csv file) can be passed into the program.

```python
with open('footballers.csv', 'rt') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
      if counter > 0:
        footballer_names.append(row[0])
      counter += 1
```

### Parsing Wikipedia

Utilising the Wikipedia library, we can load and access all the data from a Wikipedia page:

```python
for footballer_name in footballer_names:
  html_ver = wikipedia.page(footballer_name, None, True, True, True).html()
```
Now that the program has access to the information from Wikipedia, we can use BeautifulSoup to parse the page.

```python
soup = BeautifulSoup(html_ver, 'html.parser')
```
