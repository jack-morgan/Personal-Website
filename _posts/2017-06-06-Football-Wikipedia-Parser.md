<!--- 
---
layout: post
title: Football Wikipedia Parser
comments: true
tags: ['Python']
--- 
--->

After playing numerous knowledge-based football games, it appeared to me that extensive knowledge on footballers previous club history is required. Unfortunately, it is extremely time consuming to find this information for each individual player.

### Libraries used

* BeautifulSoup
* Wikipedia
* csv

### Code Walkthrough

The first thing that needs to be done is to create a blank csv file:

```python
with open("playerhistory.csv", "w") as fp:
    a = csv.writer(fp)
    a.writerow("")
```
And also create a blank list, in which the data will be added to:

```python
footballer_names = []
footballer_data = []
```
#### Automating the input to the program

It is not very efficient to manually have to type in each players name to extract information. Therefore, in order to automate this process, a list of footballers names (csv file) can be passed into the program:

```python
with open('footballers.csv', 'rt') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
      if counter > 0:
        footballer_names.append(row[0])
      counter += 1
```

#### Parsing Wikipedia

Utilising the Wikipedia library, we can load and access all the data from a Wikipedia page:

```python
for footballer_name in footballer_names:
  html_ver = wikipedia.page(footballer_name, None, True, True, True).html()
```

##### BeautifulSoup
Now that the program has access to the information from Wikipedia, we can use BeautifulSoup to parse the page:

```python
soup = BeautifulSoup(html_ver, 'html.parser')
array = []
```

By analysing the html code from a Wikipedia page, it is possible to uncover the class name for specific elements on the page. 
'infobox card' is the class name of the table that Wikipedia uses for the player information. With this information, we can now use BeautifulSoup to extract the specific information required:

```python
  for tr in soup.find("table", {"class":"infobox vcard"}).findChildren('tr'):

    if tr.text.find("Senior career*") > -1:
      found = True
      ignore_next = True
    elif tr.text.find("National team") > -1:
      found = False
      break

    if found == True and ignore_next == False:
      array.append(tr.text)
    elif ignore_next == True:
      ignore_next = False
```
The footballer_data list is now updated with the data gathered from the program:

```python
 footballer_data.append(array)
  print(array)
  array.insert(0, footballer_name)
  ```
Finally, the data that has been gathered from the parser is saved to the csv file previously created:

```python
  with open("playerhistory.csv", "a") as fp:
    a = csv.writer(fp)
    a.writerow(array)
```

### Conclusion

This small project explores some of the capabilities of Python when used for text mining / parsing. It demonstrates that through the use of logic and experimentation, processes can be made more efficient, and thus in the long term reduce the amount of time spent on a trivial task.

The full source code is available on [Github](https://github.com/jack-morgan/Football-Wikipedia-Scraper)
