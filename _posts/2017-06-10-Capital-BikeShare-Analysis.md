The aim of this project is to analyse the publicly available data of Washington DC's bike share system, [Capital Bikeshare](https://www.capitalbikeshare.com). By analysing this data, I hope to provide a clearer picture of the spatiotemporal dynamics of the bikesharing process and understand which aspects might be relevant to the success of the network. The dataset includes data such as trip duration, start & end station and the time of day at which the trip was taken.

## Libraries Used

- Numpy
- Pandas
- Seaborn
- Matplotlib

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Pre-processing

The most time consuming part of this project is the pre-processing: Ensuring that all the DataFrames are in the same format, and more importantly, in a format that enables analysis to be carried out.

I have included only some of the pre-processing steps, so please feel free to check out the full source code on Github(PUT LINK HERE).

#### Import CSV Files

For each calendar year, we are provided with four CSV files (representing each quarter of the year). Using Pandas, we can read the CSV files into a DataFrame:

```python
Q1_2011 = pd.read_csv('Trip Data/2011-Q1.csv')
Q2_2011 = pd.read_csv('Trip Data/2011-Q2.csv')
Q3_2011 = pd.read_csv('Trip Data/2011-Q3.csv')
Q4_2011 = pd.read_csv('Trip Data/2011-Q4.csv')
```
The four quarters are then concatenated into one DataFrame:

```python
Total2011 = pd.concat([Q1_2011,Q2_2011,Q3_2011,Q4_2011])
```
This process is repeated for the following years (2012,2013,2014,2015), with a final result of five DataFrames.

#### Exploring the DataFrames

The first thing to do is to have a look at the DataFrame. We can use Pandas `head()` method to preview the DataFrame.

```python
Total2011.head()
```
![HI](https://github.com/jack-morgan/Personal-Website/gh-pages/Images/df11head.png "Hi")

![2](https://github.com/jack-morgan/Personal-Website/raw/master/Images/df11head.png "2")

{% highlight r %}
    Duration        Start date     End date     Start station       End station     End date   Bike#   Member Type     
1   NaN           NaN
2
3   
{% endhighlight %}



