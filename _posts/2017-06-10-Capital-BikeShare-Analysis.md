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
df2011 = pd.concat([Q1_2011,Q2_2011,Q3_2011,Q4_2011])
```
This process is repeated for the following years (2012,2013,2014,2015), with a final result of five DataFrames.

#### Exploring the DataFrames

The first thing to do is to have a look at the DataFrame. We can use Pandas `head()` method to preview the DataFrame.

```python
df2011.head()
```
![df2011](https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/df11head.png "2011 DataFrame")

We need to put all the DataFrames into the same format/layout so that it is easier to analyse. We can re-order the columns using `reindex_axis`, using `axis=1` to ensure we are reindexing the columns and not the rows. The columns can be renamed too:

```python
df11.reindex_axis(['Bike#','Duration','End station','End date','Start station','Start date','Member Type'],axis=1)
df11.rename(columns={'End station':'End Station','Start station':'Start Station'},inplace=True)
```
Now let's look at 2012:

```python df12.head()```

![df2012](https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/df12head.png "2012 DataFrame")

The columns 'Bike Key','Subscriber Type' and 'Subscription Type' are empty and useless, so we should delete them using the method `drop()`. The column 'Type' should be renamed to 'Member Type' to keep it in the same format as the previous DataFrame.

```python
df12.drop(df12.columns[[0,7,8]],1,inplace=True)
df12.rename(columns={'Type':'Member Type'},inplace=True)
```
The argument `inplace=True` is used to make sure the changes are saved to the DataFrame.
The other DataFrames are edited in the same way so that everything is uniform (please see here)(GITHUB LINK).

#### Save the DataFrames

Obviously, it is possible to continue using the DataFrames in the same Jupyter Notebook as they are; however, I prefer saving the pre-processed data back to a CSV file in order to save time in the future and not have to re-run code. 

```python
df11.to_csv('2011.csv',index=False)
df12.to_csv('2012.csv',index=False)
df13.to_csv('2013.csv',index=False)
df14.to_csv('2014.csv',index=False)
df15.to_csv('2015.csv',index=False)
```
The `index=False` argument is passed, as we do not want Pandas to create an index, as one already exists.

## Data Analysis

In the GitHub project, you will see a new Jupyter Notebook has been created to make the workspace cleaner. In the new Notebook we can simply import the new (pre-processed) CSV files.

### Variation of Trips Throughout the Day

In order to analyse the trip data by using the time/dates we need to do a few things. We can use pandas `DatetimeIndex()` for dealing with dates. We then use `date.astype()` to create an index with values cast to datetime64. From this we can return the hour `.hour` from the datetime object, the day of the week `.dayofweek`. Finally we can create a string representing the time under a specific format using the method `.strftime()`. We use `('%a')`, which is the format string for the abbreviated weekday names (Sun, Mon, etc.).

This code snippet displays how these steps are applied to one of the DataFrames:

```python
ind11 = pd.DatetimeIndex(df11['Start date']) #Use pandas.DatatimeIndex class for dealing with dates
df11['date'] = ind11.date.astype('datetime64') #creates an index with values cast to dtypes
df11['hour'] = ind11.hour #returns the hour from the datetime object
df11['Dayofweek'] = ind11.dayofweek #returns the day of the week (Monday = 0, Sunday = 6)
df11['DOW'] = ind11.strftime('%a') #%a is the format string for Weekday Abbreviated name (Sun,Mon, etc.)
```
Now we can visualise this variation of trips throughout the day by using a combination of the Matplotlib and Seaborn libraries.  

A title can be added to each axis instance in a figure. To set the title, we use the `set_title` method in the axes instance. Similarly we can set the labels of the X and Y axes using the methods `set_xlabel` and `set_ylabel`.

Seaborn's countplot allows us to aggregate the data off a categorial feature, which in this case is `hour`. The countplot is similar to the barplot except that the estimator is explicitly counting the number of occurences (which is why we only pass the x value).

We plot each year side-by-side so that it is easier to compare visually. A common issue with Matplolib is overlapping subplots or figures. We can use `plt.tight_layout()` method, which automatically adjusts the positions of the axes on the figure canvas so that there is no overlapping content.



```python
fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(24,5),sharey=True)
sns.countplot('hour',data=df11,order=range(25),palette="GnBu_d",ax=axes[0])
sns.countplot('hour',data=df12,order=range(25),palette="GnBu_d",ax=axes[1])
sns.countplot('hour',data=df13,order=range(25),palette="GnBu_d",ax=axes[2])
sns.countplot('hour',data=df14,order=range(25),palette="GnBu_d",ax=axes[3])
sns.countplot('hour',data=df15,order=range(25),palette="GnBu_d",ax=axes[4])
for ax in axes:
    ax.set_xlabel('Hour of the Day',fontsize=14)
    ax.set_ylabel('Number of Journeys')
axes[0].set_title('Total Trips by Time of Day - 2011',fontsize=16)
axes[1].set_title('Total Trips by Time of Day - 2012',fontsize=16)
axes[2].set_title('Total Trips by Time of Day - 2013',fontsize=16)
axes[3].set_title('Total Trips by Time of Day - 2014',fontsize=16)
axes[4].set_title('Total Trips by Time of Day - 2015',fontsize=16)
plt.tight_layout()
```

![Totaltrips](https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/Tripsbytimeofday.png "Timeofdaytrips")


