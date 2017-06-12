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
![df2011](https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/df11head.png "2011 DataFrame")

We need to put all the DataFrames into the same format/layout so that it is easier to analyse. We can re-order the columns using `reindex_axis`, using `axis=1` to ensure we are reindexing the columns and not the rows. The columns can be renamed too:

```python
df11.reindex_axis(['Bike#','Duration','End station','End date','Start station','Start date','Member Type'],axis=1)
df11.rename(columns={'End station':'End Station','Start station':'Start Station'},inplace=True)
```
#### Save the DataFrames

Obviously, it is possible to continue using the DataFrames in the same Jupyter Notebook as they are; however, I prefer saving the pre-processed data back to a CSV file in order to save time in the future and not have to re-run code. 

```python
df11.to_csv('2011.csv',index=False)
df12.to_csv('2012.csv',index=False)
df13.to_csv('2013.csv',index=False)
df14.to_csv('2014.csv',index=False)
df15.to_csv('2015.csv',index=False)
```
## Data Analysis

A new Jupyter Notebook is created to make the workspace cleaner. We do not need the above notebook anymore, as we can simply import the new (pre-processed) CSV files.


