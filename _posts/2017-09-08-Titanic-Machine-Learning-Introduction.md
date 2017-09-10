The [Titanic Kaggle Data Set](https://www.kaggle.com/c/titanic) is one of the most commonly used when starting out as a data scientist. It provides an easy way to gain 'real-world' experience analysing a dataset, and enables basic machine learning algorithms to be explored. In this project I use both a decision tree classifier and a K Nearest Neighbors (KNN) classifier.

## Libraries Used

- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikitlearn

After importing the libraries, we need to read the data as a dataframe, using Pandas:

```python
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
```

We use `pandas head` method to preview the dataframe.

```python
df_train.head()
```
![df_train_head](https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/titanic_head.png "Train DataFrame")

It is clear that all the variables are not needed; for this project only 

Before delving into the problem, it is best to visualise the data first. Using `seaborn`'s `countplot` method we can gain insight into the distribution of survivors based on sex

```python
plt.title('Sex of Survived')
sns.countplot(x='Survived',hue='Sex',data=train_set)
```

<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/Survivedbysex.png" width="400" height="300" />




