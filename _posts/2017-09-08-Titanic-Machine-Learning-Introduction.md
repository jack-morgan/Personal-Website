The [Titanic Kaggle Data Set](https://www.kaggle.com/c/titanic) is one of the most commonly used when starting out as a data scientist. It provides an easy way to gain 'real-world' experience analysing a dataset, and enables basic machine learning algorithms to be explored. In this project I use both a decision tree classifier and a K Nearest Neighbors (KNN) classifier.

## Libraries Used

- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikitlearn


## Preprocessing

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

Before delving into the problem, it is best to visualise the data first. Using `seaborn`'s `countplot` method we can gain insight into the distribution of survivors based on sex

```python
plt.title('Sex of Survived')
sns.countplot(x='Survived',hue='Sex',data=train_set)
```

<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/Survivedbysex.png" width="400" height="300" />

For the majority of machine learning algorithms, the values must be numeric and thus the 'sex' must be converted from male/female to 0/1:

```python
for df in [df_test,df_train]:
    df['Sex']=df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
```

Part 1: Decision Tree Classifier

A decision tree will be used to classify the data, and thus predict whether the passenger survived (1) or not (0).

Only the 'Sex' and 'Fare' features will be used, so the other features can be dropped:


```python
df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Pclass'],axis=1,inplace=True)
df_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Pclass'],axis=1,inplace=True)
```

Furthermore, it is essential that there are no null values, as the ML algorithm will not accept these values. There are multiple methods to deal with null values, such as replacing the null with the features mean; however, for this example they will simply be deleted:

```python
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
```

IN PROGRESS --> PLEASE CHECK BACK SOON TO SEE THE FULL PROJECT!!

The set containing the features (X) must be established, as well as the target value (Y):

```python
X_train_ = df_train[['Sex','Fare','Age']]
y_train = df_train['Survived']
X_test = df_test[['Sex','Fare','Age']]
```
Now it is necessary to import `sklearn`'s `DecisionTreeClassifier`. Now an instance of the classifier must be created and then fit to the data:

```python
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
```
Now the decision tree can be used to make predictions on the target values of the test data (whether the passenger survived or not).

```python
tree_predictions = dtree.predict(X_test)
```
The decision tree can be visualised:

```python
from sklearn import tree
tree.export_graphviz(dtree, out_file='DecisionTree2.dot')
```
<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/tree.png"/>
<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/tree.png" width="400" height="300" />
<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/tree2.png" width="400" height="300" />

IN PROGRESS, PLEASE CHECK BACK SOON TO SEE THE FULL PROJECT!!
