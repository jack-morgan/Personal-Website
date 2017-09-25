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

For the majority of machine learning algorithms, the values must be numeric and thus the 'sex' must be converted from male/female to 1/0:

```python
for df in [df_test,df_train]:
    df['Sex']=df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
```

## Part 1: Decision Tree Classifier (CART)

A decision tree will be used to classify the data, and thus predict whether the passenger survived (1) or not (0). For information purposes, decision tree algorithms are also referred to as Classification and Regression Trees (CART). Decision trees are a quick and easy way to make predictions on data due to the fact that little preprocessing is required. The majority of machine learning algorithms (e.g. regression models) require the features to be scaled or normalised before fitting the model to the data; however, this is not necessary for decision trees as the tree structure is not affected. 

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

Now the training set containing the features (X) must be established, as well as the target value (Y):

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
<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/tree.png" width="680" height="400" />

## Part 2: K Nearest Neighbours (KNN)

### Preprocessing

For the KNN classifier, more features will be used. 

```python
dataclean = data.drop(['PassengerId','Ticket','Cabin','Embarked','Parch','Name'],axis=1)
dataclean['Sex']=dataclean['Sex'].apply(lambda x: 1 if x == 'male' else 0)
dataclean.dropna(inplace=True)
```
We can preview the new dataframe now using the `head()` method:

insert picture here

### Standardise the variables

Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations (and hence on the KNN classifier), than variables that are on a small scale.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataclean)
scaled_features = scaler.transform(dataclean)
```
Now transform the 'scaled_features' into a Pandas Dataframe:

```python
df_feat = pd.DataFrame(scaled_features,columns=dataclean.columns)
```

IN PROGRESS, PLEASE CHECK BACK SOON TO SEE THE FULL PROJECT!!
