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

{% highlight r %}
> colnames(shooting_data)
[1]  "id"                      "name"                    "date"                    "manner_of_death"        
[5]  "armed"                   "age"                     "gender"                  "race"                   
[9]  "city"                    "state"                   "signs_of_mental_illness" "threat_level"           
[13] "flee"                    "body_camera" {% endhighlight %}


