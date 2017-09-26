A [Kaggle Project](https://www.kaggle.com/c/invasive-species-monitoring) using a combination of computer vision and machine learning to automatically classify images and determine whether they contain an invasive hydrangea plant.

<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/invasiveIntro.png" width="340" height="300" />


## Libraries Used

- Numpy
- Pandas
- OS
- OpenCV
- Scikitlearn

## Preprocessing

The first step is to import the training target values provided by Kaggle using `pandas`:

```python
labels_csv = pd.read_csv('train_labels.csv')
```
A list of the paths of all the training images are needed. This can be obtained using the `OS` module:

```python
img_paths = os.listdir('train/')
```
A function needs to be created in order to generate an array containing the images and their corresponding target values (labels).

```python
def load_train_images(img_paths, labels_csv):
    images = []
    labels = []
    for i in img_paths:
        row = int(i.split('.')[0])
        # i.split('.') => e.g. ['1', 'jpg']
        # I want the first element of the list, that's why there's a [0]
        
        forest_class = labels_csv.iloc[row - 1]['invasive']
        img = cv2.imread('train/' + i, cv2.IMREAD_GRAYSCALE)
        
        # I resized every image into and arbitrary size to make processing time faster
        # However, this affects classification rate.
        
        img_resized = cv2.resize(img, dsize=(500, 300))
        images.append(img_resized)
        labels.append(forest_class)
    return images, labels
```

## Local Binary Pattern Texture Descriptors

For more information on local binary patterns please [click here](https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)

This function takes the list of images we created previously, computes the LBP histogram of each image and returns a new list with the corresponding LBP features of each image.

```python
def generate_lbp(images, num_points, radius):
    data = []

    # loop through the images list using enumerate because it returns the element
    # of the list (img) as well as the index of the corresponding image (i)
    
    for i, img in enumerate(images):
        # Calculate lbp and create histogram of each image
        lbp = feature.local_binary_pattern(img, num_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

        # convert histogram to float values and normalise.
        hist = hist.astype('float')
        hist /= hist.sum()

        # I printed the index to view the progress in the console.
        print(i)
        data.append(hist)
    return data
```
The image itself is not the feature vector. A histogram of each LBP code must be generated from the image in order to generate the feature vector, using `np.histogram` in the above function.

The functions have been created, and now we need to make use of them!!

```python
''' generating training data '''
print('loading images and labels...')
train_images, labels_list = load_train_images(img_paths, labels_csv)
print('generating training data...')
# generating LBP with arbitrary num_points and radius parameters
train_data = generate_lbp(train_images, 24, 8)
```
The `train_data` should be converted to a `Pandas DataFrame` in order to make the information easier to use in the future.

```python
train_data = pd.DataFrame(train_data)
```
The labels columns must be appended to the features vector (train_data), and finally our data should be exported to a csv file so that the data can simply be imported when needed.

```python
train_data = pd.concat([train_data, pd.DataFrame(labels_list)], axis=1)
train_data.to_csv('train_data.csv', index=False)
```

## Classification / Machine Learning

In python, a new file was created for this section, as technically everything previously done was 'preprocessing'. For continuity purposes I will include a code snippet showing how the data previously generated, was imported and seperate the labels array:

```python
data = pd.read_csv('train_data.csv')
train_data = data.values[:,:-1]
labels = data.values[:,-1]
```
Now that the data is in the correct format, `scikit-learn`’s `train_test_split` method can be used to split the dataset into both a training and a testing set. This is useful as we do not have the target value for the testing set, and thus it now allows us to evaluate each model.

```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data, labels, test_size=0.2, random_state=42)
```
It is possible to use a variety of different models to classify these images: SVC, DecisionTree, KNN, MLP; however, the `RandomForest` classifier achieved the highest accuracy. Obviously it is important to note that overfitting a model can be detrimental to a classification problem. Just because a model works well on training data, it doesn't mean that it will perform well on new, unseen data. Normally, it is better to create a more general model instead of trying ensure that the model gets 99% accuracy on training data.

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print('Accuracy: {:.3f}%'.format(result*100))
```

PLEASE COME BACK SOON TO SEE THE FULL PROJECT!!
