A [Kaggle Project](https://www.kaggle.com/c/invasive-species-monitoring) using a combination of computer vision and machine learning to automatically classify images and determine whether they contain an invasive hydrangea plant.

<img src="https://github.com/jack-morgan/Personal-Website/raw/gh-pages/Images/invasiveIntro.png" width="340" height="300" />


## Libraries Used

- Numpy
- Pandas
- OS
- OpenCV
- Scikitlearn

## Preprocessing

The first step Import the training target values provided by Kaggle using `pandas`:

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
        forest_class = labels_csv.iloc[row - 1]['invasive']
        img = cv2.imread('train/' + i, cv2.IMREAD_GRAYSCALE)
        
        # I resized every image into and arbitrary size to make processing time faster
        # However, this affects classification rate.
        
        img_resized = cv2.resize(img, dsize=(500, 300))
        images.append(img_resized)
        labels.append(forest_class)
    return images, labels
```

PLEASE COME BACK SOON TO SEE THE FULL PROJECT!!
