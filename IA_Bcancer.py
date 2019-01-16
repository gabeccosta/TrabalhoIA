Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 22:20:52) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.model_selection import train_test_split
>>> from sklearn import svm
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> bcancer = load_breast_cancer()
>>> bcancer.keys()
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

>>> print(bcancer['DESCR'][:900] + '\n...')

.. _breast_cancer_dataset:

Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        The mean, standard error, and "worst" or largest (mean of the three
        largest values) of these features were
...
>>> bcancer.keys()

dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
>>> bcancer['target_names']

array(['malignant', 'benign'], dtype='<U9')
>>> bcancer['feature_names']

array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'], dtype='<U23')
>>> print(bcancer['data'].shape)

(569, 30)
>>> bcancer['data'][:100]

array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
        1.189e-01],
       [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
        8.902e-02],
       [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
        8.758e-02],
       ...,
       [9.787e+00, 1.994e+01, 6.211e+01, ..., 2.381e-02, 1.934e-01,
        8.988e-02],
       [1.160e+01, 1.284e+01, 7.434e+01, ..., 8.449e-02, 2.772e-01,
        8.756e-02],
       [1.442e+01, 1.977e+01, 9.448e+01, ..., 1.565e-01, 2.718e-01,
        9.353e-02]])
>>> bcancer['target']

array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
>>> X_train, X_test, y_train, y_test = train_test_split(bcancer['data'], bcancer['target'], random_state = 0)

>>> print(X_train.shape)

(426, 30)
>>> print(X_test.shape)

(143, 30)

>>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

>>> clf.score(X_test, y_test)

0.958041958041958
>>> clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)


Warning (from warnings module):
  File "C:\Users\Windows81\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\svm\base.py", line 196
    "avoid this warning.", FutureWarning)
FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
>>> clf = svm.SVC(kernel='linear', C=1, gamma='auto').fit(X_train, y_train)
>>> clf.score(X_test, y_test)
0.958041958041958
>>> clf = svm.SVC(kernel='rbf', C=1, gamma='auto').fit(X_train, y_train)
>>> clf.score(X_test, y_test)
0.6293706293706294
>>> clf = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train, y_train)
>>> clf.score(X_test, y_test)
0.951048951048951
>>> clf = svm.SVC(kernel='linear', C=1, gamma='auto').fit(X_train, y_train)
>>> clf.score(X_test, y_test)
0.958041958041958
>>> from sklearn.neighbors import KNeighborsClassifier
>>> svm_1 = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)

Warning (from warnings module):
  File "C:\Users\Windows81\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\svm\base.py", line 196
    "avoid this warning.", FutureWarning)
FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
>>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
>>> svm_01 = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
>>> svm.fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#43>", line 1, in <module>
    svm.fit(X_train, y_train)
AttributeError: module 'sklearn.svm' has no attribute 'fit'
>>> svm_01.fit(X_train, y_train)
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
>>> svm_01.score(X_test, y_test)
0.958041958041958
>>> svm_02 = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train, y_train)
>>> svm_02.score(X_test, y_test)
0.951048951048951
>>> knn_01 = KNeighborsClassifier(n_neighbors = 1)
>>> knn_01.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=1, p=2,
           weights='uniform')
>>> knn_01.score(X_test, y_test)
0.916083916083916
>>> knn_02 = KNeighborsClassifier(n_neighbors = 2).fit(X_train, y_train)
>>> knn_02.score(X_test, y_test)
0.9020979020979021
>>> knn_03 = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
>>> knn_03.score(X_test, y_test)
0.9230769230769231
>>> knn_04 = KNeighborsClassifier(n_neighbors = 4).fit(X_train, y_train)
>>> knn_04.score(X_test, y_test)
0.9230769230769231
>>> knn_05 = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
>>> knn_05.score(X_test, y_test)
0.9370629370629371
>>> svm_03 = svm.SVC(kernel='linear', C=1, gamma='scale').fit(X_train, y_train)
>>> svm_03.score(X_test, y_test)
0.958041958041958
