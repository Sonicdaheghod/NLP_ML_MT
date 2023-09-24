# Predicting Positive and Negative Reviews Using Natural Language Processing and Random Forest Classifier
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)
* [Improvements Made](#Improvements-Made)
* [Credits](#Credits)

## Purpose of Program

This program was created to predict if a car was stolen using data about the car's color, model, and time of day.

## Screenshots

Final predictions by model using test dataset:

<img width="615" alt="image" src="https://github.com/Sonicdaheghod/MT_Naive_Bayes/assets/68253811/d33f8bda-a380-406e-922d-70554d614a93">


## Technologies
Languages/ Technologies used:

* Jupyter Notebook

* Python3

## Setup

Download the necessary libraries and packages:
```
pip install numpy
pip install pandas
pip install scikit-learn
```
Check to see if version of Python/Python3 (if on jupyter Notebook) is used, this is to ensure packages work properly. Python 3.8 or better is recommended.

```
import sys
sys.version
```

Import the following packages and libraries:

```
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
```
  
## Using the Program

1) The .csv data from the dataset was changed to numerical values for the model to better run it through the Naive- Bayes Classifier Model.

    * Turned the datapoints to numbers by using a dictionary to assign unique values to corresponding numbers and saving those changes to new .csv file.
  
2) Set Independent and Dependent variables (IV and DV, respectively)

    * We want the model to use the features of the car (color, model, day) to predict if that car was stolen or not.
    * IV are features, DV is stolen or not
      
3) Prepared the Naive- Bayes Classifier Model training and test dataset with ``` train_test_split ``` from scikit-learn using our assigned IV and DV.

4) Ran Model Using Gaussian Distribution

    * Same thing as normal distribution
    * Stolen/not stolen distributions plotted on graph for each of the three IV's

Watch StatQuest's video on [Gaussian Distribution](https://www.youtube.com/watch?v=H3EjCKtlVog&ab_channel=StatQuestwithJoshStarmer)

5) Evaluated Model's performance

Detailed description under "improvements made"

## Improvements Made

* Evaluated model's performance using a confusion matrix and classification report for better detail.

Confusion Matrix:
This shows how the model correctly/incorrectly categorized data it predicted and how it compares to the actual label for the data used.

```
array([[1, 1],
       [0, 5]], dtype=int64)
```

Classification Report:

* This tells us accuracy, precision, recall, F1 score, macro and weighted avg.
* The support column shows how many items actually belong in a specific category

**F1 score**

* the closer value is to 1, the better the model
* calculate using formula: 2* (Precision * Recall)/ (Precision + Recall)
```
     precision    recall  f1-score   support

         yes       1.00      0.50      0.67         2
          no       0.83      1.00      0.91         5

    accuracy                           0.86         7
   macro avg       0.92      0.75      0.79         7
weighted avg       0.88      0.86      0.84         7
```
## Credits

* Tutorial referenced: [Goeduhub Technologies]([https://youtu.be/3y9XVlk9cDA?](https://www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training/model-selection-results?u=74412268))
* Dataset from [User]()

