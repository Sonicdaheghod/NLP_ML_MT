# Predicting Positive and Negative Reviews Using Natural Language Processing and Random Forest Classifier
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)  
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)
* [Improvements Made](#Improvements-Made)
* [Credits](#Credits)

## Purpose of Program

This program was created to predict whether a restaraunt review was negative or positive after cleaning up the reviews and using the length of the reviews as well as the % of punctuation used.


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
pip install nltk
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support 
```
  
## Using the Program

1) The .csv data from the dataset was changed to numerical values for the model to better run it through the Naive- Bayes Classifier Model.

  
2) Set Independent and Dependent variables (IV and DV, respectively)

      
3) Prepared the Naive- Bayes Classifier Model training and test dataset with ``` train_test_split ``` from scikit-learn using our assigned IV and DV.

4) Ran Model Using Gaussian Distribution


5) Evaluated Model's performance



## Improvements Made


## Credits

* Tutorial referenced: [Goeduhub Technologies](https://www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training/model-selection-results?u=74412268)
* Dataset from [User]()

