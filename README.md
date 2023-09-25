# Predicting Positive and Negative Reviews Using Natural Language Processing and Random Forest Classifier
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)  
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)
* [Future Works](#Future-Works)
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

1) The .tsv data from the dataset was cleaned of stop words/

Two features were made regarding predicting whether a review was positive or negative:
* Function to find percentage of punctuation use out of whole text length
* Function to determine lenght of review
  
2) Vectorized Data for processing into Random Forest Classifier 
* Data was split using ``` train_test_split ``` from scikit-learn

3) Ran Random Forest Classifier\
* Classified each review from dataset as positive or negative based on features.

4) Evaluated Model's performance
* Precision, accuracy, and recall.



## Future Works
* Have model predict reviews as positive or negative using the number of certain types of punctuation used
* Predict based on certain use of adjectives


## Credits

* Tutorial referenced: [Derek Jedamski](https://www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training/model-selection-results?u=74412268)
* Dataset from [Akram](https://www.kaggle.com/datasets/akram24/restaurant-reviews)

