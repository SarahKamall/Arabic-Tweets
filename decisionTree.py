import numpy as np
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


# Loading Data
Data = pd.read_csv("NewData.tsv", delimiter="\t")

# Preprocessing Of Data
# Deleting Null Tweets
Data['tweets'].replace('', np.nan, inplace=True)
Data.dropna()
Data = shuffle(Data,random_state=80)
# Feature Engineering with TFIDF
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1),
    max_features=10000)

unigramdataGet = word_vectorizer.fit_transform(Data['tweets'].astype('str'))
unigramdataGet = unigramdataGet.toarray()
vocab = word_vectorizer.get_feature_names()
unigramdata_features = pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
unigramdata_features[unigramdata_features > 0] = 1

# Encoding The Labeled Data to ones and Zeros
pro = preprocessing.LabelEncoder()
encpro = pro.fit_transform(Data['type'])
Data['type'] = encpro

# Split The Data to Features and Target variables
x = unigramdata_features
y = Data["type"]

# Create instance of Decision Tree Classifier
DTC = DecisionTreeClassifier()

# Split The data to testing and Training with 20 Cross Validation
cv_scores = cross_val_score(DTC, x, y, cv=20)
print('average of cross validation scores: {:.3f}'.format(np.mean(cv_scores)))

# Calculate F1 Score
scores = cross_val_score(DTC, x, y, cv=20, scoring='f1_weighted')
print('average of F1 scores: {:.3f}'.format(np.mean(scores)))

# Calculate Accuracy
scores = cross_val_score(DTC, x, y, cv=20, scoring='accuracy')
print('average of Accuracy scores: {:.3f}'.format(np.mean(scores)))

