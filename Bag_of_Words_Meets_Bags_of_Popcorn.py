
# coding: utf-8

# In[1]:

import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk


# In[23]:

train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3);


# Parsing part

# In[32]:

from nltk.corpus import stopwords # Import the stop word list

def handle(text, index):
    if index % 1000 == 0:
        print("Now in text: ", index)
    return clean_text(text)

def clean_text(text):
    t = BeautifulSoup(text, "lxml")
    t = re.sub("[^a-zA-Z]", " ", t.get_text())
    t = t.lower()
    words = t.split()
    words = [w for w in words if not w in stopwords.words("english")]
    return( " ".join( words ))


# In[9]:

train = np.array([handle(x,index) for index, x in enumerate(train["review"])])
train.dump('data/trained.out');


# In[24]:

processed_train = np.load("data/trained.out")

print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(processed_train)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# In[25]:

print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )


# In[26]:

forest


# ### Testing part

# In[33]:

# Read the test data
test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t",                    quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

clean_test_reviews = np.array([handle(x,index) for index, x in enumerate(test["review"])])

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )


# In[19]:

def classified_correct(model, i):
    return (model["sentiment"][i] == 0 and int(model["id"][i].split("_")[1]) <= 5) or            (model["sentiment"][i] == 1 and int(model["id"][i].split("_")[1]) > 5)

model = pd.read_csv("Bag_of_Words_model.csv");
correct = np.array([classified_correct(model,i) for i in range(model.shape[0])])
print(correct.sum() / model.shape[0])


# In[14]:

correct.sum()


# In[ ]:



