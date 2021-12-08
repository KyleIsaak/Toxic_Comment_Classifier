from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
#for preprocessing
import re

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint


#Read the data
train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")

#Make the data lowercase
train_data["comment_text"] = train_data["comment_text"].str.lower()
print("pre clean:\n\n", train_data["comment_text"])

def cleaning(data):
    #remove the characters in the first parameter
    clean_column = re.sub('<.*?>', ' ', str(data))
    #removes non-alphanumeric characters(exclamation point, colon, etc) except periods.
    clean_column = re.sub('[^a-zA-Z0-9\.]+',' ', clean_column)       
    #tokenizing is done
    tokenized_column = word_tokenize(clean_column)
    return tokenized_column
#use panda apply to apply to each comment
train_data["cleaned"] = train_data["comment_text"].apply(cleaning)
print("post clean:\n\n", train_data["cleaned"])

#lemmatize all the words
lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    #input our data in function, take the cleaned column
    my_data = data["cleaned"]
    lemmatized_list = [lemmatizer.lemmatize(word) for word in my_data]
    return (lemmatized_list)

train_data["lemmatized"] = train_data.apply(lemmatizing, axis = 1)
print("post lemmatize:\n\n", train_data["lemmatized"])
