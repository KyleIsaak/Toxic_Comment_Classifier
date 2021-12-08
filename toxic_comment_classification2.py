import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re

from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

#from scikit-learn website
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main():
    train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
    
    #Make the data lowercase
    train_data["comment_text"] = train_data["comment_text"].str.lower()
    print("pre clean:\n\n", train_data["comment_text"])
    def cleaning(data):
        #remove the characters in the first parameter
        clean_column = re.sub('<.*?>', ' ', str(data))
        #removes non-alphanumeric characters(exclamation point, colon, etc) except periods.
        clean_column = re.sub('[^a-zA-Z0-9\.]+',' ', clean_column)       
        #tokenize
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

    train_data["comment_text"] = train_data["lemmatized"]
    train = train_data[["comment_text"]]
    # print(train.shape)
    train_labels = train_data[["toxic"]]
    # print(train_labels.shape)
    print(train)
    #2. Use train_test_split to split into train/test
    comment_train, comment_test, labels_train, labels_test = train_test_split(train, train_labels, test_size = 0.2, random_state=42)
    #Transpose and flatten so it fits the correct dimensions
    labels_train = np.transpose(labels_train)
    labels_train = np.ravel(labels_train)
    labels_test = np.transpose(labels_test)
    labels_test = np.ravel(labels_test)
    #comment_train_converted = comment_train.comment_text.astype(str)
    #comment_test_converted = comment_test.comment_text.astype(str)

    #print(comment_train_converted)
    #print(comment_train.comment_text)
    #3. CountVectorizer
    #Create a count matrix for each comment
    count_vect = CountVectorizer()
    comment_train_counts = count_vect.fit_transform(comment_train.comment_text.astype(str))

    #4. TfidfTransformer
    #Use tf-idf instead
    tf_transformer = TfidfTransformer(use_idf=False).fit(comment_train_counts)
    comment_train_tf = tf_transformer.transform(comment_train_counts)

    tfidf_transformer = TfidfTransformer()
    comment_train_tfidf = tfidf_transformer.fit_transform(comment_train_counts)


    # 5 Train a classifier
    # create the model
    clf = MultinomialNB().fit(comment_train_tfidf, labels_train)

    #make the bag of words for the test data
    comment_test_new_counts = count_vect.transform(comment_test.comment_text.astype(str))
    comment_test_new_tfidf = tfidf_transformer.transform(comment_test_new_counts)


    # #6 Train LSTM Model
    num_words = 20000
    max_len = 150
    emb_size = 128

    # vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) 
    # or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf
    tok = Tokenizer(num_words = num_words, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tok.fit_on_texts(list(comment_train.comment_text.astype(str)))

    comment_train2 = tok.texts_to_sequences(comment_train.comment_text.astype(str))
    comment_test2 = tok.texts_to_sequences(comment_test.comment_text.astype(str))

    comment_train2 = sequence.pad_sequences(comment_train2, maxlen = max_len)
    comment_test2 = sequence.pad_sequences(comment_test2, maxlen = max_len)


    inp = Input(shape = (max_len, ))
    layer = Embedding(num_words, emb_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation = 'sigmoid')(layer)
    model = Model(inputs = inp, outputs = layer)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.summary()

    file_path = 'save_best'
    checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 1, save_best_only=True)
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 1)

    model.fit(comment_train2, labels_train, batch_size = 512, epochs = 1, validation_split = 0.2, validation_data = (comment_test2, labels_test), callbacks = [checkpoint, early_stop])


    #6 Prediction:
    prediction_nb = clf.predict(comment_test_new_tfidf)
    prediction_lstm = (model.predict(comment_test2).ravel()>0.5)+0 

    print("\n")
    print("NB Accuracy:", np.mean(prediction_nb == labels_test), "\n")
    print("NB Precision, Recall, and F1 Score:\n", metrics.classification_report(labels_test, prediction_nb), "\n")
    cm_nb = metrics.confusion_matrix(labels_test, prediction_nb)
    print("NB Confusion Matrix:\n", cm_nb, "\n")
    cmd_nb = ConfusionMatrixDisplay(cm_nb, display_labels=["non_toxic", "toxic"])
    cmd_nb.plot()


    print("LSTM Accuracy:", np.mean(prediction_lstm == labels_test), "\n")
    print("LSTM Precision, Recall, and F1 Score:\n", metrics.classification_report(labels_test, prediction_lstm), "\n")
    cm_lstm = confusion_matrix(labels_test, prediction_lstm)
    print("LSTM Confusion Matrix:\n", cm_lstm, "\n")
    cmd_lstm = ConfusionMatrixDisplay(cm_lstm, display_labels=["non_toxic", "toxic"])
    cmd_lstm.plot()

    plt.show()



if __name__ == '__main__':
   main()