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

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

def main():

    #1. Group by toxic for the training data
    train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
    test_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test.csv")
    submission = pd.read_csv("jigsaw-toxic-comment-classification-challenge/sample_submission.csv")

    X = train_data.comment_text
    y = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    test = test_data.comment_text

    num_words = 20000
    max_len = 150
    emb_size = 128

    tok = Tokenizer(num_words = num_words)
    tok.fit_on_texts(list(X))

    X = tok.texts_to_sequences(X)
    test = tok.texts_to_sequences(test)

    X = sequence.pad_sequences(X, maxlen = max_len)
    X_test = sequence.pad_sequences(test, maxlen = max_len)

    inp = Input(shape = (max_len, ))
    layer = Embedding(num_words, emb_size)(inp)
    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(6, activation = 'sigmoid')(layer)
    model = Model(inputs = inp, outputs = layer)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.summary()

    file_path = 'save_best'
    checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', verbose = 1, save_best_only=True)
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 1)

    hist = model.fit(X, y, batch_size = 32, epochs = 2, validation_split = 0.2, callbacks = [checkpoint, early_stop])
    # try increasing epoch (more training time, but might overfit)

    

   

    # y_test = model.predict(X_test)


    # print(y_test)
    # print(" ")

    # submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test

    # print(submission)
    # submission.to_csv("sub.csv", index=False)



    # print('Test score:', score)
    # print('Test accuracy:', acc)


if __name__ == '__main__':
   main()