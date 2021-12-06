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

    #1. Group by toxic for the training data
    train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
    train = train_data[["comment_text"]]
    # print(train.shape)
    train_labels = train_data[["toxic"]]
    # print(train_labels.shape)
    

    #2. Use train_test_split to split into train/test
    comment_train, comment_test, labels_train, labels_test = train_test_split(train, train_labels, test_size = 0.2, random_state=42)
    #Transpose and flatten so it fits the correct dimensions
    labels_train = np.transpose(labels_train)
    labels_train = np.ravel(labels_train)
    labels_test = np.transpose(labels_test)
    labels_test = np.ravel(labels_test)

    # #3. CountVectorizer
    # #Create a count matrix for each comment
    # count_vect = CountVectorizer(tokenizer = LemmaTokenizer(),
    #                             strip_accents = 'unicode', # works 
    #                             lowercase = True)
    # comment_train_counts = count_vect.fit_transform(comment_train.comment_text)

    # #4. TfidfTransformer
    # #Use tf-idf instead
    # tf_transformer = TfidfTransformer(use_idf=False).fit(comment_train_counts)
    # comment_train_tf = tf_transformer.transform(comment_train_counts)
    # tfidf_transformer = TfidfTransformer()
    # comment_train_tfidf = tfidf_transformer.fit_transform(comment_train_counts)


    num_words = 20000
    max_len = 150
    emb_size = 128

    tok = Tokenizer(num_words = num_words)
    tok.fit_on_texts(list(comment_train.comment_text))

    comment_train2 = tok.texts_to_sequences(comment_train.comment_text)
    comment_test2 = tok.texts_to_sequences(comment_test.comment_text)

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

    prediction = (model.predict(comment_test2).ravel()>0.5)+0 

    target_names = ['non-toxic', 'toxic']
    print("\n")
    print("Accuracy:", np.mean(prediction == labels_test), "\n")
    print("Precision, Recall, and F1 Score:\n", metrics.classification_report(labels_test, prediction), "\n")
    cm = confusion_matrix(labels_test, prediction)
    print("Confusion Matrix:\n", cm, "\n")

    cmd = ConfusionMatrixDisplay(cm, display_labels=["non_toxic", "toxic"])
    cmd.plot()
    plt.show()

    # plt.savefig("CM_LSTM.png")



if __name__ == '__main__':
   main()