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

#from scikit-learn website
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main():
   
   #3. Pre-processing?(could skip)
   #4. Use sklearn.feature_extraction.text.CountVectorizer to make matrix of words based of comments

   #1. Group by toxic for the training data
   train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
   train = train_data[["comment_text"]]
   print(train.shape)
   train_labels = train_data[["toxic"]]
   print(train_labels.shape)

   #2. Use train_test_split to split into train/test
   comment_train, comment_test, labels_train, labels_test = train_test_split(train, train_labels, test_size = 0.2, random_state=42)
   #Transpose and flatten so it fits the correct dimensions
   labels_train = np.transpose(labels_train)
   labels_train = np.ravel(labels_train)
   labels_test = np.transpose(labels_test)
   labels_test = np.ravel(labels_test)

   #3. CountVectorizer
   #Create a count matrix for each comment
   count_vect = CountVectorizer(tokenizer = LemmaTokenizer(),
                                strip_accents = 'unicode', # works 
                                lowercase = True)
   comment_train_counts = count_vect.fit_transform(comment_train.comment_text)

   #4. TfidfTransformer
   #Use tf-idf instead
   tf_transformer = TfidfTransformer(use_idf=False).fit(comment_train_counts)
   comment_train_tf = tf_transformer.transform(comment_train_counts)
   tfidf_transformer = TfidfTransformer()
   comment_train_tfidf = tfidf_transformer.fit_transform(comment_train_counts)

   #5 Train a classifier
   #create the model
   clf = MultinomialNB().fit(comment_train_tfidf, labels_train)

   #make the bag of words for the test data
   comment_test_new_counts = count_vect.transform(comment_test.comment_text)
   comment_test_new_tfidf = tfidf_transformer.transform(comment_test_new_counts)

   print(comment_test_new_tfidf)
   
   #6 Prediction:
   prediction = clf.predict(comment_test_new_tfidf)
   
   print(prediction)

   print("Accuracy:", np.mean(prediction == labels_test), "\n")

   print("Precision, Recall, and F1 Score:\n", metrics.classification_report(labels_test, prediction), "\n")
   print("Confusion Matrix:\n", metrics.confusion_matrix(labels_test, prediction), "\n")

   #count_vect.vocabulary_.get(u'algorithm')
   #print(comment_train_counts)

   #print(count_vect.vocabulary_)

   #print(X_train_counts.shape())
   # test_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test.csv")
   # test = test_data.drop(["id"], axis=1)
   # print(test.shape)
   # test_labels = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test_labels.csv")
   # #test_labels = test_labels.drop(["id"], axis=1)
   # test_labels = test_labels[["toxic"]]
   # print(test_labels)



   #bayes_model = MultinomialNB().fit(train, train_labels)
   #predictions = bayes_model.score(test_data, test_labels)


if __name__ == '__main__':
   main()