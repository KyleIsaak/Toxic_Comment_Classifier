from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

def main():
   train_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
   train = train_data[["comment_text"]]
   print(train.shape)
   train_labels = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
   print(train_labels.shape)

   test_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test.csv")
   test = test_data.drop(["id"], axis=1)
   print(test.shape)
   test_labels = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test_labels.csv")
   test_labels = test_labels.drop(["id"], axis=1)
   print(test_labels.shape)


   bayes_model = GaussianNB().fit(train, train_labels)
   predictions = bayes_model.score(test_data, test_labels)


if __name__ == '__main__':
   main()