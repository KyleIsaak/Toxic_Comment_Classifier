import pandas as pd

from sklearn import preprocessing

#read csv file into panda df
train_data_df = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")

#Check if any values are null
#print(train_data_df.isna().any())

#We can see that the vast majority is non-toxic - has 0 in every column
# print(train_data_df.groupby(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult' , 
#          'identity_hate']).size())

# print(train_data_df.groupby(['toxic']).count())

# data2 = train_data_df.groupby(['toxic'])
# print(data2)

#print(train_data_df.isnull().values.any())

#print(train_data_df.describe())

#Num of toxic values
print(train_data_df['toxic'].value_counts(), "\n")
#Num of severe_toxic values
print(train_data_df['severe_toxic'].value_counts(), "\n")
#Num of obscene values
print(train_data_df['obscene'].value_counts(), "\n")
#Num of threat values
print(train_data_df['threat'].value_counts(), "\n")
#Num of insult values
print(train_data_df['insult'].value_counts(), "\n")
#Num of identity_hate values
print(train_data_df['identity_hate'].value_counts(), "\n")

print(train_data_df.head())
print(train_data_df.describe())