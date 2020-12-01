import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

import sklearn
import sklearn.model_selection as ms
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "./data/"

# parses the dataset from the csv file and sets the correct label
def parse_dataset(csv_file, label):
    print("Parsing news dataset from file: {}".format(csv_file))
    file_name = '{}{}'.format(DATA_DIR, csv_file)
    news = pd.read_csv(file_name, low_memory=False)

    # format date for each entry
    news["date"] = news["date"].apply(pd.to_datetime)

    # TODO: is subject category not needed?
    # print (news["subject"].unique())
    # news.drop("subject", axis=1, inplace=news)

    # articles with no text should be cleaned from data
    news = news[news["text"] != ""] 

    # set the appropriate label
    print("Setting label for news dataset: {}".format(label))
    news["label"] = label

    return news


# extracts individual words from news article text
def tokenize(news_data, name):
    all_tokens = [] #all tokens in fake_news articles
    article_tokens_list = [] #list of real_news articles, each in tokenized form
    
    print("Tokenizing {} dataset, this may take a few minutes...".format(name))
    for article in (news_data["text"]):
        words = word_tokenize(article)
        words = [word.lower() for word in words if word.isalpha()] #lowercase
        words = [word for word in words if word not in string.punctuation]
            
        # TODO investigate better solution for tokenization, possibly with pytorch?
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        article_tokens_list.append(words)
        all_tokens.extend(words)

    return all_tokens, article_tokens_list


# Split into training/testing data and preprocess 
def split_and_preprocess(all_tokens, tokens_per_article, all_news):

    X = np.array(tokens_per_article, dtype="object")

    labels = all_news["label"]
    y = [True if article == "FAKE" else False for article in labels]
    y = pd.DataFrame(y, columns=["label"])

    #Create 80-30 train test split
    print("Splitting data: 70% training, 30% testing")
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state=0)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Generate the Sparse Document-Term Matrix from the training data
    vectorizer = TfidfVectorizer(min_df = 0.1, preprocessor = ' '.join)
    train_sparse_matrix = vectorizer.fit_transform(X_train)
    train_feature_names = vectorizer.get_feature_names()
   
    # Generate the Sparse Document-Term Matrix from the testing data
    vectorizer = TfidfVectorizer(preprocessor = ' '.join, vocabulary = train_feature_names)
    test_sparse_matrix = vectorizer.fit_transform(X_test)
    test_feature_names = vectorizer.get_feature_names()

    denselist_train = train_sparse_matrix.todense().tolist()
    denselist_test = test_sparse_matrix.todense().tolist() 

    # convert testing and training data to pandas data frames
    X_train = pd.DataFrame(denselist_train, columns=train_feature_names)
    X_test = pd.DataFrame(denselist_test, columns=test_feature_names)
    y_train = pd.DataFrame(y_train, columns=["label"])
    y_test = pd.DataFrame(y_test, columns=["label"])


    # TODO: remove? generate CSVs to view result
    # X_train.to_csv('training_data.csv')
    # X_test.to_csv('testing_data.csv')
    # y_train.to_csv('training_labels.csv')
    # y_test.to_csv('testing_labels.csv')

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    fake_news = parse_dataset("Fake_test.csv", "FAKE")
    real_news = parse_dataset("True_test.csv", "REAL")

    fake_news_all_tokens, fake_news_tokens_per_article = tokenize(fake_news, "fake_news")
    real_news_all_tokens, real_news_tokens_per_article = tokenize(real_news, "real_news")

    # join tokens and data together
    all_tokens = fake_news_all_tokens + real_news_all_tokens
    tokens_per_article = fake_news_tokens_per_article + real_news_tokens_per_article
    
    all_news = pd.concat([fake_news, real_news], axis=0)

    print("Preview of parsed data:")
    print(all_news)

    # join the data and pass it to split data
    X_train, X_test, y_train, y_test = split_and_preprocess(all_tokens,tokens_per_article, all_news)