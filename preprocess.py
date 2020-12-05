import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import random
import json

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

import sklearn
import sklearn.model_selection as ms
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = "./data/"
PREPROCESSED_DATA_DIR = DATA_DIR + "preprocessed/"

news_networks = ["reuters", "CNN", "BBC", "theguardian", "foxnews", "nbcnews", "washingtonpost", "cbcnews", "globalnews", "ctvnews"]
common_words = ["said", "would", "image", "via"]
stop_words = list(stopwords.words("english"))
stop_words.extend(news_networks)
stop_words.extend(common_words)


# parses the dataset from the csv file and sets the correct label
def parse_dataset(csv_file, label):
    print("Parsing news dataset from file: {}".format(csv_file))
    file_name = '{}{}'.format(DATA_DIR, csv_file)
    news = pd.read_csv(file_name, low_memory=False)

    # format date for each entry
    news["date"] = news["date"].apply(pd.to_datetime)

    # articles with no text should be cleaned from data
    news = news[news["text"] != ""] 

    # set the appropriate label
    print("Setting label for news dataset: {}".format(label))
    news["label"] = label

    return news

'''
 parses data scraped from online sources, labels them according to news source
 code used from here (reworked for this project): https://github.com/riag123/FakeNewsDeepLearning/blob/master/EDA_%2B_Pre_Processing.ipynb
'''
def parse_scraped_data(file_name):
    print("Parsing scraped news from file: {}".format(file_name))
    with open("{}{}".format(DATA_DIR, file_name)) as json_data:
        scraped_data = json.load(json_data)

        df = pd.DataFrame()
        for i, site in enumerate((list(scraped_data["newspapers"]))):
            articles = list(scraped_data["newspapers"][site]["articles"])
            if i == 0:
                df = pd.DataFrame.from_dict(articles)
                df["site"] = site
            else:
                new_df = pd.DataFrame.from_dict(articles)
                new_df["site"] = site
                df = pd.concat([df, new_df], ignore_index = True)

        is_fake_source = lambda source : "FAKE" if(
            source == "breitbart" or 
            source == "infowars" or 
            source == "theonion" or
            source == "thebeaverton" or 
            source == "prntly" or
            source == "nationalreport" or 
            source == "dailybuzzlive"
            ) else "REAL"
                
        scraped = df
        scraped["label"] = scraped["site"].apply(is_fake_source)
        scraped.drop(labels=["link", "site"], axis = 1, inplace=True)

        scraped["published"] = scraped["published"].apply(lambda x: x[0:10])
        scraped["published"] = scraped["published"].apply(pd.to_datetime)

        #Combine scraped with current datasets
        scraped.rename(columns={"published": "date"}, inplace=True)

        return scraped

# extracts individual words from news article text
def tokenize(news_data, name):
    all_tokens = [] #all tokens in fake_news articles
    article_tokens_list = [] #list of real_news articles, each in tokenized form
    
    print("Tokenizing {} dataset, this may take a few minutes...".format(name))
    for article in (news_data["text"]):
        words = word_tokenize(article)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words and word not in string.punctuation]
            
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        article_tokens_list.append(words)
        all_tokens.extend(words)

    return all_tokens, article_tokens_list

def save_to_csv(X_train, X_test, y_train, y_test):
    X_train.to_csv('{}training_data.csv'.format(PREPROCESSED_DATA_DIR))
    X_test.to_csv('{}testing_data.csv'.format(PREPROCESSED_DATA_DIR))
    y_train.to_csv('{}training_labels.csv'.format(PREPROCESSED_DATA_DIR))
    y_test.to_csv('{}testing_labels.csv'.format(PREPROCESSED_DATA_DIR))

def assign_id_to_article_tokens(vocabulary, tokens_per_article):
        # assign index to each word in vocabulary
    indexed_words_per_article = list()
    for article in tokens_per_article:
        indexed_words = []
        for word in article:
            if word in vocabulary:
                indexed_words.append(vocabulary[word])
        indexed_words_per_article.append(indexed_words)

    return indexed_words_per_article

# Split into training/testing data and preprocess 
def split_and_preprocess(vocabulary, tokens_per_article, all_news):   
    X = np.array(tokens_per_article, dtype="object")

    labels = all_news["label"]
    y = [1 if article == "FAKE" else 0 for article in labels]
    y = pd.DataFrame(y, columns=["label"])  

    #Create 80-30 train test split
    print("Splitting data: 70% training, 30% testing")
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state=0)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)

    # Generate the Sparse Document-Term Matrix from the training data
    vectorizer = CountVectorizer(min_df = 0.1, preprocessor = ' '.join)
    train_sparse_matrix = vectorizer.fit_transform(X_train)
    train_feature_names = vectorizer.get_feature_names()
   
    # Generate the Sparse Document-Term Matrix from the testing data
    vectorizer = CountVectorizer(preprocessor = ' '.join, vocabulary = train_feature_names)
    test_sparse_matrix = vectorizer.fit_transform(X_test)
    test_feature_names = vectorizer.get_feature_names()

    denselist_train = train_sparse_matrix.todense().tolist()
    denselist_test = test_sparse_matrix.todense().tolist() 

    # convert testing and training data to pandas data frames
    X_train = pd.DataFrame(denselist_train, columns=train_feature_names)
    X_test = pd.DataFrame(denselist_test, columns=test_feature_names)
    y_train = pd.DataFrame(y_train, columns=["label"])
    y_test = pd.DataFrame(y_test, columns=["label"])

    return X_train, X_test, y_train, y_test

def preprocess(use_full_dataset=False):
    print("\nPreprocessing of data...\n")

    if not use_full_dataset: 
        global PREPROCESSED_DATA_DIR
        PREPROCESSED_DATA_DIR = DATA_DIR + "test_preprocessed/"

    fake = "Fake.csv" if use_full_dataset else "Fake_test.csv"
    real = "True.csv" if use_full_dataset else "True_test.csv"
    fake_news = parse_dataset(fake, "FAKE")
    print("\nPreview of Fake news Dataset")
    print(fake_news)
    print()
    
    real_news = parse_dataset(real, "REAL")
    print("\nPreview of Real news Dataset")
    print(real_news)
    print()

    all_news = None
    scraped_data = None
    
    if use_full_dataset:
        # parse the scraped news articles
        scraped_data = parse_scraped_data("scraped_articles.json")
        print("\nPreview of Scraped news Dataset")
        print(len(scraped_data))
        print(scraped_data)
        print()
        # join data
        all_news = pd.concat([fake_news, real_news, scraped_data], axis=0)

        scraped_f = scraped_data[scraped_data["label"] == "FAKE"]
        fake_news = pd.concat([fake_news,scraped_f], axis=0, ignore_index=True)

        scraped_t = scraped_data[scraped_data["label"] == "REAL"]
        real_news = pd.concat([real_news,scraped_t], axis=0, ignore_index=True)
    else:
         # join data
        all_news = pd.concat([fake_news, real_news], axis=0)
    
    fake_news_all_tokens, fake_news_tokens_per_article = tokenize(fake_news, "fake_news")
    real_news_all_tokens, real_news_tokens_per_article = tokenize(real_news, "real_news")

    print()

    # join tokens
    all_tokens = fake_news_all_tokens + real_news_all_tokens
    tokens_per_article = fake_news_tokens_per_article + real_news_tokens_per_article

    # create the vocabulary of the dataset. Assign unique numerical id to each word
    vocabulary = {}
    for i, token in enumerate(all_tokens):
        if token not in vocabulary:
            vocabulary[token] = i + 1

    print()

    # Split and preprocess the data into training and testing data
    X_train, X_test, y_train, y_test = split_and_preprocess(vocabulary,tokens_per_article, all_news)

    print("\nPreview of training data:")
    print(X_train[:5])
    print(y_train[:5])
    print()

    return  X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()