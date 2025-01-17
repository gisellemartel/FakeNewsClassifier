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
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)
vocabulary = {}

DATA_DIR = "./data/real_data/"
PREPROCESSED_DATA_DIR = DATA_DIR + "preprocessed/"

common_words = ["said", "would", "image", "via", "reuters"]
stop_words = list(stopwords.words("english"))
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

    news.drop(["subject"], axis=1)

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

def save_to_csv(X_train, X_test, y_train, y_test, type):
    X_train.to_csv('{}{}_training_data.csv'.format(PREPROCESSED_DATA_DIR, type))
    X_test.to_csv('{}{}_testing_data.csv'.format(PREPROCESSED_DATA_DIR,type))
    y_train.to_csv('{}{}_training_labels.csv'.format(PREPROCESSED_DATA_DIR, type))
    y_test.to_csv('{}{}_testing_labels.csv'.format(PREPROCESSED_DATA_DIR, type))

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

def preprocess_single_item(article, for_neural_net=False):
    article = article["text"]
    words = word_tokenize(article)

    tokens = []

    for word in words:
        if word in vocabulary:
            tokens.append(np.int64(vocabulary[word]))
    return tokens

# Split into training/testing data and preprocess 
def split_and_preprocess(vocabulary, tokens_per_article, all_news):   
    X = np.array(tokens_per_article, dtype="object")

    labels = all_news["label"]
    y = [1 if article == "FAKE" else 0 for article in labels]
    y = pd.DataFrame(y, columns=["label"])  

    #Create 80-30 train test split
    print("Splitting copy of data for traditional ML models: 70% training, 30% testing")
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state=0)

    # Generate the Sparse Document-Term Matrix from the training data
    print("Vectorizing copy of training and testing data for traditional ML models using IDF Vectorizer\n")
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

    return X_train, X_test, y_train, y_test

# Split into training/testing data and preprocess for CNN (uses CountVectorizer instead)
def split_and_preprocess_cnn(vocabulary, tokens_per_article, all_news):   
    X = np.array(tokens_per_article, dtype="object")

    labels = all_news["label"]
    y = [1 if article == "FAKE" else 0 for article in labels]
    y = pd.DataFrame(y, columns=["label"])  

    #Create 80-30 train test split
    print("Splitting copy of data for CNN model: 70% training, 30% testing")
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state=0)

    # Generate the Sparse Document-Term Matrix from the training data
    print("Vectorizing copy of training and testing data for CNN model using Count Vectorizer\n")
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
        global DATA_DIR
        DATA_DIR = "./data/mock_data/"

    fake = "kaggle_raw/Fake.csv"
    real = "kaggle_raw/True.csv"
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
    
    # parse the scraped news articles
    scraped_data = parse_scraped_data("scraped_raw/scraped_articles.json")
    print("\nPreview of Scraped news Dataset")
    print(len(scraped_data))
    print(scraped_data)

    scraped_f = scraped_data[scraped_data["label"] == "FAKE"]
    print("\n\nCombining FAKE scraped articles with fake_news data...")
    fake_news = pd.concat([fake_news,scraped_f], axis=0)

    scraped_t = scraped_data[scraped_data["label"] == "REAL"]
    print("Combining REAL scraped articles with real_news data...")
    real_news = pd.concat([real_news,scraped_t], axis=0)

    # join data
    all_news = pd.concat([fake_news, real_news], axis=0)

    print("\nSize of the cleaned data to be tokenized and split: {}\n".format(len(all_news)))
    
    fake_news_all_tokens, fake_news_tokens_per_article = tokenize(fake_news, "fake_news")
    real_news_all_tokens, real_news_tokens_per_article = tokenize(real_news, "real_news")

    print()

    # join tokens
    all_tokens = fake_news_all_tokens + real_news_all_tokens
    tokens_per_article = fake_news_tokens_per_article + real_news_tokens_per_article

    # create the vocabulary of the dataset. Assign unique numerical id to each word
    global vocabulary
    for i, token in enumerate(all_tokens):
        if token not in vocabulary:
            vocabulary[token] = i + 1

    print("\nCombining real_news and fake_news datasets...")

    # Split and preprocess the data into training and testing data
    ml_data = split_and_preprocess(vocabulary,tokens_per_article, all_news)
    cnn_data = split_and_preprocess_cnn(vocabulary,tokens_per_article, all_news)

    print("\nPreview of ML training data:")
    print(ml_data[0][:5])
    print(ml_data[2][:5])
    print()

    print("\nPreview of CNN training data:")
    print(cnn_data[0][:5])
    print(cnn_data[2][:5])
    print()

    return ml_data, cnn_data 

if __name__ == "__main__":
    ml_data, cnn_data = preprocess()