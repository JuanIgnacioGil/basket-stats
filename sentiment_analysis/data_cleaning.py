# -*- coding: utf-8 -*-

# Twitter data collection
# Original code from Yen-Chen Chou
# https://towardsdatascience.com/sentiment-analysis-on-nba-top-players-twitter-account-part2-tweets-data-cleaning-aa2cf99519b3

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk import tokenize

snowballstemmer = SnowballStemmer("english")
stopwords = stopwords.words('english')


def tweets_cleaner(text):
    semiclean_tweet = []
    for tweet in text:
        tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"[0-9]*", "", tweet)
        tweet = re.sub(r"(”|“|-|\+|`|#|,|;|\|)*", "", tweet)
        tweet = re.sub(r"&amp", "", tweet)
        tweet = tweet.lower()
        semiclean_tweet.append(tweet)

    return semiclean_tweet


def tokenization_and_stem(semiclean_tweet):
    total_token_ls = []
    total_snowballstemmer_token_ls = []

    for sentence in semiclean_tweet:
        token_ls = []
        snowballstemmer_token_ls = []
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in stopwords:
                token_ls.append(token)
                snowballstemmer_token_ls.append(snowballstemmer.stem(token))
        total_token_ls.append(token_ls)
        total_snowballstemmer_token_ls.append(snowballstemmer_token_ls)

    return total_token_ls, total_snowballstemmer_token_ls


def back_to_clean_sent(token_ls):
    """
    In order to perform sentiment analysis,
    here we put the words back into sentences.
    """
    clean_sent_ls = []
    for word_ls in token_ls:
        clean_sent = ""
        for word in word_ls:
            clean_sent += (word + " ")
        clean_sent_ls.append(clean_sent)
    return clean_sent_ls

