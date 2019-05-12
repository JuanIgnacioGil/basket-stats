# -*- coding: utf-8 -*-

# Sentiment analysis
# Original code from Yen-Chen Chou
# https://towardsdatascience.com/sentiment-analysis-on-nba-top-players-twitter-account-part3-sentiment-analysis-clustering-5e5dcd4d690f

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


def sentiment_analysis(insert_processed_sentence):
    """
    Do the sentiment analysis!!

    Parameters
    ----------
    insert_processed_sentence: list

    Returns
    -------

    """
    sentiment = [0]*len(insert_processed_sentence)

    for num, player_sent in enumerate(insert_processed_sentence):
        ll = []

        for sentence in player_sent:
            ss = analyser.polarity_scores(sentence)
            ll.append(ss)
        sentiment[num] = ll

    return sentiment


def proccess_sentiment(sentiment_token, sentiment_snowstemmed, df):
    """
    We transform the compound score into “positive”, “neutral”, and “negative” according to the previous definition
    and put into the data frame.

    Parameters
    ----------
    sentiment_token: list of dict
    sentiment_snowstemmed: list of dict
    df: pandas.DataFrame

    Returns
    -------
    pandas.DataFrame

    """
    new_df = pd.DataFrame()
    i = 0
    for senti_token, snow_stem in zip(sentiment_token, sentiment_snowstemmed):
        # senti_token_pos = [], senti_token_neu = [], senti_token_neg = []
        senti_token_compound = []
        senti_stem_compound = []
        sentiment_result_token = []
        sentiment_result_stem = []

        for s1 in senti_token:
            senti_token_compound.append(s1["compound"])
            if s1["compound"] >= 0.05:
                sentiment_result_token.append("positive")
            elif s1["compound"] <= -0.05:
                sentiment_result_token.append("negative")
            else:
                sentiment_result_token.append("neutral")

        for s2 in snow_stem:
            senti_stem_compound.append(s2["compound"])
            if s2["compound"] >= 0.05:
                sentiment_result_stem.append("positive")
            elif s2["compound"] <= -0.05:
                sentiment_result_stem.append("negative")
            else:
                sentiment_result_stem.append("neutral")

        tw_df = pd.DataFrame.from_dict({"Tweets": df["Tweets"][i],
                                        "Create_time": df["Create_time"][i],
                                        "Name": df["Name"][i],
                                        "sentiment_token_compound": senti_token_compound,
                                        "sentiment_result_token": sentiment_result_token,
                                        "sentiment_stem_compound": senti_stem_compound,
                                        "sentiment_result_stem": sentiment_result_stem})
        new_df = new_df.append(tw_df)
        i += 1

    return new_df


def fit_vectorizer(sentence_snowstemmeed):
    """

    Before begin the clustering, we must create the TF-IDF matrix and model first.
     TF-IDF refers to Term Frequency–Inverse Document Frequency and it is a product of two statistics that
     reflect how important a word is to a document in a collection or corpus:

        Term frequency: The weight of a term that occurs in a document is simply proportional to the term frequency.
        Inverse document frequency: The specificity of a term can be quantified as an inverse function of the number of
            documents in which it occurs.

    In order to make things simple, here we only use the stemmed sentence and product the TF-IDF matrix.
    The Tweets_mix is just a list that gathering all the stemmed sentences and only doing so will we able to generate
    the TF-IDF matrix.

    Parameters
    ----------
    sentence_snowstemmeed: list

    Returns
    -------

    """
    tweets_mix = []

    for i, row in enumerate(sentence_snowstemmeed):
        all_row = ""
        for sent in row:
            all_row += sent
        tweets_mix.append(all_row)

    tfidf_model = TfidfVectorizer(max_df=0.8, max_features=1000,
                                  min_df=0.2, stop_words='english',
                                  use_idf=True, tokenizer=None, ngram_range=(1, 1))

    tfidf_matrix = tfidf_model.fit_transform(tweets_mix)  # fit the vectorizer to synopses

    return tfidf_model, tfidf_matrix


def fit_kmeans(tfidf_matrix, df):
    """
    Fit kmeans clusters to sentiment indicators

    Parameters
    ----------
    tfidf_matrix
    df

    Returns
    -------

    """

    # First, we implement K-means for the clustering and we pick 3 clusters in this article:
    # implement kmeans
    num_clusters = 3
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    # Next, output the players and their corresponding clusters:
    # create DataFrame films from all of the input files.
    players_dict = {'Name': df["Name"].tolist(), 'Cluster': clusters}
    clusters_df = pd.DataFrame(players_dict, index=[clusters])

    return clusters, clusters_df, km


def find_common_words(sentence_tokenized, sentence_snowstemmeed, tfidf_model, km, clusters_df):
    """
    Remember that we try to cluster the players to see if each group of players have similar behavior when choosing
    their vocabularies. In other words, we need to know the word sets for each cluster. However, according to the
    previous process, we had used stemmed words to generate our TF-IDF matrix, which means thats the word sets for
    each clusters are also stemmed words.

    For example, that’s say if cluster one shows that “famli” and “happi” for their most frequent words, which we know
    “famli” equals to “family and “happi” equals to “happy”; while it is not always straightforward for us.

    Hence, we have to translate back to the words we are much familiar with. To do so, we create a dictionary such
    as {“happi” : “happy”} so that when we calculate the frequent stemmed word we can trace back the original one
    immediately. The docs_tokenized and docs_snowstemmed are lists that contain tokenized and stemmed words according
    to each player, and we are going to use them for our original word translation.

    Then, we output the corresponding clusters of each players and the words that frequently used by them:

    Parameters
    ----------
    sentence_tokenized
    sentence_snowstemmeed
    tfidf_model
    km
    clusters_df

    Returns
    -------

    """
    l1, l2 = "", ""
    for player in sentence_tokenized:
        for sent in player:
            l1 += sent
    docs_tokenized = l1.split()

    for player in sentence_snowstemmeed:
        for sent in player:
            l2 += sent
    docs_snowstemmed = l2.split()

    vocab_frame_dict = {docs_snowstemmed[x]: docs_tokenized[x] for x in range(len(docs_snowstemmed))}
    tf_selected_words = tfidf_model.get_feature_names()

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    cluster_keywords_summary = {}

    for i in range(km.n_clusters):
        print("Cluster " + str(i) + " words: ", end='')
        cluster_keywords_summary[i] = []

        for ind in order_centroids[i, :5]:  # replace 5 with n words per cluster
            cluster_keywords_summary[i].append(vocab_frame_dict[tf_selected_words[ind]])
            print(vocab_frame_dict[tf_selected_words[ind]] + ",", end='')

        cluster_nba = clusters_df.loc[i]['Name'].values
        print("\n", ", ".join(cluster_nba), "\n")

    return vocab_frame_dict, tf_selected_words,cluster_keywords_summary, cluster_nba


