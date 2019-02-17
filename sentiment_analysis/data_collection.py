# -*- coding: utf-8 -*-

# Twitter data collection
# Original code from Yen-Chen Chou
# https://towardsdatascience.com/do-tweets-from-nba-leading-players-have-correlations-with-their-performance-7358c79aa216

import os
import datetime
import json
import numpy as np
import nltk
import pandas as pd
import requests
import re
import tweepy
from bs4 import BeautifulSoup
from tweepy import OAuthHandler

DEFAULT_API_KEYS_JSON = os.path.join(os.path.expanduser("~"), '.api_keys.json')


def login_into_twitter(api_keys_json=DEFAULT_API_KEYS_JSON):
    """
    Login into twitter using the keys stored in a json file

    Parameters
    ----------
    api_keys_json: str

    Returns
    -------
    tweepy.API
    """
    # The code expects a json file with the Twitter api keys
    api_keys_file = open(api_keys_json, 'r')
    api_keys = json.loads(api_keys_file.read())

    # Login into twitter
    auth = OAuthHandler(api_keys["twitter_api_keys"]["consumer_key"], api_keys["twitter_api_keys"]["consumer_secret"])
    auth.set_access_token(api_keys["twitter_api_keys"]["access_token"], api_keys["twitter_api_keys"]["access_secret"])
    api = tweepy.API(auth)

    return api


def get_best_players():
    """
    Gets the 10 best players of a season using the ESPN ran,ming

    Parameters
    ----------

    Returns
    -------
    list of str

    """

    # FIXME: The url is a static story. Look for a dynamic source where we can select season and number of players
    url_player = 'http://www.espn.com/nba/story/_/id/24668720/nbarank-2018-19-1-10-best-players-season'

    # Find the best player on each team according to ESPN.com
    players = requests.get(url_player)
    player = BeautifulSoup(players.text, "lxml")
    player_list = []
    for h2 in player.findAll("h2"):
        for name in h2.findAll({"a":"href"}):
            player_list.append(name.get_text())

    return player_list


def get_url_player_stats():
    """
    Gets the url for the players stats from www.basketball-reference.com

    Now, this part is about extracting the stats that players performed during the court.
    We are taking stats between 2016–2019 season but here we run into a very serious issue.
    Each URL will be different according to each player and there is no any fixed pattern for the URL.
    Thus, the only way I come up with right now is to lists all the URL of each player.
    After that, we time the year according to each player’s URL and get the whole URLs we need.

    FIXME: Search for a way to automatize this


    Returns
    -------
    list of str

    """

    url_player_stats_ls = [
                        "https://www.basketball-reference.com/players/a/antetgi01/gamelog/",
                        "https://www.basketball-reference.com/players/h/hardeja01/gamelog/",
                        "https://www.basketball-reference.com/players/g/goberru01/gamelog/",
                        "https://www.basketball-reference.com/players/g/georgpa01/gamelog/",
                        "https://www.basketball-reference.com/players/d/duranke01/gamelog/",
                        "https://www.basketball-reference.com/players/d/davisan02/gamelog/",
                        "https://www.basketball-reference.com/players/j/jokicni01/gamelog/",
                        "https://www.basketball-reference.com/players/l/lillada01/gamelog/",
                        "https://www.basketball-reference.com/players/t/townska01/gamelog/",
                        "https://www.basketball-reference.com/players/e/embiijo01/gamelog/"
                       ]
    years = ["2019", "2018", "2017", "2016"]
    url_player_stats = []
    for i in url_player_stats_ls:
        url_stats = []
        for j in years:
            url_stats.append(i+j)
        url_player_stats.append(url_stats)

    return url_player_stats


def get_one_player_stats(url_player_stats):
    """
    Gets stats for a single player

    After generates the URLs, the send them into the following function. Here we collect the game time, game result,
    field goals(FG), field goals attempted(FGA), and 3-point field goals(FG3).
    Also, the reason why we have a lot of “if…else” statements is that reference.com marks the record with the
     specific words if the player did not play the game such as “Inactive” or “Did Not Play”.



    Parameters
    ----------
    url_player_stats

    Returns
    -------
    list

    """
    stats_list = []

    for url in url_player_stats:
        stats = requests.get(url)
        stats = BeautifulSoup(stats.text, "lxml")
        game_stats = []

        for table in stats.findAll("table", {"id": "pgl_basic"}):
            for tr in table.findAll(
                    "td",
                    {"data-stat": ["date_game", "game_result", "reason", "fg", "fga", "fg3"]}
            ):  # "fga", "fg3"
                if tr.get_text() == "Inactive":
                    game_stats.extend([None]*3)
                elif tr.get_text() == "Did Not Dress":
                    game_stats.extend([None]*3)
                elif tr.get_text() == "Did Not Play":
                    game_stats.extend([None]*3)
                elif tr.get_text() == "Not With Team":
                    game_stats.extend([None]*3)
                elif tr.get_text() == "Player Suspended":
                    game_stats.extend([None]*3)
                else:
                    game_stats.append(tr.get_text())

        stats_list.extend(game_stats)

    return stats_list


def generate_all_player_stats(url_player_stats):
    """
    Generates list of dataframes with stats

    Then, we reshape the list type data into data frames. Here we also convert data types into the correct format and
    minimize the data size through assigning the “int” type to “int16”. Recall that we are measuring the performance of
     the players and their tweets. The performance we define is the Effective Flied Goal (eFG) and the formula
     is (FG + 0.5 * 3P) / FGA.

    Parameters
    ----------
    url_player_stats: list of str

    Returns
    -------
    list of pandas.DataFrame

    """
    all_player_stats_ls = []

    for each_player in url_player_stats:
        player_stats = get_one_player_stats(each_player)
        stat_df = pd.DataFrame(np.array(player_stats).reshape((len(player_stats) // 5), 5),
                               columns=["Date", "Result", "FG", "FGA", "FG3"])
        stat_df["Date"] = pd.to_datetime(stat_df["Date"])
        stat_df["Result"] = pd.Series(map(lambda x: x[0], stat_df["Result"]))
        stat_df.dropna(axis=0, how="any", inplace=True)
        stat_df.sort_values(by='Date', inplace=True, ascending=False)
        stat_df[["FG", "FGA", "FG3"]] = stat_df[["FG", "FGA", "FG3"]].astype("int16")
        stat_df["eFG"] = round((stat_df["FG"] + 0.5 * stat_df["FG3"]) / stat_df["FGA"], 3)
        all_player_stats_ls.append(stat_df)

        return all_player_stats_ls


def get_twitter_accounts(player_list):
    """
    Get players account from basketball-reference.com

    Parameters
    ----------
    player_list: list of str

    Returns
    -------
    dict
    """
    # Extract nba player list and their twitter account from basketball-reference.com
    url_twitterlink = "https://www.basketball-reference.com/friv/twitter.html"
    tweets = requests.get(url_twitterlink)
    tweet = BeautifulSoup(tweets.text, "lxml")
    player_list2 = []
    tw_list = []

    for td in tweet.findAll("td", {"data-stat": "player"}):
        for a in td.findAll("a"):
            player_list2.append(a.get_text())

    for td in tweet.findAll("td", {"data-stat": "twitter_id"}):
        for a in td.findAll("a"):
            tw_list.append(a.get_text())

    # Merge the two data and match the players' twitter accout
    tw_account = {}
    final_tw_account = {}
    for key, value in zip(player_list2, tw_list):
        tw_account[key] = value
    for p, t in tw_account.items():
        if p in player_list:
            final_tw_account[p] = t

    return final_tw_account


def get_all_tweet(api, final_tw_account):
    """
    Gets tweets from the players

    The function here looks complex, but actually, it is just about getting the tweets, number of followers,
    number of friends, number of favorites, and combining the stats of their perfamance and the players’ name into a
    giant data frame.

    Note:
    Remember to add tweet_mode = ‘extended’ for the complete amount of characters or you only get the first 140
    characters on each tweet. Also, only the tweepy.Cursor( ).items(1000) api can have access to more than 200 tweets.
    The api.user_timeline(id = ID, count = 200) api has this limitation.

    Parameters
    ----------
    api: tweepy.API
    final_tw_account: str

    Returns
    -------
    pandas.DataFrame
    """
    all_nba_tweet = []
    all_creat_time = []
    followers = []
    friends = []
    favourites = []

    for ID in final_tw_account.values():
        tweet_list = []
        creat_time_list = []

        for twt in tweepy.Cursor(api.user_timeline, id=ID, tweet_mode='extended').items(1000):
            if twt.created_at > datetime.datetime(2016, 11, 1, 0, 0, 0):  # extract time after
                try:
                    tweet_list.append(twt.retweeted_status.full_text)
                except AttributeError:
                    tweet_list.append(twt.full_text)
                creat_time_list.append(twt.created_at)

        all_nba_tweet.append(tweet_list)
        all_creat_time.append(creat_time_list)
        followers.append(twt.user.followers_count)
        friends.append(twt.user.friends_count)
        favourites.append(twt.user.favourites_count)

    df_tweet = pd.DataFrame.from_records(
        zip(all_nba_tweet, all_creat_time, followers, friends, favourites, final_tw_account.keys()),
        columns=["Tweets", "Create_time", "Followers", "Friends", "Favourites", 'Name']
    )

    return df_tweet
