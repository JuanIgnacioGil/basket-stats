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
                        "https://www.basketball-reference.com/players/c/curryst01/gamelog/",
                        "https://www.basketball-reference.com/players/d/davisan02/gamelog/",
                        "https://www.basketball-reference.com/players/d/duranke01/gamelog/",
                        "https://www.basketball-reference.com/players/e/embiijo01/gamelog/",
                        "https://www.basketball-reference.com/players/h/hardeja01/gamelog/",
                        "https://www.basketball-reference.com/players/j/jamesle01/gamelog/",
                        "https://www.basketball-reference.com/players/l/leonaka01/gamelog/",
                        "https://www.basketball-reference.com/players/l/lillada01/gamelog/",
                        "https://www.basketball-reference.com/players/w/westbru01/gamelog/"
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
