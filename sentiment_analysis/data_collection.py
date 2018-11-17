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

