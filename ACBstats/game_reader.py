#!/usr/bin/python
# -*- encoding: utf-8 -*-

import requests
from pyquery import PyQuery as pq

class GameReader:

    def __init__(self, url):

        self.url = url
        self.doc = self.read_url()


    def read_url(self):

        response = requests.get(self.url)
        doc = pq(response.content)
        return doc

    def read_game(self):

        text = self.doc('div.titulopartidonew').text().split(' | ')
        team1 = text[0]
        vraw = text[1].split(' ')
        team2 = ' '.join(vraw[:-1])
        points1 = int(vraw[-1])
        points2 = int(text[-1])

        home = dict(team=team1, points=points1, victory=False)
        away = dict(team=team2, points=points2, victory=False)

        if points1 > points2:
            home['victory'] = True
        else:
            away['victory'] = True

        return home, away

