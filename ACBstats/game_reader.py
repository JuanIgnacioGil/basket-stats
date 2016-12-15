#!/usr/bin/python
# -*- encoding: utf-8 -*-

import requests
from pyquery import PyQuery as Pq


class GameReader:

    def __init__(self, url):

        self.url = url
        self.doc = self.read_url()
        ha = self.read_game()
        self.home = ha['home']
        self.away = ha['away']

    def __repr__(self):

        repr_string = '{} {} - {} {}'.format(self.home['team'], self.home['points'],
                                     self.away['team'], self.away['points'])
        return repr_string

    def read_url(self):

        response = requests.get(self.url)
        doc = Pq(response.content)
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

        return dict(home=home, away=away)

