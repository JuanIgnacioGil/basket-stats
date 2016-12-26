#!/usr/bin/python
# -*- encoding: utf-8 -*-

import requests
from pyquery import PyQuery as Pq
import pandas as pd
import numpy as np


class GameReader:

    def __init__(self, **kwargs):

        self.url = kwargs.get('url')
        self.doc = kwargs.get('doc')
        self.home = kwargs.get('home')
        self.away = kwargs.get('away')
        self.game_number = kwargs.get('game_number')
        self.date = kwargs.get('date')
        self.venue = kwargs.get('venue')
        self.audience = kwargs.get('audience')
        self.referees = kwargs.get('referees')
        self.partials = kwargs.get('partials')

    def summarize(self):

        text = ['{} {} - {} {}'.format(self.home['team'], self.home['points'],
                                     self.away['team'], self.away['points'])]
        text.append('Jornada {}, {}'.format(self.game_number, self.date))
        text.append('{}, {} espectadores'.format(self.venue, self.audience))
        text.append('Árbitros: {}, {}, {}'.format(self.referees[0], self.referees[1], self.referees[2]))
        text.append('Parciales: {}-{}, {}-{}, {}-{}, {}-{}'.format(self.partials[0, 0], self.partials[0, 1],
                                                                   self.partials[1, 0], self.partials[1, 1],
                                                                   self.partials[2, 0], self.partials[2, 1],
                                                                   self.partials[3, 0], self.partials[3, 1]))
        return '\n'.join(text)

    @staticmethod
    def read_url(url):

        response = requests.get(url)
        doc = Pq(response.content)
        return doc

    @classmethod
    def game_from_url(cls, url):

        # Read url
        doc = cls.read_url(url)

        # Result
        text = doc('div.titulopartidonew').text().split(' | ')
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

        # Date and audience
        text = doc('tr.estnegro').text().split(' | ')
        game_number = int(text[0][2:])
        date = pd.to_datetime(' '.join(text[1:3]))
        venue = text[3]
        audience = int(text[4].split('Público:')[1])

        # Referees
        text = doc('tr.estnaranja').text()
        delete_numbers = str.maketrans('', '', '1234567890|')
        referees = text.translate(delete_numbers).replace('Árb: ', '').split(', ')
        referees = [r.rstrip() for r in referees]
        # Partials
        delete_characters = str.maketrans('|', ' ', 'abcdefghijklmnñopqrstuvwxyz.,:áéíóúü')
        partials = text.lower().translate(delete_characters).split(' ')
        partials = np.array([int(p) for p in partials if p])
        partials = np.reshape(partials, (-1, 2))

        return cls(url=url, doc=doc, home=home, away=away, game_number=game_number,
                   date=date, venue=venue, audience=audience, referees=referees, partials=partials)

    @classmethod
    def read_statistics(cls, url):

        html = pd.read_html(url, attrs={"class": "estadisticasnew"}, header=1)

        table_data = dict(html[1])

        names = table_data['Nombre']

        # Separate the two teams
        separator = names[names == 'Nombre'].index[0]

        team1 = cls.make_team_table(table_data, separator, 1)
        team2 = cls.make_team_table(table_data, separator, 2)

        return team1, team2

    @classmethod
    def make_team_table(cls, table_data, separator, team):

        names = cls.slice_series(table_data['Nombre'], separator, team)
        minutes = cls.slice_series(table_data['Min'], separator, team)
        points = cls.slice_series(table_data['P'], separator, team)

        # 2 points
        # t2raw = team['T2']

        data = [('Name', names), ('Minutes', minutes), ('Points', points)]
        table = pd.DataFrame.from_items(data)
        table = table[table.Minutes.notnull()]
        table = table[table.Name != '200:0']
        return table

    @classmethod
    def slice_series(cls, series, separator, team):
        if team == 1:
            sliced = series[:separator]
        else:
            sliced = series[separator+1:]

        return sliced
