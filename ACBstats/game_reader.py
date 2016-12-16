#!/usr/bin/python
# -*- encoding: utf-8 -*-

import requests
from pyquery import PyQuery as Pq
import pandas as pd

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

    def __repr__(self):

        text = ['{} {} - {} {}'.format(self.home['team'], self.home['points'],
                                     self.away['team'], self.away['points'])]
        text.append('Jornada {}, {}'.format(self.game_number, self.date))
        text.append('{}, {} espectadores'.format(self.venue, self.audience))
        text.append('Árbitros: {}, {}, {}'.format(self.referees[0], self.referees[1], self.referees[2]))
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

        # Referees and partials
        text = doc('tr.estnaranja').text()
        delete_numbers = str.maketrans('','','1234567890|')
        referees = text.translate(delete_numbers).replace('Árb: ', '').split(', ')
        referees = [r.rstrip() for r in referees]

        return cls(url = url, doc=doc, home=home, away=away, game_number=game_number,
                   date=date, venue=venue, audience=audience, referees=referees)


