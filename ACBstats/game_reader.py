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

        #Statistics
        stats = cls.read_statistics(url)

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
        t2raw = cls.slice_series(table_data['T2'], separator, team)
        t3raw = cls.slice_series(table_data['T3'], separator, team)
        t1raw = cls.slice_series(table_data['T1'], separator, team)
        reb = cls.slice_series(table_data['D+O'], separator, team)
        assists = cls.slice_series(table_data['A'], separator, team)
        steals = cls.slice_series(table_data['BR'], separator, team)
        turnovers = cls.slice_series(table_data['BP'], separator, team)
        blocks = cls.slice_series(table_data['F'], separator, team)
        blocks_received = cls.slice_series(table_data['C.1'], separator, team)
        dunks = cls.slice_series(table_data['M'], separator, team)
        counterattacks = cls.slice_series(table_data['C'], separator, team)
        fouls = cls.slice_series(table_data['C.2'], separator, team)
        fouls_received = cls.slice_series(table_data['F.1'], separator, team)
        plus_minus = cls.slice_series(table_data['+/-'], separator, team)
        acb_valuation = cls.slice_series(table_data['V'], separator, team)

        data = [('Name', names), ('Time', minutes), ('Points', points), ('2p_goals', t2raw),
                ('2p_attempts', t2raw), ('3p_goals', t3raw), ('3p_attempts', t3raw),
                ('1p_goals', t1raw), ('1p_attempts', t1raw),
                ('Defensive_rebounds', reb), ('Offensive_rebounds', reb), ('Assists', assists),
                ('Steals', steals), ('Turnovers', turnovers), ('Blocks', blocks),
                ('Blocks_received', blocks_received), ('Dunks', dunks), ('Counterattacks', counterattacks),
                ('Fouls', fouls), ('Fouls_received', fouls_received), ('Plus_minus', plus_minus),
                ('ACB_valuation', acb_valuation)]
        table = pd.DataFrame.from_items(data)
        table = table[table.Time.notnull()]
        table = table[table.Name != '200:0']

        # Minutes as timedeltas
        table['Time'] = table['Time'].apply(lambda x: pd.to_timedelta(':'.join(['0', x])))

        # T2
        table['2p_attempts'] = table['2p_goals'].apply(lambda x: int(x.split('/')[1]))
        table['2p_goals'] = table['2p_goals'].apply(lambda x: int(x.split('/')[0]))

        # T3
        table['3p_attempts'] = table['3p_goals'].apply(lambda x: int(x.split('/')[1]))
        table['3p_goals'] = table['3p_goals'].apply(lambda x: int(x.split('/')[0]))

        # T1
        table['1p_attempts'] = table['1p_goals'].apply(lambda x: int(x.split('/')[1]))
        table['1p_goals'] = table['1p_goals'].apply(lambda x: int(x.split('/')[0]))

        # Rebounds
        table['Defensive_rebounds'] = table['Defensive_rebounds'].apply(lambda x: int(x.split('+')[0]))
        table['Offensive_rebounds'] = table['Offensive_rebounds'].apply(lambda x: int(x.split('+')[1]))



        return table


    @classmethod
    def slice_series(cls, series, separator, team):
        if team == 1:
            sliced = series[:separator]
        else:
            sliced = series[separator+1:]

        return sliced
