from ACBstats.game_reader import GameReader as Gr

url1 = 'http://www.acb.com/fichas/LACB61096.php'
url2 = 'http://www.acb.com/fichas/LACB61092.php'

game = Gr.game_from_url(url2)
print(game)

