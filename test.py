from ACBstats.game_reader import GameReader as Gr

url1 = 'http://www.acb.com/fichas/LACB61096.php'
url2 = 'http://www.acb.com/fichas/LACB61092.php'
url3 = 'http://www.acb.com/fichas/LACB61102.php'

game = Gr.game_from_url(url1)
print(game.summarize())
print(game.home)
print(game.away['coach'])