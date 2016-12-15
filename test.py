from ACBstats.game_reader import GameReader

url1 = 'http://www.acb.com/fichas/LACB61096.php'
url2 = 'http://www.acb.com/fichas/LACB61092.php'

game = GameReader(url2)
print(game.read_game())

