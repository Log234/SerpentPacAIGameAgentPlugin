from .memreader import MemoryReader

class Game:
    def __init__(self, **kwargs):
        self.reader = MemoryReader()
        
    def GetLives(self):
        lives = self.reader.GetLives()

        return {
            1077952576: 1,
            555761728: 2,
            555753760: 3
        }.get(lives, 4)

    def GetScore(self):
        return self.reader.GetScore()

    def IsPaused(self):
        paused = self.reader.GetPaused()

        return {
            0: True,
            1: False
        }[paused]

    def IsOver(self):
        game_over = self.reader.GetGameOver()
        
        return {
            0: True,
            1: False
        }[game_over]