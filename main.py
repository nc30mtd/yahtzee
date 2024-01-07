import numpy as np

import sys; sys.path.append('./lib/')
from comfct.debug import lp
from yahtzee import Dice, ScoreBoard, Game
from sklearn.neural_network import MLPRegressor
import bot
import datetime as dt
import pickle

# Benchmark games should not overlap with training games
BENCHMARK_SEED = 618225912

BENCHMARK_SEED = 618225912
if __name__ == '__main__':
    #player = bot.PlayerAI_full_v2(fn='./trainedBots/PlayerAI_full_v2-nGame8000.pick')
    player = bot.PlayerAI_full_v2(fn=None)
    m, s = player.benchmark(seed=BENCHMARK_SEED)
    print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))

    lstPlayersAlg = []
    lstPlayersAlg += [
            bot.PlayerAlg_oneShot_greedy(),
            bot.PlayerAlg_full_greedy(),
                    ]
    for player in lstPlayersAlg:
        m, s = player.benchmark(seed=BENCHMARK_SEED)
        print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))