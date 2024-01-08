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

def get_rolls(dices=5):
    rolls = []
    flag = True
    while(flag):
        print("input rolls(dices):",dices)
        text = input()
        for j in range(len(text)):
            v = int(text[j])
            if(v > 0 and v < 7):
                rolls.append(v)
        if(len(rolls) == dices):
            flag = False

    print(rolls)

    return rolls

if __name__ == '__main__':
    # #player = bot.PlayerAI_full_v2(fn='./trainedBots/PlayerAI_full_v2-nGame8000.pick')
    player = bot.PlayerAI_full_v2(fn='./trainedBots/PlayerAI_full_v2-nGame200.pick')
    # m, s = player.benchmark(seed=BENCHMARK_SEED)
    # print('\t{:50} {:.1f} +/- {:.1f}'.format(player.name+':', m, s))

    game = Game(player, autoplay=False)
    
    
    sb = ScoreBoard()

    for turn in range(0, 12):
        rolls = get_rolls()
        prevSb = sb.copy()
        
        # 1投目
        dice0 = Dice(rolls)
        deci0, info0 = player.choose_reroll(sb, dice0, 0)
        # lp(dice0)
        # lp(deci0)
        # lp(info0)
        # lp(player.choose_reroll(sb, dice0, 0))
        # lp(type(player.choose_reroll(sb, dice0, 0)))
        # 残すサイコロ列挙する
        print("---")
        for i in range(len(dice0.vals)):
            if(deci0[i]):
                print(dice0.vals[i], "")
            else:
                print(dice0.vals[i], "残す")
        print("---")
        print("振り直すサイコロの数:", deci0.count(True))
        rolls = get_rolls(deci0.count(True))
        for i in range(len(deci0)):
            if(deci0[i] == False):
                rolls.append(dice0.vals[i])
        print(rolls)
        dice1 = Dice(rolls)
        lp(dice1)

        # 2投目
        deci1, info1 = player.choose_reroll(sb, dice1, 1)
        
        # 残すサイコロ列挙する
        print("---")
        for i in range(len(dice1.vals)):
            if(deci1[i]):
                print(dice1.vals[i], "")
            else:
                print(dice1.vals[i], "残す")
        print("---")
        print("振り直すサイコロの数:", deci1.count(True))
        rolls = get_rolls(deci1.count(True))
        for i in range(len(deci1)):
            if(deci1[i] == False):
                rolls.append(dice1.vals[i])
        print(rolls)
        dice2 = Dice(rolls)
        lp(dice1)


        # choose cat
        deci2, info2 = player.choose_cat(sb, dice2)
        sb.add(dice2, deci2)
#            lp(self.log)
        game.log.loc[turn, :] = [
                prevSb,
                dice0, deci0, info0, dice1, deci1, info1,
                dice2, deci2, info2]
        lp(deci2)
        lp(info2)
        print(sb)

