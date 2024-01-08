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

# 結果表示
def show_dist(result_score):
    import random
    import collections
    import pandas as pd
    import numpy as np
    import json
    import copy
    from matplotlib import pyplot as plt
    from matplotlib import font_manager


    # ********** 日本語フォントの設定 ************
    font_path = "BIZUDPGothic-Regular.ttf" # <- どこからか入手して、作業ディレクトリに置く。
    font_prop = font_manager.FontProperties(fname=font_path)
    font_prop.set_style('normal')
    font_prop.set_weight('light') 
    font_prop.set_size('12') # 文字サイズ

    # plt.rcParams["font.family"] = "BIZ UDPGothic"   # 使用するフォント
    # plt.rcParams["font.size"] = 12                 # 文字の大きさ

    bins = np.linspace(0, 330, 34)
    scores = pd.Series(result_score)
    freq = scores.value_counts(bins=bins, sort=False)
    #print(freq)
    class_value = (bins[:-1] + bins[1:]) / 2  # 階級値
    rel_freq = freq / scores.count()  # 相対度数
    cum_freq = freq.cumsum()  # 累積度数
    rel_cum_freq = rel_freq.cumsum()  # 相対累積度数
    dist = pd.DataFrame(
        {
            "階級値": class_value,
            "度数": freq,
            "相対度数": rel_freq,
            "累積度数": cum_freq,
            "相対累積度数": rel_cum_freq,
        },
        index=freq.index
    )
    fig, ax1 = plt.subplots()
    fig.suptitle('スコア度数分布', fontsize=12, fontproperties=font_prop)
    dist.plot.bar(x="階級値", y="度数", ax=ax1, width=1, ec="k", lw=1)
    hans, labs = ax1.get_legend_handles_labels()
    ax1.legend(handles=hans, labels=labs)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(dist)), dist["相対累積度数"], "--o", color="k")
    ax2.set_ylabel("累積相対度数", fontproperties=font_prop)
    fig.savefig("dist.png")
    #fig.show()
    #input()

def func2(player):
    game = Game(player, autoplay=True)
    # print(game)
    sb = game.log.loc[12, 'scoreBoard']
    return sb.getSum()

def benchmark():
    player = bot.PlayerAI_full_v2(fn='./trainedBots/PlayerAI_full_v2-nGame_3100.pick')

    result_score = []
    from concurrent.futures import ProcessPoolExecutor #マルチプロセス
    from concurrent.futures import ThreadPoolExecutor #マルチスレッド
    import time

    start = time.time()

    with ProcessPoolExecutor(max_workers=40) as executor:  #マルチプロセス
    #with ThreadPoolExecutor(max_workers=40) as executor: #マルチスレッド
        for i in range(0, 10000):
            future = executor.submit(func2, player)    
            result_score.append(future.result())

    print (time.time()-start)
    show_dist(result_score)

def play():
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
if __name__ == '__main__':
    benchmark()

