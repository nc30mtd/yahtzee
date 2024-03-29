#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:53:36 2019

@author: user
"""
# --- imports
import numpy as np
#np.random.seed(0)
import pandas as pd
from comfct.debug import lp
from copy import deepcopy
from itertools import product

MAX_TURNS = 12

def roll_dice(nDice=5):
    #dice = np.empty(shape=nDice,dtype=int)
    dice=0
    for ii in range(0,nDice):
        #dice[ii]=random.randint(1,6)
        dice*=10
        dice+=np.random.randint(1,6)
    return dice

class Dice:
    """Stores the result of 0 to 5 dice"""
    def __init__(self, vals=None):
        if vals is None:
            self.vals = np.sort(np.random.randint(1,7,5))
        else:
            assert len(vals) <= 5
            self.vals = np.sort(vals)
#    # trivial iteration for lst = list(game) functionality
#    # https://www.programiz.com/python-programming/iterator
#    def __iter__(self):
#        self.isNext = True
#        return self
#    def __next__(self):
#        if self.isNext:
#            self.isNext = False
#            return self
#        else:
#            raise StopIteration
    def set(self, mask, vals):
        """arr is a boolen array: True->reroll this dice"""
        assert len(mask) == 5, str(mask)
#        lp(self.vals)
#        lp(arr)

        newVals = np.where(mask, vals, self.vals)

        newVals = np.sort(newVals)
        return Dice(newVals)
    
    def roll(self, mask):
        """arr is a boolen array: True->reroll this dice"""
        assert len(mask) == 5, str(mask)
#        lp(self.vals)
#        lp(arr)
        randVals = np.random.randint(1, 7, 5)

        newVals = np.where(mask, randVals, self.vals)

        newVals = np.sort(newVals)
        return Dice(newVals)
#        lp(self.vals)
    
    def keep(self, reroll):
        """reroll: [bool]*5, True means reroll
        returns 0 to 5 dice as Dice object"""
        keepDice = self.vals[np.logical_not(reroll)]
        return Dice(keepDice)
    
    def __str__(self):
        return ', '.join(['{:.0f}'.format(val) for val in self.vals])
    
    def to_str(self, mask = [False]*5):
        """mask : bool array of len 5; marks if dice is intended fore reroll"""
        _str = ','.join(
                [str(v) + 'r' if b else str(v) for v, b in zip(self.vals, mask)])
        return '[{:}]'.format(_str)
#        _str = ''
#        for v, b in zip(self.vals, mask):
#            suf = 'r' if b else ''
#            _str += str(v) + suf
#        return _str
    
    def copy(self):
        return deepcopy(self)
    
    def compress(self):
        """Result of the 5 dice is encoded as a 5 digit integer"""
        comp = 0
        for ii in range(len(self.vals)):
            comp += 10**ii * self.vals[len(self.vals)-1-ii]
        return comp

    
    def read_compressed(self, intComp):
        """Writes compressed integer into self.vals"""
        pass
        
class ScoreBoard:
    cats = [
            'Aces','Twos','Threes','Fours','Fives','Sixes',
            'Four Of A Kind','Full House',
            'Small Straight','Large Straight','Yahtzee','Chance']
    def __init__(self):
        self.scores=np.ma.masked_array(np.empty(shape=MAX_TURNS),mask=True,dtype=int)
#    # trivial iteration for lst = list(game) functionality
#    # https://www.programiz.com/python-programming/iterator
#    def __iter__(self):
#        self.isNext = True
#        return self
#    def __next__(self):
#        if self.isNext:
#            self.isNext = False
#            return self
#        else:
#            raise StopIteration
    def copy(self):
        return deepcopy(self)
    
    @property
    def data(self):
        return self.scores.data
    @property
    def mask(self):
        return self.scores.mask
    
    @property
    def score(self):
        return self.getSum()
    
    
    @classmethod
    def get_cat_points(cls, dice, cat):
        if isinstance(dice, Dice):
            dice = dice.vals
        if len(dice)==0:
            return 0
        assert len(dice) <= 5
        assert isinstance(dice, np.ndarray), str(dice) + '; ' + str(type(dice))

        score = 0
        if cat>=0 and cat<=5:
            score=np.sum(dice[dice==(cat+1)])
        elif cat==6:
            if np.amax(np.bincount(dice))>=4:
                score=np.sum(dice)
        elif cat==7:
            sd=np.sort(np.bincount(dice))
            if (sd[-1]==3 and sd[-2]==2) or sd[-1]==5:
                score=25
        elif cat==8:
            sd=np.bincount(dice)
            lenStraight=0
            for ii in range(0,len(sd)):
                if sd[ii]>0:
                    lenStraight+=1
                else:
                    lenStraight=0
                if lenStraight>=4:
                    score=15
                    break
        elif cat==9:
            sd=np.where(np.bincount(dice)>0,1,0)
            if np.sum(sd)==5 and (not (dice==1).any() or not (dice==6).any()):
                score=30
        elif cat==10:
            if np.amax(np.bincount(dice))==5:
                score=50
        elif cat==11:
            score=np.sum(dice)
        else:
            assert False, 'invalid category position, cat='+str(cat)
        return score
        
    
    @classmethod
    def stat_cat_score(cls, dice):
        """Statistical category score
        
        forcasts the statistically EXACT score in each category.
        
        dice are the fixed dice only
        if len(dice.vals) == 5,
        this is equivalent to check_points for all cats simulatenously
        
        returns
        meanComb : np.ndarray shape=(MAX_TURNS,)
            mean score in each cat
        semComb : np.ndarray shape=(MAX_TURNS,)
            standard error of the mean
        """
        assert 0 <= len(dice.vals) <= 5, str(dice)
#        self.scores=np.ma.masked_array(np.empty(shape=MAX_TURNS),mask=True,dtype=int)
        
        nRr = 5-len(dice.vals)
        nCombs = 6**nRr

        yCombs = np.empty(shape=(nCombs, MAX_TURNS))
        for mm, comb in enumerate(product([1, 2, 3, 4, 5, 6], repeat=nRr)):
#            diceNew = np.copy(dice.vals)
            diceNew = Dice(list(comb) + list(dice.vals))
#            diceNew = Dice(diceNew)
#                lp(diceOld, deciRr, comb)
            for cc in range(MAX_TURNS):
#                lp(diceNew, cc)
                yCombs[mm, cc] = cls.get_cat_points(diceNew, cc)
        meanComb = np.mean(yCombs, axis=0)
        semComb = np.std(yCombs, axis=0) / nCombs**.5
        
        return meanComb, semComb
    
    
    def check_points(self, dice, cat):
        """Just check how many points one would get by assigning dice
        to category number cat
        dice:Dice
        cat: int.
        """
#        lp(dice, type(dice))
        score = ScoreBoard.get_cat_points(dice, cat)
        
        bonus = 0
        us = self.getUpperSum()
        if us < 63 and cat < 6 and us + score >= 63:
            bonus = 35
        return score, bonus
    
    @classmethod
    def check_points_max(self, cat):
        # cats = [
        #     0'Aces',
        #     1'Twos',
        #     2'Threes',
        #     3'Fours',
        #     4'Fives',
        #     5'Sixes',
        #     6'Four Of A Kind',
        #     7'Full House',
        #     8'Small Straight',
        #     9'Large Straight',
        #     10'Yahtzee',
        #     11'Chance']
        maxPnts = {0: 5,
                   1: 10,
                   2: 15,
                   3: 20,
                   4: 25,
                   5: 30,
                   6: 30,
                   7: 30,
                   8: 15,
                   9: 30,
                   10: 50,
                   11: 30}
        return maxPnts[cat]
    
    
    def add(self, dice, cat):
        """dice: Dice instance; posCat: int index for the category of the sb"""
        #dice: np.ndarray, posCat int
#        assert isinstance(posCat, int)
        assert np.issubdtype(type(cat), np.signedinteger)
        
        assert np.ma.getmask(self.scores)[cat]==1 \
        , str(self.scores)+ '\n'+str(self.open_cats_mask()) \
        +'\nposCat='+str(cat)+'\ntry to add to used category!'
#        dice = dice.vals
        self.scores[cat], bonus = self.check_points(dice, cat)
#        assert isinstance(diceInt,int)
        #convert dice to np.ndarray of len 5
#        dice=np.empty(shape=5,dtype=int)
#        for ii in range(0,5):
#            dice[-ii]=diceInt%10
#            diceInt=diceInt//10
        
            
#        if np.ma.getmask(self.scores)[cat]:
#            self.scores[cat]=0
            
    def open_cats_mask(self):
        return np.ma.getmask(self.scores).astype(int)
    
    def open_cats(self):
        inds = np.array(list(range(len(self.cats))))
#        lp(inds)
#        lp(self.scores.mask)
        return inds[self.scores.mask]
#        return np.maA.getmask(self.scores).astype(int)

    def getUpperSum(self):
        uSum=np.sum(self.scores[:6])
        if np.ma.is_masked(uSum):  # all entries masked
            uSum=0
        return uSum
        
    def getLowerSum(self):
        lSum=np.sum(self.scores[6:])
        if np.ma.is_masked(lSum):  # all entries masked
            lSum=0
        return lSum
    
    def getSum(self):
        uSum = self.getUpperSum()
        lSum = self.getLowerSum()
        bonus = 0
        if uSum >= 63:
            bonus += 35

        return uSum + bonus + lSum
    
#    def print(self):
#        print('='*10 + ' Score Board: ' + '='*10)
##        print()
#        for ii in range(MAX_TURNS):
#            if ii == 6:
#                print('-'*34)
#            score = '--' if self.scores.mask[ii] else str(self.scores[ii])
#            print('{:16}: {:2}'.format(ScoreBoard.cats[ii], score))
#        print('='*34)
#        print(' '*10 + 'Score: ' + str(self.getSum()) + ' '*10)
#        print('='*34)
    def print(self):
        print(self)
        
    def __str__(self):
        _str = ''
        _str += ('='*10 + ' Score Board: ' + '='*10 + '\n')
#        print()
        for ii in range(MAX_TURNS):
            if ii == 6:
                _str += ('-'*34 + '\n')
            score = '--' if self.scores.mask[ii] else str(self.scores[ii])
            _str += ('{:16}: {:2}'.format(ScoreBoard.cats[ii], score) + '\n')
        _str += ('='*34 + '\n')
        _str += (' '*10 + 'Score: ' + str(self.getSum()) + ' '*10 + '\n')
        _str += ('='*34 + '\n')
        return _str


class Game:
    """Stores a complete game"""
    
    def __init__(self, player, autoplay=True):
        """player: AbstracPlayer"""
        self.log = pd.DataFrame(
                columns=['scoreBoard',
                         'dice0', 'deci0', 'info0',
                         'dice1', 'deci1', 'info1',
                         'dice2', 'deci2', 'info2'],
                index=list(range(14)))
    
        if(autoplay):
            self.autoplay(player)

#     def step(self, player, turn=0, sb=None, dice=None):
#         if(sb==None):
#             sb = ScoreBoard()

#         prevSb = sb.copy()
        
#         dice0 = Dice()
#         if(dice is not None):
#             dice0 = dice

# #            lp(player.name)
#         deci0, info0 = player.choose_reroll(sb, dice0, 0)
# #            lp(deci0)
# #            lp(info0)
# #            lp(player.choose_reroll(sb, dice0, 0))
# #            lp(type(player.choose_reroll(sb, dice0, 0)))
#         dice1 = dice0.roll(deci0)
#         deci1, info1 = player.choose_reroll(sb, dice1, 1)
#         # choose cat
#         dice2 = dice1.roll(deci1)
#         deci2, info2 = player.choose_cat(sb, dice2)
#         sb.add(dice2, deci2)
# #            lp(self.log)
#         self.log.loc[turn, :] = [
#                 prevSb,
#                 dice0, deci0, info0, dice1, deci1, info1,
#                 dice2, deci2, info2]
        

    def autoplay(self, player):
        """Plays and stores a Yahtzee game.
        
        Returns
        -------
        None
            Description of return value
        
        See Also
        --------
        otherfunc : some related other function
        
        Examples
        --------
        These are written in doctest format, and should illustrate how to
        use the function.
        
        >>> a=[1,2,3]
        >>> [x + 3 for x in a]
        [4, 5, 6]
        """
        sb = ScoreBoard()
        for cc in range(0,MAX_TURNS):
            prevSb = sb.copy()
            
            dice0 = Dice()
#            lp(player.name)
            deci0, info0 = player.choose_reroll(sb, dice0, 0)
#            lp(deci0)
#            lp(info0)
#            lp(player.choose_reroll(sb, dice0, 0))
#            lp(type(player.choose_reroll(sb, dice0, 0)))
            dice1 = dice0.roll(deci0)
            deci1, info1 = player.choose_reroll(sb, dice1, 1)
            # choose cat
            dice2 = dice1.roll(deci1)
            deci2, info2 = player.choose_cat(sb, dice2)
            sb.add(dice2, deci2)
#            lp(self.log)
            self.log.loc[cc, :] = [
                    prevSb,
                    dice0, deci0, info0, dice1, deci1, info1,
                    dice2, deci2, info2]
        self.log.loc[MAX_TURNS, 'scoreBoard'] = sb

#        self.finalScoreBoard = sb
    

        
    def __str__(self, debugLevel=0):
        _str = ''
        # sort log by categories
#        lp(self.log)
#        dfLog = pd.DataFrame(
#                self.log, columns=['scores',
#                                   'dice0', 'deci0',
#                                   'dice1', 'deci1',
#                                   'dice2', 'deci2', 'info2'])
        dfLog = self.log
#        lp(dfLog)
        dfLog.index.name ='round'
        sb = self.log.loc[MAX_TURNS, 'scoreBoard']
        
        if debugLevel >= 1:
            for ii in range(MAX_TURNS):
                _str += 'ROUND ' + str(ii) +'\n'
                _str += 'Dice0: ' + str(dfLog.loc[ii,'dice0']) +'\n'
                _str += 'Deci0: ' + str(dfLog.loc[ii,'deci0']) +'\n'
                _str += 'Info0: ' + str(dfLog.loc[ii,'info0']) +'\n'
                _str += '--\n'
                _str += 'Dice1: ' + str(dfLog.loc[ii,'dice1']) +'\n'
                _str += 'Deci1: ' + str(dfLog.loc[ii,'deci1']) +'\n'
                _str += 'Info1: ' + str(dfLog.loc[ii,'info1']) +'\n'
                _str += '--\n'
                _str += 'Dice2: ' + str(dfLog.loc[ii,'dice2']) +'\n'
                _str += 'Deci2: ' + str(ScoreBoard.cats[dfLog.loc[ii,'deci2']]) +'\n'
                _str += 'Info2: ' + str(dfLog.loc[ii,'info2']) +'\n'
                _str += '===============\n'
#                for jj in range(3):
#                _str += (
#                        '--\n'
#                        + 'DICE: ' + str(dfLog.loc[ii,'dice2'])
#                        + ';\nEVAL: ' + str(dfLog.loc[ii,'info2'])
#                        + ';\nDECISION: ' + ScoreBoard.cats[dfLog.loc[ii,'deci2']]
#                        )
#                _str += '\n'
        
        dfLog = dfLog.reset_index()
        dfLog =dfLog.set_index('deci2')
        dfLog = dfLog.sort_index()

        n = 36
        _str += ('='*n + ' Score Board ' + '='*n +'\n')
        _str += ('{:16}: {:5} | round - dice (r = reroll)\n'
              .format('Category', 'Score'))
        _str += ('-'*(2*n + MAX_TURNS) +'\n')
        
        for ii in range(MAX_TURNS):
            
            if ii == 6:
                _str += ('-'*(2*n+MAX_TURNS) +'\n')
                upperSum = sb.getUpperSum()
                _str += '{:16}: {:5} |\n'.format(
                        'Upper Sum', str(upperSum))
                _str += '{:16}: {:5} |\n'.format(
                        'Bonus', '35' if upperSum >= 63 else '--')
                _str += ('-'*(2*n+MAX_TURNS) +'\n')
            score = str(sb.scores[ii])
            line = '{:16}: {:5} | {:>5} - '.format(ScoreBoard.cats[ii], score,
                    str(dfLog.loc[ii, 'round']))
            for rr in [0,1]:
                dice = dfLog.loc[ii, 'dice' + str(rr)]
                deci = dfLog.loc[ii, 'deci' + str(rr)]
                line += '{:} -> '.format(dice.to_str(deci))
            line += dfLog.loc[ii, 'dice2'].to_str()
            _str += (line +'\n')
        
        _str += ('='*(2*n+MAX_TURNS) +'\n')
        _str += (' '*(n+0) + ' Score: {:5d}\n'.format(sb.getSum()))
        _str += ('='*(2*n+MAX_TURNS) +'\n')
        return _str
    
    @property
    def score(self):
        return self.log.loc[MAX_TURNS, 'scoreBoard'].getSum()