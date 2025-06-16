import numpy as np
import random
from collections import defaultdict
from open_spiel.python import rl_agent
import pyspiel

class GTOKuhnPoker(rl_agent.AbstractAgent):
    def __init__(self, game: pyspiel.Game, player_id: int, name='rwywe_agent'):
        # Initialize using only player_id per PySpiel Bot API
        self.game = game
        self.player_id = player_id
    
    def step(self, state):
        iss = state.information_state_string(self.player_id)
        card = int(iss[0])
        r = np.random.random()
        turn = len(iss)
        # print(turn, iss, card)
        if self.player_id == 1:
            if card==2: return 1
            if card==1:
                if iss[1]=='p': return 0
                else: return int(r<1/3)
            if card==0:
                if iss[1]=='b': return 0
                else: return int(r<1/3)
        else:
            alpha = 0 # np.random.random() / 3
            r = np.random.random()
            if card==0:
                if turn==3: return 0
                else: return int(r<alpha)
            if card==1:
                if turn==1: return 0
                else: return int(r<alpha+1/3)
            if card==2:
                if turn==3: return 1
                else: return r<3*alpha
        