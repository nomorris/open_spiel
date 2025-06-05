
import numpy as np
import random
from collections import defaultdict
from open_spiel.python import rl_agent
import pyspiel

class RWYWEAgent(rl_agent.AbstractAgent):
    def __init__(self, game: pyspiel.Game, player_id: int, name='rwywe_agent'):
        # Initialize using only player_id per PySpiel Bot API
        self.game = game
        self.player_id = player_id
        self.k = 0.018
        self.opponent_model = defaultdict(lambda: defaultdict(int))
        self.opponent_belief = defaultdict()
        self.bet = 0
        self.gains = 0
        self.card = ''
        self.hist = []
        self.num_rounds = 1

    def restart(self): pass
    
    def step(self, state):
        """Choose an action that maximizes expected value while staying within safety bound k"""        
        # Calculate expected value for each action
        self.card = state.history()[self.player_id]
        action_values = {}
        options = ['fold', 'call'] if len(state.history())<4 else ['check', 'bet']
        for action in options:
            win = self.opponent_model.get((tuple(self.hist+[action]), self.card, 2))
            los = self.opponent_model.get((tuple(self.hist+[action]), self.card, 1))
            win = 0 if not win else win
            los = 0 if not los else los
            action_values[action] = 0.5 if win+los==0 else win / (win + los)
        
    
        # Find the best action within safety bounds
        best_action = None
        best_value = -np.inf
        
        for action in options:
            if action_values[action] >= self.k and action_values[action] > best_value:
                best_value = action_values[action]
                best_action = action
        
        # If no safe actions, fall back to GTO strategy
        if best_action is None:
            best_action = gto_p1(state.legal_actions()[0], state.legal_actions()[1], self.card) if self.player_id==1 else gto_p2(options[0], options[1], self.card)
        self.hist += [best_action]
        return 1 if best_action in ('fold', 'check') else 0

    def update_opponent_model(self, move):
        self.opponent_model[move] = 1 if  move not in self.opponent_model[move] else self.opponent_model[move]+1


    def decision(self, opt1, opt2, card):
        """Make a decision according to RWYWE algorithm"""
        options = [opt1, opt2]
        action = self.safe_best_response(options, self.hist, card, is_p1=self.player_id==1)
        return action
    
    def update_k(self, utility):
        """Update the safety buffer k based on observed utility"""
        self.k = max(0, self.k + utility - self.num_rounds*0.018)

def gto_p2(opt1, opt2, card):
    if card==2: return opt2 
    elif card==1:
        if opt1=='check': return 'check'
        else: return ['fold', 'call', 'fold'][np.random.randint(3)]
    else:
        if opt1=='fold': return 'fold'
        else: return ['check', 'bet', 'check'][np.random.randint(3)]
        

def gto_p1(opt1, opt2, card):
    alpha = 0.2
    r = np.random.random()
    r = 1 if r<alpha else 0
    if card==2: return opt2 
    elif card==1:
        if opt1=='check': return 'check'
        else: return ['fold', 'call'][r]
    else:
        if opt1=='fold': return 'fold'
        else: return ['check', 'bet'][r]