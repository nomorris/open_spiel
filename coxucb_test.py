import numpy as np
import pyspiel
from open_spiel.python.algorithms import coxucb

# game = pyspiel.load_game("kuhn_poker")       # or "kuhn_poker", "go(9)", etc.

def random_eval(state, player):
    return 0.0

# values, best_action = maxn.maxn_search(
#     game,
#     state=None,              # None means “start from initial state”
#     value_function=random_eval,
#     depth_limit=3            # tune depth to your liking
# )

# print("Estimated returns per player:", values)
# print("Best root action for player 0:", best_action)

res = [0,0]
game = pyspiel.load_game("kuhn_poker")
cox_ucb = coxucb.CoxUCBBot(game, 0, -0.35,-0.20)
for k in range(1000):
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
            state.apply_action(a)
        else:
            pid = state.current_player()
            if pid==1:
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
            else:
                # iss = state.information_state_string(cox_ucb.player_id)
                # la = state.legal_actions()
                # pv = cox_ucb.compute_payoff_vector(state)
                # cr = cox_ucb.compute_confidence_region(iss, la)
                # ucs = cox_ucb.construct_utility_constrained_set(iss, la, pv)
                # strat = cox_ucb.select_strategy(iss, la, pv)
                best_action = cox_ucb.step(state=state)
                state.apply_action(best_action)
    res = [res[0] + state.returns()[0], res[1] + state.returns()[1]]
print("Agent:", res[0]/1000, "Random", res[1]/1000)
# print(cox_ucb.counts)
# print(cox_ucb.visits)
# print('confidence region: ', cr)
# print('payoff vector: ', pv)
# print('utility constrained set', ucs)
# print('strategy', strat)