import numpy as np
import pyspiel
from open_spiel.python.examples import rwywe
from open_spiel.python.algorithms import random_agent, maxn


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
for k in range(1000):
    game = pyspiel.load_game("kuhn_poker")
    bot1 = rwywe.RWYWEAgent(game, 1-k%2)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
            state.apply_action(a)
        else:
            # legal_actions = state.legal_actions()
            # action = np.random.choice(legal_actions)
            # state.apply_action(action)
            pid = state.current_player()
            if pid==k%2:
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
                state.apply_action(action)
            else:
                a = bot1.step(state)
                state.apply_action(a)
    result = 1 if state.rewards()[1-k%2] > 0 else -1
    bot1.update_k(result)
    bot1.update_opponent_model(tuple(state.history() + [result]))
    bot1.num_rounds += 1
    res = [res[0] + state.returns()[k%2], res[1] + state.returns()[1-k%2]]
print("Agent:", res[1]/1000, "Random", res[0]/1000)
