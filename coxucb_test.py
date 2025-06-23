import numpy as np
import pyspiel
from open_spiel.python.algorithms import coxucb

res1 = [0,0]
res2 = [0,0]
game = pyspiel.load_game("kuhn_poker")
cum_regrets = []
iterations = 1000
coxucb1 = coxucb.CoxUCBBot(game, 0)
coxucb2 = coxucb.CoxUCBBot(game, 1)
for k in range(iterations):
    game1 = pyspiel.load_game("kuhn_poker")
    game2 = pyspiel.load_game("kuhn_poker")
    state1 = game1.new_initial_state()
    state2 = game2.new_initial_state()
    while not state1.is_terminal():
        if state1.is_chance_node():
            a, _ = state1.chance_outcomes()[np.random.randint(len(state1.chance_outcomes()))]
            state1.apply_action(a)
        else:
            pid = state1.current_player()
            if pid==1:
                legal_actions = state1.legal_actions()
                action = np.random.choice(legal_actions)
                state1.apply_action(action)
            else:
                best_action = coxucb1.step(state1)
                state1.apply_action(best_action)
    while not state2.is_terminal():
        if state2.is_chance_node():
            a, _ = state2.chance_outcomes()[np.random.randint(len(state2.chance_outcomes()))]
            state2.apply_action(a)
        else:
            pid = state2.current_player()
            if pid==0:
                legal_actions = state2.legal_actions()
                action = np.random.choice(legal_actions)
                state2.apply_action(action)
            else:
                best_action = coxucb2.step(state2)
                state2.apply_action(best_action)

    res1 = [res1[0] + state1.returns()[0], res1[1] + state1.returns()[1]]
    res2 = [res2[0] + state2.returns()[0], res2[1] + state2.returns()[1]]
print("AgentP1:", res1[0]/iterations, "RandomP2:", res1[1]/iterations)
print("AgentP2:", res2[1]/iterations, "RandomP1:", res2[0]/iterations)


# import numpy as np
# import pyspiel
# from open_spiel.python.algorithms import coxucb
# from open_spiel.python.examples import gto_kuhn_poker

# # game = pyspiel.load_game("kuhn_poker")     

# def random_eval(state, player):
#     return 0.0

# # values, best_action = maxn.maxn_search(
# #     game,
# #     state=None,              # None means “start from initial state”
# #     value_function=random_eval,
# #     depth_limit=3            # tune depth to your liking
# # )

# # print("Estimated returns per player:", values)
# # print("Best root action for player 0:", best_action)

# res = [0,0]
# game = pyspiel.load_game("kuhn_poker")
# agent_player_id = 0
# cum_regrets = []
# cox_ucb = coxucb.CoxUCBBot(game, agent_player_id)
# for k in range(10000):
#     game = pyspiel.load_game("kuhn_poker")
#     state = game.new_initial_state()
#     while not state.is_terminal():
#         if state.is_chance_node():
#             a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
#             state.apply_action(a)
#         else:
#             pid = state.current_player()
#             if pid==0:
#                 legal_actions = state.legal_actions()
#                 action = np.random.choice(legal_actions)
#                 state.apply_action(action)
#             else:
#                 # iss = state.information_state_string(cox_ucb.player_id)
#                 # print(iss)
#                 # la = state.legal_actions()
#                 # pv = cox_ucb.compute_payoff_vector(state)
#                 # cr = cox_ucb.compute_confidence_region(iss, la)
#                 # ucs = cox_ucb.construct_utility_constrained_set(iss, la, pv)
#                 # strat = cox_ucb.select_strategy(iss, la, pv)
#                 best_action = cox_ucb.step(state=state)
#                 state.apply_action(best_action)
#                 # print(state.legal_actions(), state.information_state_string(cox_ucb.player_id))
#     iss_agent = state.information_state_string(agent_player_id)
#     iss_opp = state.information_state_string(1 - agent_player_id)
#     best_value = 0 if iss_opp[0] > iss_agent[1] else 2
#     regret = best_value

#     res = [res[0] + state.returns()[agent_player_id], res[1] + state.returns()[1-agent_player_id]]
# print("Agent:", res[0]/10000, "Random", res[1]/10000)
# # print(cox_ucb.counts)
# # print(cox_ucb.visits)
# # print('confidence region: ', cr)
# # print('payoff vector: ', pv)
# # print('utility constrained set', ucs)
# # print('strategy', strat)