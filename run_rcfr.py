# run_rcfr.py
import pyspiel
import torch
import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python.pytorch import rcfr, rcfr_with_alpha
from open_spiel.python.algorithms import coxucb, coxucb2, coxucb3
from open_spiel.python.algorithms import exploitability

def train_fn(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch)           
        loss = torch.nn.functional.mse_loss(preds, y_batch) 
        loss.backward()
        optimizer.step()


def _new_model():
  return rcfr_with_alpha.DeepRcfrModel(
      pyspiel.load_game('kuhn_poker'),
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=2,
      use_skip_connections=True)

game = pyspiel.load_game("kuhn_poker")
res  = [[0], [0]]
solver = rcfr_with_alpha.RcfrSolver(game, [_new_model(), _new_model()])
transformer_player_id = 0
cox_ucb = coxucb2.CoxUCBBot(game, 1 - transformer_player_id)
alpha = 0.9
for i in range(10000):
    solver.evaluate_and_update_policy(train_fn)
    if i%5==4:
        game = pyspiel.load_game("kuhn_poker")
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                a, _ = state.chance_outcomes()[np.random.randint(len(state.chance_outcomes()))]
                state.apply_action(a)
            else:
                pid = state.current_player()
                if pid==transformer_player_id:
                    dummy = [None]*2
                    dummy[transformer_player_id] = solver._cumulative_seq_probs[transformer_player_id]
                    policy_fn = solver._root_wrapper.sequence_weights_to_policy_fn(dummy)                    
                    pf = policy_fn(state)
                    action = np.random.random()
                    state.apply_action(0 if pf[0] < action else 1)
                else:                   
                    # best_action = cox_ucb.step(state)
                    # state.apply_action(best_action)
                    legal_actions = state.legal_actions()
                    action = np.random.choice(legal_actions)
                    state.apply_action(action)
        if i<500:
            res[0].append(res[0][-1] + state.returns()[transformer_player_id])
            res[1].append(res[1][-1] + state.returns()[1-transformer_player_id])
    if i in [9,99,999,9999]:
        wrapper = solver._root_wrapper
        # pick player 0 or 1
        player_idx = 0

        # make a list that has your target weights in the right slot,
        # and dummy zeros for the other player
        num_players = solver._game.num_players()
        dummy = [None]*num_players
        dummy[player_idx] = solver._cumulative_seq_probs[player_idx]

        # build a policy‐function for only that player
        policy_fn = wrapper.sequence_weights_to_policy_fn(dummy)
    
        for state in rcfr_with_alpha.all_states(solver._game.new_initial_state(),
                                    depth_limit=-1,
                                    include_terminals=False,
                                    include_chance_states=False):
            if state.current_player() == player_idx:
                print(state.information_state_string(),
                    policy_fn(state))
        print() ; print() ; print()

        avg_policy = solver.average_policy()
        avg_policy = sorted(avg_policy.items(), key=lambda x : x[0])
        # nash_conv = exploitability.nash_conv(game, avg_policy)  # :contentReference[oaicite:0]{index=0}
        # print(f"Iter {i:4d} → NashConv (exploitability):")
        # for p in avg_policy: print(p)



    #     avg_policy = solver.average_policy()
    #     avg_policy = sorted(avg_policy.items(), key=lambda x : x[0])
    #     # nash_conv = exploitability.nash_conv(game, avg_policy)  # :contentReference[oaicite:0]{index=0}
    #     # print(f"Iter {i:4d} → NashConv (exploitability):")
    #     # for p in avg_policy: print(p)

# print("Transformer:", res[0]/100, "Random", res[1]/100)


x = list(range(len(res[0])))
print(res[0][-1] / len(res[0]))
plt.figure()
plt.plot(x, res[1], marker='o', label='COX-UCB with exponential decay')
plt.plot(x, res[0], marker='o', label='Transformer')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 