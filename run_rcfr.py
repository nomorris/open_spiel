# run_rcfr.py
import pyspiel
from open_spiel.python.pytorch import rcfr
from open_spiel.python.algorithms import exploitability

game = pyspiel.load_game("kuhn_poker")
solver = rcfr.RcfrSolver(game, [[0,0,0],[0,0,0]])

for i in range(100):
    solver.evaluate_and_update_policy(lambda x : 0)
    if i % 100 == 0:
        avg_policy = solver.average_policy()
        nash_conv = exploitability.nash_conv(game, avg_policy)  # :contentReference[oaicite:0]{index=0}
        print(f"Iter {i:4d} → NashConv (exploitability): {nash_conv:.6f}")

# finally, you can also save or inspect solver.average_policy() as you like