import pandas as pd
import matplotlib.pyplot as plt
out_dir = "results/leak_layer_csvs/"
df = pd.read_csv(f"{out_dir}/qwq_leak_layer_ranking_thr_0_5.csv").sort_values("layer")
layers = df["layer"].to_numpy()
counts = df["flagged_count"].to_numpy()
plt.figure(figsize=(12, 4))
plt.bar(layers, counts, color="#4C78A8")  # change colors/styles here
plt.xlabel("Layer index")
plt.ylabel("Flagged neurons (|d| >= 0.5)")
plt.title("Flagged neurons by layer for qwq-32b")
plt.tight_layout()
plt.show()