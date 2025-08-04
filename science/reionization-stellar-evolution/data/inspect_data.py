import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

data = pd.read_csv("10k_curves_physical.csv")
# Subsample data to 50k
data = data.sample(n=50000, random_state=42).reset_index(drop=True)

shorter_names = {
    "z"            : "z",
    "F_STAR10"     : "f_*",
    "F_ESC10"      : "f_esc",
    "ALPHA_STAR"   : "α_*",
    "ALPHA_ESC"    : "α_esc",
    "M_TURN"       : "M_t",
    "L_X"          : "L_X",
    "t_STAR"       : "τ_*",
    "xHI"          : "x_HI",
    "Ts"           : "T_s",
    "Tb"           : "T_b"
}

var_names = ["z", "F_STAR10", "F_ESC10", "ALPHA_STAR", "ALPHA_ESC", "M_TURN", "L_X", "t_STAR"]

fig, ax = plt.subplots(1,1, figsize=(10, 8), sharex=True)
ax.plot(data["z"], data["xHI"], 'b.', alpha=0.5, markersize=1)
ax.set_ylabel("Neutral Hydrogen Fraction (xHI)")
ax.set_xlabel("Redshift (z)")
ax.grid(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(10, 8), sharex=True)
ax.plot(data["z"], data["Ts"],  'r.', alpha=0.5, markersize=1)
ax.set_ylabel("Spin Temperature (Ts)")
ax.set_xlabel("Redshift (z)")
ax.grid(True)
plt.tight_layout()
plt.show()

print(data)