# check_surrogates.py
# Generates two plots per (category, direction):
#   1) True spline: Z vs Y for each fixed X
#   2) Surrogate:   model(x, y) vs Y for each fixed X
#
# Expects files named "{category}_{direction}_best.pt"
# e.g., "Regional Dealer_BUY_best.pt"

import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# ---------------------------
# 1) Define your spline_dict
# ---------------------------
# Put ALL your (category, direction) → (X,Y,Z) entries here.
# For brevity, I include the one you shared. Add the others similarly.
spline_dict = {

}

# ---------------------------
# 2) Model arch must match training
# ---------------------------
class SurrogateNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),                 # you trained with GELU(approximate='none')
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 3) True spline evaluator
# ---------------------------
def _hat(grid, q):
    j1 = np.searchsorted(grid, q, side="right")
    j0 = max(0, min(j1 - 1, len(grid) - 2))
    j1 = j0 + 1
    g0, g1 = grid[j0], grid[j1]
    t = 0.0 if g1 == g0 else (q - g0) / (g1 - g0)
    b = np.zeros(len(grid))
    b[j0], b[j1] = 1.0 - t, t
    return b

def true_spline_val(x, y, X, Y, Z):
    bx = _hat(X, x); by = _hat(Y, y)
    return float(bx @ Z @ by)

# ---------------------------
# 4) Plot helpers
# ---------------------------
def plot_true_surface(category, direction, X, Y, Z, outdir):
    plt.figure(figsize=(8,5))
    for i, xi in enumerate(X):
        plt.plot(Y, Z[i, :], label=f"X={xi:g}")
    plt.title(f"True surface: Z vs Y  —  {category} / {direction}")
    plt.xlabel("Y"); plt.ylabel("Z"); plt.legend(ncol=3, fontsize=8); plt.tight_layout()
    path = os.path.join(outdir, f"{category}_{direction}_true.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return path

def plot_surrogate_surface(category, direction, X, Y, predict_fn, outdir):
    plt.figure(figsize=(8,5))
    for xi in X:
        z_sur = np.array([predict_fn(xi, yj) for yj in Y], dtype=float)
        plt.plot(Y, z_sur, label=f"X={xi:g}")
    plt.title(f"Surrogate surface: Ẑ vs Y  —  {category} / {direction}")
    plt.xlabel("Y"); plt.ylabel("Ẑ"); plt.legend(ncol=3, fontsize=8); plt.tight_layout()
    path = os.path.join(outdir, f"{category}_{direction}_surrogate.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return path

# ---------------------------
# 5) Build predictor from .pt
# ---------------------------
def make_predictor_from_pt(pt_path, X, Y, hidden=128):
    """
    Returns a function predict(x,y) that:
      - clamps to domain
      - minmax-normalizes to [-1,1]
      - forwards through the trained torch model
    """
    device = torch.device("cpu")
    model = SurrogateNet(hidden=hidden).to(device)
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())

    @torch.no_grad()
    def predict(x, y):
        # clamp to match spline's outside behavior
        x = min(max(float(x), x_min), x_max)
        y = min(max(float(y), y_min), y_max)
        # normalize
        xs = (x - x_min) / (x_max - x_min) * 2.0 - 1.0
        ys = (y - y_min) / (y_max - y_min) * 2.0 - 1.0
        inp = torch.tensor([[xs, ys]], dtype=torch.float32, device=device)  # (1,2)
        out = model(inp).item()  # scalar
        return out

    return predict

# ---------------------------
# 6) Drive all combinations
# ---------------------------
def main():
    categories = ["Regional Dealer", "Originator", "Hedge Fund", "Real Money", "Money Manager"]
    directions = ["BUY", "SELL"]
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    for cat in categories:
        for d in directions:
            key = (cat, d)
            pt_name = f"{cat}_{d}_best.pt"
            if not os.path.exists(pt_name):
                print(f"[skip] {pt_name} not found.")
                continue

            if key not in spline_dict:
                print(f"[warn] spline_dict missing key {key}; add its (X,Y,Z) to plot.")
                continue

            X, Y, Z = spline_dict[key]
            print(f"[ok] {pt_name}: plotting…")

            # true surface
            p1 = plot_true_surface(cat, d, X, Y, Z, outdir)

            # surrogate surface
            predict_fn = make_predictor_from_pt(pt_name, X, Y, hidden=128)
            p2 = plot_surrogate_surface(cat, d, X, Y, predict_fn, outdir)

            print(f" saved: {p1}")
            print(f" saved: {p2}")

if __name__ == "__main__":
    main()
