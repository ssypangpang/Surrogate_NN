# train_spline_surrogate.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import random
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt


def build_dataset(X, Y, Z, n_random=2000, knot_dup=50):
    # knots: every (xi,yj) from Z
    knots = []
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            knots.append((xi, yj, Z[i,j]))
    # Duplicate knots to overweight them
    knot_list = knots * knot_dup

    # Random interior points: sample within bounding box (uniform)
    x_min, x_max = X[0], X[-1]
    y_min, y_max = Y[0], Y[-1]
    rng = np.random.default_rng(12345)
    random_pts = []
    for _ in range(n_random):
        xr = float(rng.uniform(x_min, x_max))
        yr = float(rng.uniform(y_min, y_max))
        # compute true bilinear via hat weights (cheap)
        # find nearest cell
        def hat(grid, q):
            j1 = np.searchsorted(grid, q, side='right')
            j0 = max(0, min(j1 - 1, len(grid) - 2))
            j1 = j0 + 1
            g0, g1 = grid[j0], grid[j1]
            t = 0.0 if g1 == g0 else (q - g0) / (g1 - g0)
            b = np.zeros(len(grid))
            b[j0], b[j1] = 1.0 - t, t
            return b
        bx = hat(X, xr); by = hat(Y, yr)
        val = float(bx @ Z @ by)
        random_pts.append((xr, yr, val))

    all_pts = knot_list + random_pts
    random.shuffle(all_pts)
    xs = np.array([[pt[0], pt[1]] for pt in all_pts], dtype=float)
    ys = np.array([pt[2] for pt in all_pts], dtype=float)
    return xs, ys

def normalize_np(xy):
    # map to roughly [-1,1]
    xs = (xy[:,0] - x_min) / (x_max - x_min) * 2 - 1
    ys = (xy[:,1] - y_min) / (y_max - y_min) * 2 - 1
    return np.stack([xs, ys], axis=1)

class SurrogateNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)
    
# function: true spline (bilinear)
def true_spline_val(xq, yq):
    def hat(grid, q):
        j1 = np.searchsorted(grid, q, side='right')
        j0 = max(0, min(j1 - 1, len(grid) - 2))
        j1 = j0 + 1
        g0, g1 = grid[j0], grid[j1]
        t = 0.0 if g1 == g0 else (q - g0) / (g1 - g0)
        b = np.zeros(len(grid))
        b[j0], b[j1] = 1.0 - t, t
        return b
    bx = hat(X, xq); by = hat(Y, yq)
    return float(bx @ Z @ by)




# ------------------------------
# 1) Your grid (from screenshot)
# ------------------------------

spline_dict = {
    ('Regional Dealer', 'BUY'): [
np.array([], dtype=float),
np.array([], dtype=float),

np.array([

], dtype=float)
    ],


}   



for k in spline_dict.keys():
    category, direction = k
    print("================== {} {} ==================".format(category, direction))
    X, Y, Z = spline_dict[k]
    # Build dataset
    X_data, y_data = build_dataset(X, Y, Z, n_random=3000, knot_dup=80)
    print("Dataset size:", X_data.shape, y_data.shape)
    device = torch.device("cpu")
    # normalization
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())

    # Torch dataset
    X_norm = normalize_np(X_data)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y_data.reshape(-1,1), dtype=torch.float32)

    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    # Small MLP

    model = SurrogateNet(hidden=128).to(device)
    # ------------------------------
    # 4) Train
    # ------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.5)
    loss_fn = nn.MSELoss()

    # training loop
    n_epochs = 1200
    best_loss = 1e9
    for ep in range(1, n_epochs+1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss) * xb.shape[0]
        sched.step()
        epoch_loss = running / len(ds)
        if ep % 100 == 0 or ep == 1:
            print(f"Epoch {ep:4d} loss {epoch_loss:.6e}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # save best weights
            torch.save(model.state_dict(), "{}_{}_best.pt".format(category, direction))

    print("Best train loss:", best_loss)    