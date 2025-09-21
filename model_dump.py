# ---- build_nn_models.py (paste into a cell or .py file) ----
import os
import numpy as np
import torch

# 1) Bring in your NNModel implementation
# If NNModel is in the same file, remove this import.
# Otherwise change the import path below to where NNModel lives.
from your_module import NNModel   # <-- EDIT if needed

# 2) Your category/direction -> integer mapping
CATEGORY_TO_INT = {
    "Originator":       1,
    "Regional Dealer":  2,
    "Money Manager":    3,
    "Hedge Fund":       4,
    "Real Money":       5,
}
DIRECTION_TO_INT = {"BUY": 1, "SELL": 2}

# 3) List of names we’ll iterate over
CATEGORIES = ["Regional Dealer", "Originator", "Hedge Fund", "Real Money", "Money Manager"]
DIRECTIONS = ["BUY", "SELL"]

# 4) Helper: robustly extract linear weights/biases from a state_dict
def extract_wb_from_state(state_dict):
    """
    Returns W0, b0, W1, b1, W2, b2 as numpy arrays (float64),
    assuming a 3-linear-layer MLP:
        Linear(2->H) -> GELU -> Linear(H->H) -> GELU -> Linear(H->1)
    Works with keys like:
       net.0.weight/bias, net.2.weight/bias, net.4.weight/bias
    and also tolerates prefixes like 'module.' if present.
    """
    # unwrap common wrappers (e.g., {'state_dict': ...})
    if isinstance(state_dict, dict) and "state_dict" in state_dict and all(
        k.startswith("module.") or k.startswith("net.") or ".weight" in k or ".bias" in k
        for k in state_dict["state_dict"].keys()
    ):
        sd = state_dict["state_dict"]
    else:
        sd = state_dict

    # If everything has a "module." prefix, strip it on the fly via lookups
    def get(k):
        if k in sd:
            return sd[k]
        mk = f"module.{k}"
        if mk in sd:
            return sd[mk]
        # as fallback, find the last key that endswith k (handles slight name diffs)
        matches = [kk for kk in sd.keys() if kk.endswith(k)]
        if not matches:
            raise KeyError(f"Key '{k}' not found in state_dict keys: {list(sd.keys())[:10]} ...")
        return sd[matches[-1]]

    # Sorted lists should give ['net.0.weight','net.2.weight','net.4.weight'] etc.
    w_keys = sorted([k for k in sd.keys() if k.endswith(".weight")])
    b_keys = sorted([k for k in sd.keys() if k.endswith(".bias")])
    if len(w_keys) < 3 or len(b_keys) < 3:
        raise ValueError(f"Expected 3 linear layers; got weights={w_keys}, biases={b_keys}")

    W0 = get(w_keys[0]).cpu().numpy().astype(np.float64)
    W1 = get(w_keys[1]).cpu().numpy().astype(np.float64)
    W2 = get(w_keys[2]).cpu().numpy().astype(np.float64)

    b0 = get(b_keys[0]).cpu().numpy().astype(np.float64).ravel()
    b1 = get(b_keys[1]).cpu().numpy().astype(np.float64).ravel()
    b2 = get(b_keys[2]).cpu().numpy().astype(np.float64).ravel()

    # Normalize shapes a bit:
    # W2 may be (1,H) or (H,); NNModel __init__ you showed already handles both.
    return W0, b0, W1, b1, W2, b2

# 5) Build all NNModel instances from *.pt, using X/Y ranges from your spline_dict
def build_models_from_checkpoints(spline_dict, hidden=128, folder="."):
    """
    Parameters
    ----------
    spline_dict : dict
        { (Category, Direction) : (X, Y, Z) }  used to derive x_min..y_max
    hidden : int (unused here, only for doc)
        Must match the hidden size used in training (e.g., 128).
    folder : str
        Directory where '{Category}_{Direction}_best.pt' files live.

    Returns
    -------
    models : dict
        { (category_int, direction_int) : NNModel(...) }
    """
    models = {}
    for cat in CATEGORIES:
        for d in DIRECTIONS:
            key = (cat, d)
            pt_path = os.path.join(folder, f"{cat}_{d}_best.pt")
            if not os.path.exists(pt_path):
                print(f"[skip] missing checkpoint: {pt_path}")
                continue
            if key not in spline_dict:
                print(f"[skip] no (X,Y,Z) in spline_dict for {key}")
                continue

            X, Y, Z = spline_dict[key]
            x_min, x_max = float(np.min(X)), float(np.max(X))
            y_min, y_max = float(np.min(Y)), float(np.max(Y))

            # load torch state and extract W/b
            state = torch.load(pt_path, map_location="cpu")
            W0, b0, W1, b1, W2, b2 = extract_wb_from_state(state)

            cat_int = CATEGORY_TO_INT[cat]
            dir_int = DIRECTION_TO_INT[d]

            # instantiate your NNModel
            model = NNModel(
                x_min, x_max, y_min, y_max,
                W0, b0, W1, b1, W2, b2
            )
            models[(cat_int, dir_int)] = model
            print(f"[ok] built model for ({cat_int}, {dir_int}) from '{pt_path}'")

    return models

# 6) EXAMPLE USAGE (uncomment when running)
# models = build_models_from_checkpoints(spline_dict, hidden=128, folder=".")
# # Example prediction with Regional Dealer (2), BUY (1)
# z = models[(2, 1)].predict(0.12, 1.00)
# print("pred:", z)



import numpy as np

def format_array(arr, max_per_line=32, indent="       "):
    """Format a numpy array with line wrapping."""
    flat = arr.flatten()
    parts = []
    for i in range(0, len(flat), max_per_line):
        chunk = flat[i:i+max_per_line]
        parts.append(indent + " " + np.array2string(
            chunk,
            separator=", ",
            max_line_width=10**6  # don't wrap automatically
        ))
    return "[\n" + ",\n".join(parts) + "\n" + indent + "]"

def dump_models_dict(models, max_per_line=32):
    """
    Pretty-print the models dictionary as Python code text,
    wrapping long arrays for readability.
    """
    print("{")
    for (cat_int, dir_int), m in models.items():
        print(f"  ({cat_int}, {dir_int}): NNModel(")
        print(f"       x_min={m.x_min},")
        print(f"       x_max={m.x_max},")
        print(f"       y_min={m.y_min},")
        print(f"       y_max={m.y_max},")
        for name in ["W0","b0","W1","b1","W2","b2"]:
            arr = getattr(m, name)
            if arr.ndim == 1 or arr.ndim == 2:
                # pretty format for 1D or 2D weights
                if arr.ndim == 1:
                    print(f"       {name}=np.array({format_array(arr, max_per_line)}, dtype=float),")
                else:
                    # for 2D arrays, format each row separately
                    rows = []
                    for row in arr:
                        rows.append(format_array(row, max_per_line, indent="           "))
                    print(f"       {name}=np.array([\n" + ",\n".join(rows) + "\n       ], dtype=float),")
            else:
                print(f"       {name}=np.array({arr.tolist()}, dtype=float),")
        print("  ),")
    print("}")
