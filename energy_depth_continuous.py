#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous (kernel-smoothed) Energy-Distance analysis vs normalized CMJ depth.

- Depth: per-Split normalized mean distance-to-medulla (outer→CMJ = 0→1).
- Shape: precomputed pairwise distances (distance_matrix.npy subset).
- Cell composition: Dirichlet-smoothed enrichment ratios vs Young baseline.
- For each depth t∈[0,1], compute kernel weights w_i(t) (Gaussian bandwidth h),
  group-normalize within Young/Old, and evaluate weighted energy distance.
  This yields smooth curves f_shape(t) and f_cell(t).

Outputs:
  - Plot of shape energy distance curve vs depth
  - Plot of composition energy distance curve vs depth
"""

import os, re, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from tqdm import tqdm

# -------------------- SETTINGS --------------------

PATCH_CSV          = "patch_df_cortex_cell.csv"
INDEX_MAP_PATH     = "patch_index_map.json"
DISTANCE_PATH      = "distance_matrix.npy"

MASK_DIR           = "cortex_mask"      # contains <Split>_mask.png
MEDULLA_RGB        = (125, 164, 152)    # medulla color in masks

SELECT_CELL_TYPES  = [
    "DN3", "ISP8",
    "Postsel DP (CD69+)", "Postsel DP (CD69-)", "Presel DP",
    "SP4 (CD69+)", "SP8 (CD69+)",
]

AGE_YOUNG_VALS     = {"Y", "Young", "young", "y"}
AGE_OLD_VALS       = {"O", "Old", "old", "o"}

RANDOM_SEED        = 0
OUT_DIR            = "depth_energy_outputs"

# Enrichment params
ALPHA_DIRICHLET    = 2.0

# Continuous depth params
H_BANDWIDTH        = 0.10       # Gaussian kernel bandwidth on [0,1]
GRID_POINTS        = 51         # number of t points in [0,1]
MIN_EFF_PER_AGE    = 25.0       # minimum effective sample size per age at t

# -------------------- UTILITIES --------------------

def infer_base_from_patchname(patch_name: str) -> str:
    m = re.search(r"(.*)_r\d+_c\d+", patch_name)
    return m.group(1) if m else os.path.splitext(patch_name)[0]

def medulla_distance_transform(mask_rgb_array: np.ndarray, medulla_rgb=(125, 164, 152)) -> np.ndarray:
    M = np.asarray(mask_rgb_array)
    med = (M[..., 0] == medulla_rgb[0]) & (M[..., 1] == medulla_rgb[1]) & (M[..., 2] == medulla_rgb[2])
    dt = ndi.distance_transform_edt(~med)  # distance *to* medulla
    return dt.astype(np.float32)

def mean_distance_in_rect(dt: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    H, W = dt.shape[:2]
    x0 = max(0, min(int(x0), W)); x1 = max(0, min(int(x1), W))
    y0 = max(0, min(int(y0), H)); y1 = max(0, min(int(y1), H))
    if x1 <= x0 or y1 <= y0: return np.nan
    block = dt[y0:y1, x0:x1]
    if block.size == 0: return np.nan
    return float(block.mean())

def gaussian_kernel(u):
    return np.exp(-0.5 * u * u, dtype=np.float64)

def effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2"""
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    q = (w * w).sum()
    if q <= 0: return 0.0
    return float((s * s) / q)

def energy_weighted(D: np.ndarray, idxY: np.ndarray, idxO: np.ndarray,
                    wY: np.ndarray, wO: np.ndarray) -> float:
    """
    Weighted energy distance using group-normalized weights (sum=1 within Y/O).
    ED = 2 μ_YO - μ_YY - μ_OO
    """
    if len(idxY) == 0 or len(idxO) == 0: return np.nan
    wY = np.asarray(wY, dtype=np.float64); wO = np.asarray(wO, dtype=np.float64)
    sY = wY.sum(); sO = wO.sum()
    if sY <= 0 or sO <= 0: return np.nan
    wY = wY / sY; wO = wO / sO

    Dab = D[np.ix_(idxY, idxO)]
    Da  = D[np.ix_(idxY, idxY)]
    Db  = D[np.ix_(idxO, idxO)]

    mu_ab = float(wY @ Dab @ wO)
    mu_aa = float(wY @ Da  @ wY)
    mu_bb = float(wO @ Db  @ wO)
    return 2.0 * mu_ab - mu_aa - mu_bb

def pairwise_euclid(A: np.ndarray) -> np.ndarray:
    G = A @ A.T
    nn = np.sum(A * A, axis=1, keepdims=True)
    sq = (nn + nn.T) - 2.0 * G
    np.maximum(sq, 0.0, out=sq)
    D = np.sqrt(sq, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    return D

def compute_enrichment(X_counts: np.ndarray, alpha: float,
                       df_rows: pd.DataFrame) -> np.ndarray:
    """
    Compute enrichment ratios vs Young baseline with Dirichlet smoothing.
    """
    X = np.asarray(X_counts, dtype=np.float64)
    N, J = X.shape
    row_sum = X.sum(axis=1, keepdims=True)
    R = (X + alpha) / np.maximum(row_sum + alpha * J, 1e-12)

    maskP = df_rows["age"].astype(str).isin(AGE_YOUNG_VALS).to_numpy()
    if not np.any(maskP):
        raise RuntimeError("No rows match Young age labels.")
    Xb = X[maskP]
    Y = (Xb.sum(axis=0, keepdims=True) + alpha) / np.maximum(Xb.sum() + alpha * J, 1e-12)
    E = R / np.maximum(Y, 1e-12)
    return E.astype(np.float32)

def compute_kernel_energy_curves(z, isY, isO, D_shape, D_comp, t_grid, h, min_eff):
    """
    Compute kernel-smoothed energy-distance curves for shape and composition.
    Returns (kept_mask, f_shape, f_cell) aligned to t_grid.
    """
    f_shape = np.full_like(t_grid, np.nan, dtype=np.float64)
    f_cell  = np.full_like(t_grid, np.nan, dtype=np.float64)
    kept    = np.zeros_like(t_grid, dtype=bool)

    IY_all = np.where(isY)[0]
    IO_all = np.where(isO)[0]

    for k, t in enumerate(t_grid):
        w = gaussian_kernel((z - t) / max(h, 1e-6))
        wY = w * isY
        wO = w * isO
        nY = effective_n(wY); nO = effective_n(wO)

        if nY >= min_eff and nO >= min_eff:
            eS = energy_weighted(D_shape, IY_all, IO_all, wY[IY_all], wO[IO_all])
            eC = energy_weighted(D_comp,  IY_all, IO_all, wY[IY_all], wO[IO_all])
            f_shape[k] = eS; f_cell[k] = eC; kept[k] = True

    return kept, f_shape, f_cell

# -------------------- MAIN --------------------

def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None

    # 1) Read CSV and harmonize age
    df = pd.read_csv(PATCH_CSV)
    needed = {"patch name", "age", "x_min", "x_max", "y_min", "y_max", "Split"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {sorted(missing)}")

    age_norm = []
    for a in df["age"].astype(str):
        if a in AGE_YOUNG_VALS: age_norm.append("Y")
        elif a in AGE_OLD_VALS: age_norm.append("O")
        else: age_norm.append(None)
    df["age_norm"] = age_norm
    df = df[df["age_norm"].isin(["Y", "O"])].reset_index(drop=True)

    if "base" not in df.columns:
        df["base"] = df["patch name"].astype(str).apply(infer_base_from_patchname)

    # 2) Load per-Split masks & compute distance transforms
    mask_dir = Path(MASK_DIR)
    if not mask_dir.exists():
        raise RuntimeError(f"Mask directory not found: {mask_dir}")

    dt_by_split = {}
    splits = sorted(df["Split"].astype(str).unique())
    for split in tqdm(splits, desc="Loading masks & computing distance transforms"):
        expected = mask_dir / f"{split}_mask.png"
        if expected.is_file():
            mask_path = expected
        else:
            exts = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
            cands = [p for p in mask_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts and p.stem.startswith(split)]
            if len(cands) == 1:
                mask_path = cands[0]
            elif len(cands) > 1:
                cands.sort(key=lambda p: len(p.stem))
                mask_path = cands[0]
            else:
                warnings.warn(f"[Split={split}] mask file not found in '{mask_dir}'. Distances will be NaN.")
                dt_by_split[split] = None
                continue
        im = Image.open(str(mask_path)).convert("RGB")
        dt_by_split[split] = medulla_distance_transform(np.array(im), medulla_rgb=MEDULLA_RGB)

    # 3) Mean distance-to-medulla per patch
    dist_means = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing mean CMJ distance per patch"):
        split = str(row["Split"])
        dt = dt_by_split.get(split, None)
        if dt is None:
            dist_means.append(np.nan); continue
        dmean = mean_distance_in_rect(dt, int(row["x_min"]), int(row["x_max"]),
                                      int(row["y_min"]), int(row["y_max"]))
        dist_means.append(dmean)

    df["cmj_dist_mean"] = dist_means
    df = df[~df["cmj_dist_mean"].isna()].reset_index(drop=True)

    # 4) Per-Split normalization: depth = 1 - mean / max(mean within Split)
    S_by_split = df.groupby("Split")["cmj_dist_mean"].transform("max").to_numpy()
    depth = 1.0 - np.clip(df["cmj_dist_mean"].to_numpy() / np.maximum(S_by_split, 1e-8), 0.0, 1.0)
    df["depth"] = depth

    # 5) Align to distance_matrix.npy (shape distances)
    with open(INDEX_MAP_PATH, "r") as f:
        mapping = json.load(f)
    name2i = {str(k): int(v) for k, v in mapping["name2i"].items()}
    present = df["patch name"].astype(str).isin(name2i)
    if not present.all():
        warnings.warn(f"{(~present).sum()} patch names missing in index_map; dropping them.")
        df = df.loc[present].reset_index(drop=True)
    idx_map = np.array([name2i[nm] for nm in df["patch name"].astype(str)], dtype=int)
    D_full = np.load(DISTANCE_PATH, mmap_mode="r")
    D_shape = np.asarray(D_full[np.ix_(idx_map, idx_map)], dtype=np.float32)

    # 6) Build cell-composition distance matrix (enrichment vs Young baseline)
    comp_cols = [c for c in SELECT_CELL_TYPES if c in df.columns]
    if not comp_cols:
        raise RuntimeError("No composition columns found among SELECT_CELL_TYPES.")
    X_counts = df[comp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    X_comp = compute_enrichment(X_counts, alpha=ALPHA_DIRICHLET, df_rows=df)
    D_comp = pairwise_euclid(X_comp)

    # 7) Compute kernel-smoothed energy distance curves
    t_grid = np.linspace(0.0, 1.0, GRID_POINTS)
    z   = df["depth"].to_numpy(np.float64)
    ag  = df["age_norm"].to_numpy(str)
    isY = (ag == "Y")
    isO = (ag == "O")

    kept, fS, fC = compute_kernel_energy_curves(
        z, isY, isO, D_shape, D_comp, t_grid, H_BANDWIDTH, MIN_EFF_PER_AGE
    )

    t_kept  = t_grid[kept]
    fS_kept = fS[kept]
    fC_kept = fC[kept]
    if t_kept.size < 3:
        raise RuntimeError("Too few depth points with adequate support. "
                           "Try increasing H_BANDWIDTH or decreasing MIN_EFF_PER_AGE.")

    # 8) Plot energy distance curves
    def plot_curve(xs, ys, title, ylabel, out):
        plt.figure(figsize=(6, 4), dpi=140)
        plt.plot(xs, ys, linewidth=2)
        plt.xlabel("Depth t (outer→CMJ)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3, linestyle=":")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()

    plot_curve(t_kept, fS_kept,
               title="Kernel energy (Shape): Y vs O over depth",
               ylabel="Energy distance (shape)",
               out=os.path.join(OUT_DIR, "curve_shape_energy_continuous.png"))

    plot_curve(t_kept, fC_kept,
               title="Kernel energy (Composition, enrich): Y vs O over depth",
               ylabel="Energy distance (composition)",
               out=os.path.join(OUT_DIR, "curve_cell_energy_continuous.png"))

    print(f"\nDone. Plots saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
