#!/usr/bin/env python3
import os, re, math, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyclustering.cluster.kmedoids import kmedoids

# -------------------- USER SETTINGS --------------------
N_CLUSTERS        = 3
random_seed       = 0
np.random.seed(random_seed)

# Permutation to align composition cluster labels to shape cluster labels.
# COMP_PERMUTATION[old_label] = new_label.  Identity = no relabelling.
COMP_PERMUTATION  = np.array([0, 1, 2], dtype=int)

# IO: patch-level (quadrant) CSV and shape distance
patch_csv          = "patch_df_cortex_cell.csv"   # <- input CSV (READ-ONLY)
distance_path      = "distance_matrix.npy"
index_map_path     = "patch_index_map.json"

# Cell-level point data + phenotypes (READ-ONLY)
cell_csv           = "Combmatrices_after_joining_TEC_Thy.csv"
phenos_csv         = "Combphenos_after_joining_TEC_Thy.csv"

# Big masks (used for overlays only)
input_dir          = "Segmentation_Masks_Ben/Old_vs_Young"

# Root output folder
output_root        = "shape_vs_comp_clustering_results_TEC_1hop_all_cell_types_ordered"

# Overlays sub-settings
filter_name_hint   = "K8"                          # set to "" to include all .tif/.tiff

# Tiling / mask (must match how tiles were generated)
threshold          = 0
margin             = 0
patch_w, patch_h   = 200, 200

# Enrichment pipeline
ALPHA              = 0.5

# 1-hop neighbourhood settings
# resolution 0.5 µm/px, thymocyte diameter ~6 µm (radius ~3 µm).
# Using 10 px (~5 µm) as default max centre-to-centre distance to nearest TEC cell.
HOP_RADIUS_PX      = 10.0

# -------- Cell type sets --------
# Composition is defined using ONLY these thymocyte types
SELECTED_CELL_TYPES = [
    "DN3",
    "Presel DP",
    "Postsel DP (CD69-)",
    "Postsel DP (CD69+)",
]

# TEC scaffold cell types
TEC_CELL_TYPES = [
    "TEC (PDPN+)",
    "cTEC hi (Ly51+)",
    "cTEC hi (Ly51-)",
    "cTEC lo (Ly51+)",
    "cTEC lo (Ly51-)",
]

# Map numeric Split -> label Split, to match patch_df
split_map = {
    1: "Y1",
    2: "O1",
    3: "O2",
    4: "O3",
    5: "Y2",
    6: "O4",
    7: "O5",
}

# Colors: avoid jet extremes by sampling inside this range
COLOR_RANGE = (0.15, 0.85)

# -------------------- HELPERS --------------------
def run_kmedoids_from_distance(D, n_clusters, seed):
    n = D.shape[0]
    init = list(np.random.choice(n, n_clusters, replace=False))
    km = kmedoids(D, init, data_type="distance_matrix", random_state=seed)
    km.process()
    labels = np.empty(n, dtype=int)
    for cid, pts in enumerate(km.get_clusters()):
        labels[pts] = cid
    return labels

def compute_enrichment_matrix(df, alpha, phenotype_cols):
    """
    Compute the enrichment matrix E (obs_prop / Young-baseline_prop)
    over phenotype_cols.

    Returns
    -------
    E : (N, P) float32
        Enrichment ratios per patch and phenotype.
    ph_cols : list of str
        The phenotype columns actually used.
    """
    ph_cols = [c for c in phenotype_cols if c in df.columns]
    if not ph_cols:
        raise RuntimeError("No phenotype columns available for enrichment.")

    counts = df[ph_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    N, P = counts.shape

    # per-row smoothed proportions
    denom    = counts.sum(axis=1, keepdims=True, dtype=np.float32) + np.float32(alpha * P)
    obs_prop = (counts + np.float32(alpha)) / denom

    # Young-group baseline proportions
    mask = (df["age"].astype(object) == "Y")
    if not np.any(mask):
        raise RuntimeError("No rows match age == 'Y'")
    g_tot  = counts[mask].sum(axis=0, keepdims=True)
    g_prop = (g_tot + np.float32(alpha)) / (g_tot.sum() + np.float32(alpha * P))
    E      = (obs_prop / g_prop).astype(np.float32)

    return E, ph_cols

def build_enrichment_distance(df, alpha, phenotype_cols):
    """
    Build distances on L2-normalized enrichment vectors using ONLY phenotype_cols.
    """
    E, ph_cols = compute_enrichment_matrix(df, alpha, phenotype_cols)

    # L2-normalize rows; cosine distance -> Euclidean on the unit sphere
    norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12).astype(np.float32)
    X     = E / norms
    G     = X @ X.T
    D2    = np.clip(2.0 - 2.0 * G, 0.0, None, dtype=np.float32)
    np.sqrt(D2, out=D2)
    return D2.astype(np.float32)

def compute_onehop_counts_per_patch(
    patch_df,
    cell_df,
    tec_types,
    tcell_types,
    hop_radius_px,
):
    """
    For each patch (quadrant), recompute counts of selected thymocyte
    cell types (tcell_types) restricted to those whose centroids lie
    within hop_radius_px (in pixels) of at least one TEC cell centroid
    in the SAME patch.

    Uses:
      - patch_df columns: 'Split', 'x_min', 'x_max', 'y_min', 'y_max'
      - cell_df columns: 'Split', 'XPos', 'YPos', 'CellType'

    Returns
    -------
    patch_df : the same DataFrame, with tcell_types columns updated
               to contain 1-hop counts.
    """
    patch_df = patch_df.copy()
    cell_df  = cell_df.copy()

    # use a common string key for Split (e.g. "Y1", "O1", ...)
    patch_df["Split_key"] = patch_df["Split"].astype(str)
    cell_df["Split_key"]  = cell_df["Split"].astype(str)

    # Initialise / reset counts for the selected T cell types
    for ct in tcell_types:
        patch_df[ct] = 0

    # Group cells by split key to reduce search space
    cells_by_split = {
        s: sub.reset_index(drop=True)
        for s, sub in cell_df.groupby("Split_key")
    }

    for i, row in tqdm(patch_df.iterrows(),
                       total=len(patch_df),
                       desc="1-hop counts per patch"):
        split_key = row["Split_key"]

        if split_key not in cells_by_split:
            continue

        cells_split = cells_by_split[split_key]

        x_min, x_max = row.get("x_min", np.nan), row.get("x_max", np.nan)
        y_min, y_max = row.get("y_min", np.nan), row.get("y_max", np.nan)

        if np.isnan([x_min, x_max, y_min, y_max]).any():
            continue

        # Restrict to cells whose centroid lies inside this patch
        in_patch = cells_split[
            (cells_split["XPos"] >= x_min) & (cells_split["XPos"] < x_max) &
            (cells_split["YPos"] >= y_min) & (cells_split["YPos"] < y_max)
        ]
        if in_patch.empty:
            continue

        tec_cells = in_patch[in_patch["CellType"].isin(tec_types)]
        if tec_cells.empty:
            continue

        t_cells = in_patch[in_patch["CellType"].isin(tcell_types)]
        if t_cells.empty:
            continue

        tec_xy = tec_cells[["XPos", "YPos"]].to_numpy(np.float32)
        t_xy   = t_cells[["XPos", "YPos"]].to_numpy(np.float32)

        # For each T cell, distance to nearest TEC cell
        diff   = t_xy[:, None, :] - tec_xy[None, :, :]
        dist2  = np.sum(diff * diff, axis=2)
        min_d  = np.sqrt(dist2.min(axis=1))

        onehop_mask = (min_d <= hop_radius_px)
        if not np.any(onehop_mask):
            continue

        t_onehop = t_cells.iloc[onehop_mask]
        counts = t_onehop["CellType"].value_counts()

        for ct in tcell_types:
            patch_df.at[i, ct] = int(counts.get(ct, 0))

    return patch_df

def make_overlays(cluster_map, out_dir, *, n_clusters, color_positions):
    """
    Create overlays, coloring each quadrant by its cluster, and add
    a scale bar showing ONE QUADRANT length in X and Y directions.
    """
    os.makedirs(out_dir, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None

    # jet colors but avoid extremes by sampling inside COLOR_RANGE
    cmap_cont      = plt.get_cmap("jet")
    cluster_colors = [cmap_cont(pos)[:3] for pos in color_positions]

    bases = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff"))
        and (filter_name_hint in f if filter_name_hint else True)
    ]

    for base_fn in tqdm(bases, desc=f"Overlays -> {os.path.basename(out_dir)}"):
        base, _ = os.path.splitext(base_fn)

        # load & square-crop mask
        img  = Image.open(os.path.join(input_dir, base_fn))
        arr  = np.array(img)
        mask = np.any(arr[..., :3] > threshold, axis=2) if arr.ndim == 3 else arr > threshold
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        y0b, y1b = ys.min(), ys.max()
        x0b, x1b = xs.min(), xs.max()
        y0 = max(y0b - margin, 0); x0 = max(x0b - margin, 0)
        y1 = min(y1b + margin, arr.shape[0] - 1); x1 = min(x1b + margin, arr.shape[1] - 1)
        h, w = y1 - y0 + 1, x1 - x0 + 1
        side = max(h, w)
        cy, cx = y0 + h // 2, x0 + w // 2
        y0s = max(cy - side // 2, 0); x0s = max(cx - side // 2, 0)
        sq  = arr[y0s:y0s + side, x0s:x0s + side]
        seg = np.array(Image.fromarray(sq).convert("L"))

        # pad to patch grid
        H, W   = seg.shape
        pad_h  = (math.ceil(H / patch_h) * patch_h) - H
        pad_w  = (math.ceil(W / patch_w) * patch_w) - W
        seg_pad   = np.pad(seg, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        mask_pad  = seg_pad > threshold
        H2, W2    = seg_pad.shape
        nrows, ncols = H2 // patch_h, W2 // patch_w

        # overlay RGB
        rgb_overlay = np.zeros((H2, W2, 3), dtype=float)

        # paint every patch that belongs to this base
        for patch_name, cl in cluster_map.items():
            if not str(patch_name).startswith(base + "_"):
                continue
            m = re.search(r"_r(\d+)_c(\d+)", str(patch_name))
            if not m:
                continue
            r, cidx = map(int, m.groups())
            y0p = (nrows - 1 - r) * patch_h  # reverse row index
            x0p = cidx * patch_w
            pm  = mask_pad[y0p:y0p + patch_h, x0p:x0p + patch_w]
            ci  = int(cl)
            if not (0 <= ci < n_clusters):
                continue
            color = cluster_colors[ci]
            rgb_overlay[y0p:y0p + patch_h, x0p:x0p + patch_w][pm] = color

        # plot & save
        fig, ax = plt.subplots(figsize=(W2 / 100, H2 / 100), dpi=100)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax.imshow(seg_pad, cmap="gray")
        ax.imshow(rgb_overlay, alpha=0.6)
        ax.axis("off")

        # SCALE BAR: ONE QUADRANT (patch_w x patch_h)
        margin_frac = 0.05
        x0_bar = W2 * margin_frac
        y0_bar = H2 * (1.0 - margin_frac)

        # horizontal bar (X direction)
        ax.plot(
            [x0_bar, x0_bar + patch_w],
            [y0_bar, y0_bar],
            linewidth=3,
            color='white'
        )

        # vertical bar (Y direction)
        ax.plot(
            [x0_bar, x0_bar],
            [y0_bar, y0_bar - patch_h],
            linewidth=3,
            color='white'
        )

        fig.savefig(
            os.path.join(out_dir, f"{base}_clusters_jet.png"),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.5,
        )
        plt.close(fig)

# -------------------- MAIN --------------------
def main():
    os.makedirs(output_root, exist_ok=True)

    # 1) Read filtered patch CSV (READ-ONLY)
    df = pd.read_csv(patch_csv)
    df["patch name"] = df["patch name"].astype(str)
    df["Split"] = df["Split"].astype(str)

    # 1b) Load cell-level data + map Phenos -> CellType
    cells = pd.read_csv(cell_csv)
    names_df = pd.read_csv(phenos_csv, header=None)
    cluster_names = names_df.iloc[0].tolist()
    phenos_mapping = {i + 1: name for i, name in enumerate(cluster_names)}
    cells["CellType"] = cells["Phenos"].map(phenos_mapping)

    # Map numeric Split -> 'Y1', 'O1', ... to match patch_df
    cells["Split"] = cells["Split"].map(split_map)
    cells["Split"] = cells["Split"].astype(str)

    # 1c) Recompute counts for SELECTED_CELL_TYPES using 1-hop around TEC cells
    df = compute_onehop_counts_per_patch(
        df,
        cells,
        tec_types=TEC_CELL_TYPES,
        tcell_types=SELECTED_CELL_TYPES,
        hop_radius_px=HOP_RADIUS_PX,
    )

    # 2) Load name<->index map, filter df to names present in the distance matrix
    with open(index_map_path, "r") as f:
        mapping = json.load(f)
    name2i = {str(k): int(v) for k, v in mapping["name2i"].items()}

    present_mask = df["patch name"].isin(name2i)
    if not present_mask.all():
        missing = (~present_mask).sum()
        print(f"[warn] {missing} rows have patch names not found in index map; dropping them.")
        df = df.loc[present_mask].reset_index(drop=True)

    idx    = [name2i[nm] for nm in df["patch name"]]
    D_full = np.load(distance_path, mmap_mode="r")
    D_sub  = np.asarray(D_full[np.ix_(idx, idx)], dtype=np.float32)
    print(f"Distance submatrix shape: {D_sub.shape}")

    # 3) Enrichment-based distances on the SAME rows, using Y-group baseline
    D_enrich = build_enrichment_distance(df, alpha=ALPHA, phenotype_cols=SELECTED_CELL_TYPES)

    # 4) K-medoids clustering
    print(f"[info] Running k-medoids with k={N_CLUSTERS} for shape and composition...")
    labels_dist   = run_kmedoids_from_distance(D_sub, N_CLUSTERS, random_seed)
    labels_enrich = run_kmedoids_from_distance(D_enrich, N_CLUSTERS, random_seed)

    # Apply permutation to align composition labels to shape labels
    labels_enrich_aligned = COMP_PERMUTATION[labels_enrich]

    # Build cluster maps for overlays
    cluster_map_dist = {name: int(lbl) for name, lbl in zip(df["patch name"], labels_dist)}
    cluster_map_enrich_aligned = {
        name: int(lbl) for name, lbl in zip(df["patch name"], labels_enrich_aligned)
    }

    # 5) Colors + overlays
    c_lo, c_hi = COLOR_RANGE
    color_positions = np.linspace(c_lo, c_hi, N_CLUSTERS)

    # Shape clustering overlays
    make_overlays(
        cluster_map_dist,
        os.path.join(output_root, f"distance_kmedoids_{N_CLUSTERS}"),
        n_clusters=N_CLUSTERS,
        color_positions=color_positions,
    )
    # Composition clustering overlays
    make_overlays(
        cluster_map_enrich_aligned,
        os.path.join(output_root, f"enrichment_aligned_{N_CLUSTERS}"),
        n_clusters=N_CLUSTERS,
        color_positions=color_positions,
    )

if __name__ == "__main__":
    main()
