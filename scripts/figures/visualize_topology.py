"""
scripts/visualize_topology.py
==============================
Visualisasi struktur skeleton topology BlockGCN sebelum dan sesudah training.

Yang divisualisasi:
  1. Static adjacency matrix A (graph COCO 17-joint)
  2. Learned fc1 weights per GCN block (init = identity → berubah saat training)
  3. Effective topology per blok (sum of 3 partitions, averaged over 8 heads)
  4. Perbedaan before vs after training

Usage:
  python scripts/visualize_topology.py
  python scripts/visualize_topology.py --weights weights/40_(17bal)/runs-113-10057.pt
  python scripts/visualize_topology.py --weights weights/40_(17bal)/runs-113-10057.pt --out topology_viz.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import networkx as nx

WEIGHT_PATH_DEFAULT = 'weights_se/03_17bal/runs-43-6751.pt'

# ── COCO 17-joint metadata ──────────────────────────────────────────────────
JOINT_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_sho', 'R_sho', 'L_elb', 'R_elb', 'L_wri', 'R_wri',
    'L_hip', 'R_hip', 'L_kne', 'R_kne', 'L_ank', 'R_ank',
]

COCO_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Posisi canonical skeleton (x, y) untuk visualisasi graph 2D
JOINT_POS = {
    0:  (0.5, 0.95),   # nose
    1:  (0.42, 0.90),  # L_eye
    2:  (0.58, 0.90),  # R_eye
    3:  (0.35, 0.87),  # L_ear
    4:  (0.65, 0.87),  # R_ear
    5:  (0.30, 0.75),  # L_sho
    6:  (0.70, 0.75),  # R_sho
    7:  (0.20, 0.60),  # L_elb
    8:  (0.80, 0.60),  # R_elb
    9:  (0.12, 0.45),  # L_wri
    10: (0.88, 0.45),  # R_wri
    11: (0.38, 0.52),  # L_hip
    12: (0.62, 0.52),  # R_hip
    13: (0.35, 0.35),  # L_kne
    14: (0.65, 0.35),  # R_kne
    15: (0.33, 0.15),  # L_ank
    16: (0.67, 0.15),  # R_ank
}


def build_model_untrained(graph_cls='graph.yolo.Graph'):
    """Model dengan random init (sebelum training) — fc1 = identity matrices."""
    from model.BlockGCN_SE import Model
    model = Model(
        num_class=2, num_point=17, num_person=1,
        graph=graph_cls,
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3, drop_out=0.0, adaptive=True, alpha=False,
    )
    model.eval()
    return model


def load_model_trained(weights_path, graph_cls='graph.yolo.Graph'):
    """Model dari checkpoint (sesudah training)."""
    from model.BlockGCN_SE import Model
    model = Model(
        num_class=2, num_point=17, num_person=1,
        graph=graph_cls,
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3, drop_out=0.0, adaptive=True, alpha=False,
    )
    state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_fc1_matrices(model):
    """
    Ekstrak fc1 dari setiap GCN block.
    fc1 shape: (3, num_heads, V, V)
    Return: list of 10 ndarray, masing-masing (3, 8, 17, 17)
    """
    blocks = [model.l1, model.l2, model.l3, model.l4, model.l5,
              model.l6, model.l7, model.l8, model.l9, model.l10]
    result = []
    for blk in blocks:
        fc1 = blk.gcn1.fc1.detach().cpu().numpy()  # (3, H, V, V)
        result.append(fc1)
    return result


def effective_adj(fc1_arr):
    """
    fc1_arr: (3, H, V, V)
    Effective adjacency = sum partitions, mean heads → (V, V)
    """
    return fc1_arr.sum(axis=0).mean(axis=0)  # (V, V)


def off_diagonal(mat):
    """Mask diagonal ke 0, return only off-diagonal connections."""
    m = mat.copy()
    np.fill_diagonal(m, 0)
    return m


# ── Plot helpers ────────────────────────────────────────────────────────────

def plot_adjacency_heatmap(ax, mat, title, vmin=None, vmax=None, cmap='RdBu_r', center=None):
    if center is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=8)
    ax.set_xticks(range(0, 17, 4))
    ax.set_xticklabels(range(0, 17, 4), fontsize=6)
    ax.set_yticks(range(0, 17, 4))
    ax.set_yticklabels(range(0, 17, 4), fontsize=6)
    return im


def plot_skeleton_graph(ax, edge_weights=None, title='Skeleton Graph'):
    """
    Visualisasi skeleton sebagai graph 2D.
    edge_weights: dict {(i,j): weight} atau None (uniform)
    """
    G = nx.Graph()
    G.add_nodes_from(range(17))
    G.add_edges_from(COCO_PAIRS)

    pos = {k: (v[0], v[1]) for k, v in JOINT_POS.items()}

    if edge_weights is None:
        widths  = [2.0] * len(COCO_PAIRS)
        ecolors = ['#333333'] * len(COCO_PAIRS)
    else:
        max_w = max(abs(w) for w in edge_weights.values()) + 1e-8
        widths  = [max(0.5, 4.0 * abs(edge_weights.get((i,j), 0)) / max_w)
                   for i, j in COCO_PAIRS]
        ecolors = ['red' if edge_weights.get((i,j), 0) > 0 else 'blue'
                   for i, j in COCO_PAIRS]

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=ecolors, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80,
                           node_color='#FFA500', alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: JOINT_NAMES[i] for i in range(17)},
                            font_size=4.5)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


# ── Main visualization ───────────────────────────────────────────────────────

def visualize(weights_path=None, out_path='topology_visualization.png'):
    """
    4-row figure:
      Row 0: Static COCO graph A + skeleton diagram + explanation
      Row 1: OFF-DIAGONAL fc1 sebelum training (all zeros — identity init)
      Row 2: OFF-DIAGONAL fc1 sesudah training (learned non-zero connections)
      Row 3: Skeleton graphs with learned edge weights per block (top 5 blocks)
    """
    print("Loading models...")
    model_before = build_model_untrained()

    if weights_path:
        model_after = load_model_trained(weights_path)
        print(f"Loaded trained weights: {weights_path}")
    else:
        model_after = None
        print("No weights provided — only show before-training topology")

    fc1_before = get_fc1_matrices(model_before)
    fc1_after  = get_fc1_matrices(model_after) if model_after else None
    has_after  = model_after is not None

    from graph.yolo import Graph
    g = Graph(labeling_mode='spatial')
    A = g.A  # (3, 17, 17)

    n_rows = 3 + (1 if has_after else 0)
    fig = plt.figure(figsize=(22, 5 * n_rows))
    fig.suptitle('BlockGCN — Skeleton Topology: Before vs After Training\n'
                 '(fc1 adaptive graph weights, off-diagonal = learned joint connections)',
                 fontsize=13, fontweight='bold', y=0.995)

    NCOLS = 12  # subplot columns per row

    # ── Row 0: Static adjacency + skeleton ────────────────────────────────
    ax_a0 = fig.add_subplot(n_rows, NCOLS, 1)
    ax_a1 = fig.add_subplot(n_rows, NCOLS, 2)
    ax_a2 = fig.add_subplot(n_rows, NCOLS, 3)
    ax_sk = fig.add_subplot(n_rows, NCOLS, (4, 6))
    ax_exp = fig.add_subplot(n_rows, NCOLS, (7, NCOLS))

    plot_adjacency_heatmap(ax_a0, A[0], 'A[0]\nself-link', cmap='Blues')
    plot_adjacency_heatmap(ax_a1, A[1], 'A[1]\ncentripetal', cmap='Greens')
    plot_adjacency_heatmap(ax_a2, A[2], 'A[2]\ncentrifugal', cmap='Reds')
    plot_skeleton_graph(ax_sk, title='Static COCO Graph\n(predefined, tidak berubah)')

    ax_exp.axis('off')
    txt = (
        "CARA BACA:\n"
        "  Graph A = static skeleton (predefined COCO_PAIRS, tidak berubah saat training)\n"
        "  fc1 = learnable parameter di setiap unit_gcn, shape (3 partisi, 8 heads, 17, 17)\n"
        "  Init: fc1 = identity matrix per head  ->  hanya self-connection (diagonal)\n"
        "  Sesudah training: off-diagonal menjadi non-zero\n"
        "    = model PELAJARI hubungan antar joint di luar graph asli\n\n"
        "  Baris 1 (BEFORE): off-diagonal = 0 (belum ada hubungan dipelajari)\n"
        "  Baris 2 (AFTER):  off-diagonal != 0 (koneksi baru dipelajari)\n"
        "  Baris 3: skeleton graph dengan tebal tepi = kekuatan koneksi dipelajari"
    )
    ax_exp.text(0.02, 0.5, txt, transform=ax_exp.transAxes,
                fontsize=8.5, va='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Row 1: OFF-DIAGONAL fc1 sebelum training ──────────────────────────
    # Sebelum training: off-diagonal = 0 (identity init), semua plot hitam/putih rata
    for blk_idx in range(10):
        ax = fig.add_subplot(n_rows, NCOLS, NCOLS + blk_idx + 1)
        mat = off_diagonal(effective_adj(fc1_before[blk_idx]))
        # Gunakan skala dari fc1_after agar perbandingan fair
        plot_adjacency_heatmap(ax, mat, f'B{blk_idx+1}\nBEFORE\n(off-diag)',
                               cmap='RdBu_r', vmin=-0.15, vmax=0.15, center=0)
    ax_lbl1 = fig.add_subplot(n_rows, NCOLS, NCOLS + 11)
    ax_lbl1.axis('off')
    ax_lbl1.text(0.05, 0.5,
                 "BEFORE training\nOff-diagonal = 0\n(identity init)\nTidak ada koneksi\nyang dipelajari",
                 transform=ax_lbl1.transAxes, fontsize=8, va='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    if has_after:
        # ── Row 2: OFF-DIAGONAL fc1 sesudah training ──────────────────────
        off_vals = [off_diagonal(effective_adj(fc1_after[i])) for i in range(10)]
        omax = max(np.abs(v).max() for v in off_vals) + 1e-8
        for blk_idx in range(10):
            ax = fig.add_subplot(n_rows, NCOLS, 2 * NCOLS + blk_idx + 1)
            plot_adjacency_heatmap(ax, off_vals[blk_idx],
                                   f'B{blk_idx+1}\nAFTER\n(off-diag)',
                                   cmap='RdBu_r', vmin=-omax, vmax=omax, center=0)
        ax_lbl2 = fig.add_subplot(n_rows, NCOLS, 2 * NCOLS + 11)
        ax_lbl2.axis('off')
        ax_lbl2.text(0.05, 0.5,
                     "AFTER training\nOff-diagonal != 0\nMerah = koneksi\nkuat positif\nBiru = negatif",
                     transform=ax_lbl2.transAxes, fontsize=8, va='center',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # ── Row 3: Skeleton graphs dengan learned edge weights ─────────────
        # Tampilkan 5 block (L1, L3, L5, L8, L10) sebagai skeleton graph
        show_blocks = [0, 2, 4, 7, 9]
        for idx, blk_idx in enumerate(show_blocks):
            ax = fig.add_subplot(n_rows, NCOLS, 3 * NCOLS + idx + 1)
            off = off_vals[blk_idx]
            # edge weights: rata-rata |w_ij| + |w_ji| untuk tiap pair
            ew = {}
            for i, j in COCO_PAIRS:
                ew[(i, j)] = float((off[i, j] + off[j, i]) / 2)
            plot_skeleton_graph(ax, edge_weights=ew,
                                title=f'Block {blk_idx+1}\nLearned edges')
        for idx in range(len(show_blocks), NCOLS - 2):
            ax = fig.add_subplot(n_rows, NCOLS, 3 * NCOLS + idx + 1)
            ax.axis('off')
        ax_lbl3 = fig.add_subplot(n_rows, NCOLS, 3 * NCOLS + 11)
        ax_lbl3.axis('off')
        ax_lbl3.text(0.05, 0.5,
                     "Skeleton graph:\nTebal = strength\nkoneksi dipelajari\nMerah = positif\nBiru = negatif\n"
                     "(L1,L3,L5,L8,L10)",
                     transform=ax_lbl3.transAxes, fontsize=8, va='center',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    return fig


def visualize_per_block_detail(weights_path, block_idx=0, out_path='topology_block_detail.png'):
    """
    Detail satu GCN block: tampilkan 8 heads × 3 partisi sebelum dan sesudah.
    block_idx: 0-9
    """
    model_before = build_model_untrained()
    model_after  = load_model_trained(weights_path)

    fc1_b = get_fc1_matrices(model_before)[block_idx]  # (3, 8, 17, 17)
    fc1_a = get_fc1_matrices(model_after)[block_idx]

    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    fig.suptitle(f'GCN Block {block_idx+1} — fc1 per partition per head\n'
                 f'(Atas=before training, Bawah=after training)',
                 fontsize=11, fontweight='bold')

    part_names = ['Self-link', 'Centripetal', 'Centrifugal']
    for p in range(3):
        for h in range(8):
            mat_b = fc1_b[p, h]  # (17, 17)
            mat_a = fc1_a[p, h]

            ax_b = axes[p][h]
            vmax = max(abs(mat_b).max(), abs(mat_a).max()) + 1e-8
            im = ax_b.imshow(mat_b, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
            ax_b.set_title(f'{part_names[p]}\nH{h+1}', fontsize=6)
            ax_b.set_xticks([]); ax_b.set_yticks([])

            # Overlay batas delta sebagai alpha layer
            delta = mat_a - mat_b
            ax_b.contour(delta, levels=5, colors='yellow', linewidths=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {out_path}")


def print_topology_stats(weights_path):
    """Print statistik perubahan topology tiap block."""
    model_before = build_model_untrained()
    model_after  = load_model_trained(weights_path)

    fc1_before = get_fc1_matrices(model_before)
    fc1_after  = get_fc1_matrices(model_after)

    print("\n=== Topology Change Statistics (per GCN block) ===")
    print(f"{'Block':<6} {'Max|D|':<10} {'Mean|D|':<10} {'Off-diag change %':<20}")
    print("-" * 50)
    for i in range(10):
        eff_b = effective_adj(fc1_before[i])
        eff_a = effective_adj(fc1_after[i])
        delta = eff_a - eff_b
        off_mask = ~np.eye(17, dtype=bool)
        off_diag_change = np.abs(delta[off_mask]).mean() / (np.abs(eff_b[off_mask]).mean() + 1e-8) * 100
        print(f"L{i+1:<5} {np.abs(delta).max():<10.4f} {np.abs(delta).mean():<10.4f} {off_diag_change:<20.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=WEIGHT_PATH_DEFAULT,
                        help='Path ke trained weights (.pt)')
    parser.add_argument('--out', default='topology_visualization.png',
                        help='Output image path')
    parser.add_argument('--block-detail', type=int, default=None,
                        help='Jika diset, tampilkan detail per head untuk block ini (0-9)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Hanya print statistik, tidak render gambar')
    args = parser.parse_args()

    weights = args.weights if os.path.exists(args.weights) else None
    if weights is None and args.weights:
        print(f"Warning: weights not found at {args.weights}, running without trained model")

    if args.stats_only and weights:
        print_topology_stats(weights)
    else:
        visualize(weights_path=weights, out_path=args.out)
        if args.block_detail is not None and weights:
            detail_out = args.out.replace('.png', f'_block{args.block_detail+1}_detail.png')
            visualize_per_block_detail(weights, block_idx=args.block_detail, out_path=detail_out)
        if weights:
            print_topology_stats(weights)
