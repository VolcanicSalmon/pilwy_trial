"""
build_graphs.py

Combines:
  merge_reweight.py  — data loading, BSE, feature table, store
  aaencode.py        — AAEmbedding, NodeEncoder, resdf_to_aa_idx

Produces:
  feature_store.pkl  — unified per-sample feature store
  dataset.pkl        — list of PyG Data objects ready for GNN+SVM
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from argparse import ArgumentParser

# local imports
from merge_reweight import (
    load_usalign,
    load_gbsa,
    load_ring_nodes,
    load_ring_edges,
    build_residue_feature_table,
    build_feature_store,
    compute_bse,
)
from aaencode import (
    AAEmbedding,
    NodeEncoder,
    resdf_to_aa_idx,
)

INTERACT_ORD = {t: i/9 for i, t in enumerate(
    ['HBOND','IONIC','PICATION','PIPISTACK',
     'VDW','SSBOND','IAC','MC_MC','MC_SC','SC_SC'])}

DONOR_ORD = {t: i/3 for i, t in enumerate(
    ['MC','SC','LIG','UNK'])}


# ----------------------------------------------------------------
# args
# ----------------------------------------------------------------
def get_args():
    p = ArgumentParser()
    p.add_argument('--usa_pattern',
                   default='*/abonly/*summary.tsv',
                   help='glob for USalign summary TSVs')
    p.add_argument('--gbsa_pattern',
                   default='*/abonly/gbsa/gbsa.csv',
                   help='glob for GBSA CSVs')
    p.add_argument('--nodes_pattern',
                   default='*/ring/frame_*.pdb_ringNodes',
                   help='glob for RING node files')
    p.add_argument('--edges_pattern',
                   default='*/ring/frame_*.pdb_ringEdges',
                   help='glob for RING edge files')
    p.add_argument('--pdb_pattern',
                   default='*/abonly/frames/frame_0000.pdb',
                   help='glob for reference frame PDB per sample')
    p.add_argument('--dcd_pattern',
                   default='*/abonly/*.dcd',
                   help='glob for DCD trajectories')
    p.add_argument('--pathochain', default='A')
    p.add_argument('--plantchain', default='B')
    p.add_argument('--stride',     default=5,   type=int)
    p.add_argument('--feat_dim',   default=32,  type=int,
                   help='AAEmbedding + NodeEncoder output dim')
    p.add_argument('--store_out',  default='feature_store.pkl')
    p.add_argument('--dataset_out',default='dataset.pkl')
    return p.parse_args()


# ----------------------------------------------------------------
# BSE computation per sample
# ----------------------------------------------------------------
def compute_bse_store(pdb_pattern:  str,
                       dcd_pattern:  str,
                       pathochain:   str,
                       plantchain:   str,
                       stride:       int) -> dict:
    bse_store = {}
    pdbs = sorted(glob.glob(pdb_pattern))
    if not pdbs:
        print(f"[WARN] No PDBs found: {pdb_pattern}")
        return bse_store

    for pdb in pdbs:
        sample  = pdb.split(os.sep)[0]
        dcds    = glob.glob(os.path.join(sample, 'abonly', '*.dcd'))
        if not dcds:
            print(f"[SKIP BSE] no DCD for {sample}")
            bse_store[sample] = None
            continue
        print(f"[BSE] {sample}")
        try:
            bse_store[sample] = compute_bse(
                pdb, dcds[0], pathochain, plantchain, stride=stride)
        except Exception as ex:
            print(f"[ERROR BSE] {sample}: {ex}")
            bse_store[sample] = None

    return bse_store


# ----------------------------------------------------------------
# graph builder — uses NodeEncoder for node features
# ----------------------------------------------------------------
def store_to_graphs(store:       dict,
                    node_encoder: NodeEncoder) -> list:
    """
    One PyG Data per (sample, frame).
    Node features produced by NodeEncoder:
      aa_idx  → AAEmbedding → RBF → MLP   (feat_dim)
      extra   → extra_proj              (feat_dim)
      fused                             (feat_dim)

    data.x       : (N, feat_dim)   fused node embedding
    data.aa_idx  : (N,)            raw AA indices (kept for inspection)
    data.extra   : (N, 15)         raw structural/dynamic features
    data.edge_index: (2, E)
    data.edge_attr : (E, 4)
    data.graph_feat: (1, 6)        graph-level for SVM head
    data.y         : (1,)          ebind × tm1 target
    """
    EXTRA_FEAT_COLS = NodeEncoder.EXTRA_FEAT_COLS   # 15 cols

    dataset = []
    node_encoder.eval()

    for sample, feats in store.items():
        res_df_all = feats['res_feats']
        edges_all  = feats['edges']

        if res_df_all.empty:
            continue

        for frame in sorted(res_df_all['frame'].unique()):
            res_df   = res_df_all[res_df_all['frame'] == frame].copy()
            edges_df = edges_all[edges_all['frame']   == frame].copy()

            if res_df.empty:
                continue

            # --- normalise continuous cols ---
            max_pos = float(res_df['resid'].max()) or 1.
            res_df['pos_norm']     = res_df['resid']   / max_pos
            res_df['bfactor_norm'] = res_df['bfactor'] / 100.
            res_df['x_norm']       = res_df['x']       / 100.
            res_df['y_norm']       = res_df['y']       / 100.
            res_df['z_norm']       = res_df['z']       / 100.
            res_df['chain_flag']   = (res_df['chain'] != 'A').astype(float)
            res_df['ebind_norm']   = res_df['ebind']   / 100.

            # --- AA indices for embedding ---
            aa_idx = resdf_to_aa_idx(res_df)   # (N,) long

            # --- extra structural/dynamic features ---
            extra = torch.tensor(
                res_df[EXTRA_FEAT_COLS].fillna(0.).values.astype(np.float32),
                dtype=torch.float)             # (N, 15)

            # --- fused node features from NodeEncoder ---
            with torch.no_grad():
                x = node_encoder(aa_idx, extra)   # (N, feat_dim)

            # --- edge index + features ---
            node_ids = res_df['node_id'].tolist()
            node_idx = {nid: i for i, nid in enumerate(node_ids)}

            src_list, dst_list, edge_feat = [], [], []
            for _, row in edges_df.iterrows():
                src = row.get('NodeId1') or row.get('Residue1')
                dst = row.get('NodeId2') or row.get('Residue2')
                if src not in node_idx or dst not in node_idx:
                    continue
                src_list.append(node_idx[src])
                dst_list.append(node_idx[dst])
                edge_feat.append([
                    INTERACT_ORD.get(str(row.get('Interaction', '')), 0.),
                    DONOR_ORD.get(str(row.get('Donor', '')), 1.),
                    float(row.get('Distance', 0.)) / 10.,
                    float(row.get('Angle',    0.)) / 180.,
                ])

            if not src_list:
                print(f"[WARN] no edges for {sample} {frame}")
                continue

            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long)
            edge_attr  = torch.tensor(
                edge_feat, dtype=torch.float)

            # --- graph-level features (6 dims) for SVM head ---
            frame_tm    = float(res_df['tm1'].iloc[0])
            frame_ebind = float(res_df['ebind'].iloc[0])
            graph_feat  = torch.tensor([[
                frame_tm,
                frame_ebind,
                feats['graph_feat'][0],   # tm_mean
                feats['graph_feat'][3],   # ebind_mean
                feats['graph_feat'][4],   # ebind_std
                feats['graph_feat'][5],   # ebind_min
            ]], dtype=torch.float)

            target = frame_ebind * frame_tm

            data             = Data(
                x            = x,
                edge_index   = edge_index,
                edge_attr    = edge_attr,
                y            = torch.tensor([target], dtype=torch.float))
            data.graph_feat  = graph_feat
            data.aa_idx      = aa_idx     # keep for inspection / soft_forward
            data.extra       = extra      # keep for debugging
            data.sample      = sample
            data.frame       = frame
            data.tm1         = frame_tm

            dataset.append(data)

    dataset.sort(key=lambda d: d.tm1, reverse=True)
    print(f"\nDataset: {len(dataset)} graphs | "
          f"node_dim={dataset[0].x.shape[1] if dataset else '?'} | "
          f"edge_dim=4 | graph_dim=6")
    return dataset


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------
def main():
    args = get_args()

    print("Step 1: Loading data...")
    usa_df   = load_usalign(args.usa_pattern)
    print(f"  usa_df:   {len(usa_df)} rows")
    gbsa_df  = load_gbsa(args.gbsa_pattern)
    print(f"  gbsa_df:  {len(gbsa_df)} rows")
    nodes_df = load_ring_nodes(args.nodes_pattern)
    print(f"  nodes_df: {len(nodes_df)} rows")
    edges_df = load_ring_edges(args.edges_pattern)
    print(f"  edges_df: {len(edges_df)} rows")

    print("\nStep 2: Computing BSE...")
    bse_store = compute_bse_store(
        args.pdb_pattern, args.dcd_pattern,
        args.pathochain, args.plantchain, args.stride)
    print(f"  bse_store: {len(bse_store)} samples | "
          f"non-None: {sum(v is not None for v in bse_store.values())}")

    print("\nStep 3: Building feature store...")
    store = build_feature_store(
        usa_df, gbsa_df, nodes_df, edges_df,
        bse_store, outpkl=args.store_out)
    print(f"  store: {len(store)} samples")
    print(f"  pkl written: {os.path.exists(args.store_out)}")

    print("\nStep 4: Building graphs...")
    node_encoder = NodeEncoder(feat_dim=args.feat_dim)
    dataset      = store_to_graphs(store, node_encoder)
    print(f"  dataset: {len(dataset)} graphs")

    print("\nStep 5: Saving dataset...")
    with open(args.dataset_out, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"  dataset pkl written: {os.path.exists(args.dataset_out)}")

    if dataset:
        d = dataset[0]
        print(f"\nTop graph: {d.sample} {d.frame}")
        print(f"  x:          {d.x.shape}")
        print(f"  edge_index: {d.edge_index.shape}")
        print(f"  edge_attr:  {d.edge_attr.shape}")
        print(f"  graph_feat: {d.graph_feat.shape}")
        print(f"  y:          {d.y.item():.4f}")
        print(f"  tm1:        {d.tm1:.4f}")


if __name__ == '__main__':
    main()
