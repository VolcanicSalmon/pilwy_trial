# gnn_sage.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from argparse import ArgumentParser
from aaencode import AAEmbedding, NodeEncoder, resdf_to_aa_idx


# ----------------------------------------------------------------
# GraphSAGE layer — pure PyTorch, no PyG
# ----------------------------------------------------------------
class SAGEConv(nn.Module):
    """
    GraphSAGE: h_v = MLP(concat(h_v, mean(h_neighbours)))
    No MessagePassing, no PyG — just sparse matmul + MLP.
    """
    def __init__(self, in_dim, out_dim,
                 edge_dim=4, dropout=0.15):
        super().__init__()
        # aggregate neighbour features
        self.agg_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
        )
        # edge feature MLP → scalar attention weight
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid(),
        )
        # update: concat self + aggregated neighbour
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm     = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """
        x          : (N, in_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_dim)
        """
        N   = x.shape[0]
        src = edge_index[0]   # source nodes
        dst = edge_index[1]   # destination nodes

        # edge attention weights
        e_w = self.edge_mlp(edge_attr)   # (E, 1)

        # gather source features
        x_src = x[src]                   # (E, in_dim)

        # weighted neighbour features
        x_agg = self.agg_mlp(x_src) * e_w   # (E, out_dim)

        # mean aggregation using scatter
        # manual scatter_mean without PyG
        out = torch.zeros(N, x_agg.shape[-1], device=x.device)
        cnt = torch.zeros(N, 1, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(x_agg), x_agg)
        cnt.scatter_add_(0, dst.unsqueeze(-1),
                          torch.ones(len(dst), 1, device=x.device))
        cnt  = cnt.clamp(min=1)
        out  = out / cnt                     # (N, out_dim)

        # update: concat self + aggregated
        h   = self.update_mlp(
            torch.cat([x, out], dim=-1))    # (N, out_dim)

        # residual + norm
        h   = self.norm(h + self.residual(x))
        return self.dropout(h)


# ----------------------------------------------------------------
# full model
# ----------------------------------------------------------------
class BindingSAGE(nn.Module):
    """
    GraphSAGE-based PPI interface predictor.
    
    Node features: AAEmbedding (RBF) + structural/dynamic feats
    Edge features: RING interaction type, donor, distance, angle
    Graph features: TM-score, ebind_mean etc → broadcast to nodes

    Outputs:
      node_logits : (N,) is_interface probability
      graph_pred  : (B,) binding affinity
    """
    def __init__(self,
                 aa_feat_dim:    int   = 32,
                 edge_dim:       int   = 4,
                 hidden:         int   = 64,
                 n_layers:       int   = 3,
                 dropout:        float = 0.15,
                 graph_feat_dim: int   = 6):
        super().__init__()

        self.node_encoder    = NodeEncoder(feat_dim=aa_feat_dim)
        self.graph_feat_proj = nn.Linear(graph_feat_dim, hidden)

        # SAGE layers
        self.convs = nn.ModuleList()
        in_dim = aa_feat_dim
        for _ in range(n_layers):
            self.convs.append(
                SAGEConv(in_dim, hidden, edge_dim, dropout))
            in_dim = hidden

        # node head: node + graph context
        self.node_head = nn.Sequential(
            nn.Linear(hidden + hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # graph head: pooled + graph_feat
        self.graph_head = nn.Sequential(
            nn.Linear(hidden + graph_feat_dim, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, aa_idx, extra, edge_index,
                edge_attr, batch, graph_feat):
        # sanitise inputs
        extra      = torch.nan_to_num(extra,      nan=0.0)
        edge_attr  = torch.nan_to_num(edge_attr,  nan=0.0)
        graph_feat = torch.nan_to_num(graph_feat, nan=0.0)

        # normalise graph_feat (safe std)
        gf_mean = graph_feat.mean(dim=-1, keepdim=True)
        gf_std  = graph_feat.std(dim=-1,  keepdim=True).clamp(min=1.0)
        gf_norm = (graph_feat - gf_mean) / gf_std

        # node encoding
        x = self.node_encoder(aa_idx, extra)   # (N, aa_feat_dim)
        x = torch.nan_to_num(x, nan=0.0)

        # SAGE layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.nan_to_num(x, nan=0.0)

        # broadcast graph context to nodes
        g_ctx   = self.graph_feat_proj(gf_norm)   # (B, hidden)
        g_broad = g_ctx[batch]                     # (N, hidden)

        # node prediction
        node_logits = self.node_head(
            torch.cat([x, g_broad], dim=-1)).squeeze(-1)  # (N,)

        # graph pooling (manual mean pool — no PyG)
        B       = graph_feat.shape[0]
        g_pool  = torch.zeros(B, x.shape[-1], device=x.device)
        cnt     = torch.zeros(B, 1, device=x.device)
        g_pool.scatter_add_(
            0, batch.unsqueeze(-1).expand_as(x), x)
        cnt.scatter_add_(
            0, batch.unsqueeze(-1),
            torch.ones(len(batch), 1, device=x.device))
        g_pool  = g_pool / cnt.clamp(min=1)        # (B, hidden)

        graph_pred = self.graph_head(
            torch.cat([g_pool, gf_norm], dim=-1)).squeeze(-1)  # (B,)

        return node_logits, graph_pred


# ----------------------------------------------------------------
# simple batching — no PyG DataLoader needed
# ----------------------------------------------------------------
class GraphBatch:
    """Manual graph batching — replaces PyG DataLoader."""
    def __init__(self, graphs: list):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    @staticmethod
    def collate(graphs: list):
        """Merge list of graph dicts into one batched dict."""
        aa_idx_list, extra_list, y_list    = [], [], []
        ei_list, ea_list, gf_list, yg_list = [], [], [], []
        batch_list = []
        node_offset = 0

        for i, g in enumerate(graphs):
            n = g['aa_idx'].shape[0]
            aa_idx_list.append(g['aa_idx'])
            extra_list.append(g['extra'])
            y_list.append(g['y'])
            # offset edge indices
            ei_list.append(g['edge_index'] + node_offset)
            ea_list.append(g['edge_attr'])
            gf_list.append(g['graph_feat'])
            yg_list.append(g['y_graph'])
            batch_list.append(torch.full((n,), i, dtype=torch.long))
            node_offset += n

        return {
            'aa_idx':     torch.cat(aa_idx_list),
            'extra':      torch.cat(extra_list),
            'y':          torch.cat(y_list),
            'edge_index': torch.cat(ei_list, dim=1),
            'edge_attr':  torch.cat(ea_list),
            'graph_feat': torch.stack(gf_list),
            'y_graph':    torch.cat(yg_list),
            'batch':      torch.cat(batch_list),
            'num_graphs': len(graphs),
        }


class SimpleDataLoader:
    """Pure PyTorch DataLoader replacement."""
    def __init__(self, graphs, batch_size=4, shuffle=True):
        self.graphs     = graphs
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __iter__(self):
        idx = list(range(len(self.graphs)))
        if self.shuffle:
            import random
            random.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield GraphBatch.collate(
                [self.graphs[i] for i in batch_idx])

    def __len__(self):
        return (len(self.graphs) + self.batch_size - 1) \
               // self.batch_size


# ----------------------------------------------------------------
# dataset builder
# ----------------------------------------------------------------
def build_dataset(store: dict) -> list:
    """
    Build list of graph dicts from feature_store.pkl.
    No PyG Data objects — plain dicts with tensors.
    """
    EXTRA_COLS  = [c for c in NodeEncoder.EXTRA_FEAT_COLS
                   if c != 'is_interface']
    INTERACT_ORD = {t: i/9 for i, t in enumerate(
        ['HBOND','IONIC','PICATION','PIPISTACK',
         'VDW','SSBOND','IAC','MC_MC','MC_SC','SC_SC'])}
    DONOR_ORD = {t: i/3 for i, t in enumerate(
        ['MC','SC','LIG','UNK'])}

    dataset = []
    for sample, feats in store.items():
        rf_all    = feats['res_feats']
        edges_all = feats['edges']
        if rf_all.empty:
            continue

        for frame in sorted(rf_all['frame'].unique()):
            rf  = rf_all[rf_all['frame'] == frame].copy()
            edg = edges_all[edges_all['frame'] == frame].copy()
            if rf.empty:
                continue

            # normalise
            max_pos = float(rf['resid'].max()) or 1.
            rf['pos_norm']     = rf['resid']   / max_pos
            rf['bfactor_norm'] = rf['bfactor'] / 100.
            rf['x_norm']       = rf['x']       / 100.
            rf['y_norm']       = rf['y']       / 100.
            rf['z_norm']       = rf['z']       / 100.
            rf['chain_flag']   = (rf['chain'] != 'A').astype(float)
            rf['ebind_norm']   = rf['ebind']   / 100.

            # fill missing cols
            for c in EXTRA_COLS:
                if c not in rf.columns:
                    rf[c] = 0.

            aa_idx = resdf_to_aa_idx(rf)
            extra  = torch.tensor(
                rf[EXTRA_COLS].fillna(0.).values.astype(np.float32))
            extra  = torch.nan_to_num(extra, nan=0.0)

            y_node = torch.tensor(
                rf['is_interface'].fillna(0).values.astype(np.float32))

            # edges
            node_ids = rf['node_id'].tolist()
            node_idx = {nid: i for i, nid in enumerate(node_ids)}
            src, dst, efeats = [], [], []

            for _, row in edg.iterrows():
                s = row.get('NodeId1') or row.get('Residue1')
                d = row.get('NodeId2') or row.get('Residue2')
                if s not in node_idx or d not in node_idx:
                    continue
                src.append(node_idx[s])
                dst.append(node_idx[d])
                efeats.append([
                    INTERACT_ORD.get(str(row.get('Interaction','')), 0.),
                    DONOR_ORD.get(str(row.get('Donor','')), 1.),
                    float(row.get('Distance', 0.)) / 10.,
                    float(row.get('Angle',    0.)) / 180.,
                ])

            if not src:
                continue

            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr  = torch.tensor(efeats, dtype=torch.float)
            edge_attr  = torch.nan_to_num(edge_attr, nan=0.0)

            graph_feat = torch.tensor(
                feats['graph_feat'], dtype=torch.float)
            graph_feat = torch.nan_to_num(graph_feat, nan=0.0)

            dataset.append({
                'aa_idx':     aa_idx,
                'extra':      extra,
                'y':          y_node,
                'edge_index': edge_index,
                'edge_attr':  edge_attr,
                'graph_feat': graph_feat,
                'y_graph':    torch.tensor(
                    [feats['target']], dtype=torch.float),
                'sample':     sample,
                'frame':      frame,
            })

    print(f"Dataset: {len(dataset)} graphs")
    return dataset


# ----------------------------------------------------------------
# debug — check for NaN at each layer
# ----------------------------------------------------------------
def debug_forward(model, dataset):
    print("=== DEBUG FORWARD ===")
    g = GraphBatch.collate(dataset[:2])

    with torch.no_grad():
        extra      = torch.nan_to_num(g['extra'], nan=0.0)
        edge_attr  = torch.nan_to_num(g['edge_attr'], nan=0.0)
        graph_feat = torch.nan_to_num(g['graph_feat'], nan=0.0)

        def tstats(name, t):
            print(f"  {name:25s} shape={tuple(t.shape)} "
                  f"min={t.min():.3f} max={t.max():.3f} "
                  f"nan={torch.isnan(t).sum().item()}")

        tstats("extra",      extra)
        tstats("edge_attr",  edge_attr)
        tstats("graph_feat", graph_feat)

        x = model.node_encoder(g['aa_idx'], extra)
        tstats("node_encoder", x)

        for i, conv in enumerate(model.convs):
            x = conv(x, g['edge_index'], edge_attr)
            tstats(f"conv{i}", x)
            if torch.isnan(x).any():
                print(f"  ^^^ NaN first at conv{i} — stopping")
                return

        gf_std  = graph_feat.std(dim=-1, keepdim=True).clamp(min=1.0)
        gf_mean = graph_feat.mean(dim=-1, keepdim=True)
        gf_norm = (graph_feat - gf_mean) / gf_std
        tstats("graph_feat_norm", gf_norm)

        g_ctx = model.graph_feat_proj(gf_norm)
        tstats("graph_feat_proj", g_ctx)

    print("=== DEBUG DONE ===")


# ----------------------------------------------------------------
# training
# ----------------------------------------------------------------
def train_epoch(model, loader, optimizer,
                lambda_node=1.0, lambda_graph=0.1):
    model.train()
    total_loss  = 0.
    nan_batches = 0

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        node_logits, graph_pred = model(
            batch['aa_idx'], batch['extra'],
            batch['edge_index'], batch['edge_attr'],
            batch['batch'], batch['graph_feat'])

        if torch.isnan(node_logits).any():
            nan_batches += 1
            continue

        loss_node = F.binary_cross_entropy_with_logits(
            node_logits, batch['y'],
            pos_weight=torch.tensor([2.0]))
        loss_graph = F.mse_loss(
            graph_pred, batch['y_graph'].view(-1))
        loss = lambda_node * loss_node + lambda_graph * loss_graph

        if torch.isnan(loss):
            nan_batches += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if nan_batches:
        print(f"[WARN] {nan_batches} NaN batches skipped")
    return total_loss / max(len(loader) - nan_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        node_logits, _ = model(
            batch['aa_idx'], batch['extra'],
            batch['edge_index'], batch['edge_attr'],
            batch['batch'], batch['graph_feat'])

        probs = torch.sigmoid(node_logits).cpu()
        probs = torch.nan_to_num(probs, nan=0.5)
        all_logits.append(probs)
        all_labels.append(batch['y'].cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    logits = np.nan_to_num(logits, nan=0.5)

    if len(np.unique(labels)) < 2:
        return 0.5, 0.0

    preds = (logits > 0.5).astype(int)
    auc   = roc_auc_score(labels, logits)
    f1    = f1_score(labels, preds, zero_division=0)
    return auc, f1


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------
def get_args():
    p = ArgumentParser()
    p.add_argument('--store',    default='feature_store_boltzmann.pkl')
    p.add_argument('--outdir',   default='gnn_results')
    p.add_argument('--epochs',   default=100,  type=int)
    p.add_argument('--lr',       default=1e-4, type=float)
    p.add_argument('--hidden',   default=64,   type=int)
    p.add_argument('--n_layers', default=3,    type=int)
    p.add_argument('--batch',    default=4,    type=int)
    p.add_argument('--dropout',  default=0.15, type=float)
    p.add_argument('--feat_dim', default=32,   type=int)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.store, 'rb') as f:
        store = pickle.load(f)

    dataset = build_dataset(store)

    # debug first
    model_tmp = BindingSAGE(
        aa_feat_dim    = args.feat_dim,
        hidden         = args.hidden,
        n_layers       = args.n_layers,
        dropout        = args.dropout,
        graph_feat_dim = 6)
    debug_forward(model_tmp, dataset)

    # sample-stratified split
    all_samples = list(store.keys())
    n_test      = max(1, len(all_samples) // 5)
    rng         = np.random.default_rng(42)
    test_samps  = set(rng.choice(all_samples, n_test, replace=False))

    train_data = [g for g in dataset if g['sample'] not in test_samps]
    test_data  = [g for g in dataset if g['sample'] in test_samps]
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    train_loader = SimpleDataLoader(train_data,
                                     batch_size=args.batch, shuffle=True)
    test_loader  = SimpleDataLoader(test_data,
                                     batch_size=args.batch, shuffle=False)

    model = BindingSAGE(
        aa_feat_dim    = args.feat_dim,
        hidden         = args.hidden,
        n_layers       = args.n_layers,
        dropout        = args.dropout,
        graph_feat_dim = 6)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    train_losses, test_aucs, test_f1s = [], [], []
    best_auc  = 0.
    best_path = os.path.join(args.outdir, 'best_sage.pt')

    for epoch in range(1, args.epochs + 1):
        loss    = train_epoch(model, train_loader, optimizer)
        auc, f1 = eval_epoch(model, test_loader)
        scheduler.step()

        train_losses.append(loss)
        test_aucs.append(auc)
        test_f1s.append(f1)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | "
                  f"AUC={auc:.3f} | F1={f1:.3f} | best={best_auc:.3f}")

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(train_losses); axes[0].set_title('Train loss')
    axes[1].plot(test_aucs);    axes[1].set_title('Test AUC')
    axes[2].plot(test_f1s);     axes[2].set_title('Test F1')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'training_sage.png'), dpi=150)
    plt.close()

    print(f"\nBest AUC: {best_auc:.3f}")
    print(f"Saved: {best_path}")

    with open(os.path.join(args.outdir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
