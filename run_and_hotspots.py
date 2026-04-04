
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, Selection
import MDAnalysis as mda
from protein_domain_segmentation import MerizoCluster
from argparse import ArgumentParser


def get_args():
    p = ArgumentParser()
    p.add_argument('--store',       default='feature_store_boltzmann.pkl')
    p.add_argument('--gnn_clusters',default='gnn_viz/graph_clusters.csv')
    p.add_argument('--outdir',      default='hotspot_results')
    p.add_argument('--alpha',       default=0.5,  type=float,
                   help='TM weight in combined score (0=energy, 1=TM)')
    p.add_argument('--top_n',       default=5,    type=int,
                   help='Top N complexes for Merizo')
    p.add_argument('--pathochain',  default='A')
    p.add_argument('--plantchain',  default='B')
    p.add_argument('--cutoff',      default=5.0,  type=float,
                   help='Interface cutoff in Angstrom')
    return p.parse_args()


# ----------------------------------------------------------------
# 1. ranking with tunable alpha
# ----------------------------------------------------------------
def rank_complexes(store:       dict,
                   alpha:       float = 0.5,
                   gnn_clusters: str  = None) -> pd.DataFrame:
    """
    Rank complexes by combined TM + binding energy score.

    score = alpha * z(TM) + (1-alpha) * z(-ebind)

    Using z-scores ensures both terms are on the same scale
    before combining — avoids TM (0-1) being dominated by
    ebind (-60 to +50 kcal/mol).

    alpha=1.0 → rank by structural similarity only
    alpha=0.0 → rank by binding energy only
    alpha=0.5 → equal balance
    """
    records = []
    for sample, feats in store.items():
        gf = feats['graph_feat']
        records.append({
            'sample':    sample,
            'tm_mean':   float(gf[0]),
            'tm_max':    float(gf[1]),
            'ebind_mean':float(gf[3]),
            'ebind_std': float(gf[4]),
            'ebind_min': float(gf[5]),
            'n_interface': int(feats['res_feats']['is_interface'].sum())
                           if not feats['res_feats'].empty else 0,
        })

    df = pd.DataFrame(records)

    # z-score normalise — handles different scales
    df['z_tm']    =  zscore(df['tm_mean'])
    df['z_ebind'] = -zscore(df['ebind_mean'])  # negate: more negative = better

    # combined score
    df['combined_score'] = (alpha       * df['z_tm'] +
                             (1 - alpha) * df['z_ebind'])

    # add GNN cluster if available
    if gnn_clusters and os.path.exists(gnn_clusters):
        cl_df = pd.read_csv(gnn_clusters)
        # aggregate cluster per sample (most common)
        cl_agg = cl_df.groupby('sample')['cluster'].agg(
            lambda x: x.mode()[0]).reset_index()
        df = df.merge(cl_agg, on='sample', how='left')
    else:
        df['cluster'] = -1

    df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    print(f"\n=== Complex ranking (alpha={alpha}) ===")
    print(f"{'Rank':4s} {'Sample':45s} {'TM':6s} {'ebind':8s} "
          f"{'z_tm':6s} {'z_e':6s} {'score':7s} {'cluster':7s}")
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{int(row['rank']):4d} {row['sample']:45s} "
              f"{row['tm_mean']:6.3f} {row['ebind_mean']:8.2f} "
              f"{row['z_tm']:6.3f} {row['z_ebind']:6.3f} "
              f"{row['combined_score']:7.3f} "
              f"{str(int(row['cluster'])) if row['cluster']>=0 else '-':7s}")
    return df


def plot_ranking(df: pd.DataFrame, outdir: str, alpha: float):
    """Visualise the TM vs ebind tradeoff space."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # TM vs ebind scatter — colour by combined score
    sc = axes[0].scatter(df['tm_mean'], df['ebind_mean'],
                          c=df['combined_score'], cmap='RdYlGn',
                          s=60, zorder=3)
    plt.colorbar(sc, ax=axes[0], label='Combined score')
    for _, row in df.iterrows():
        axes[0].annotate(
            row['sample'].split('_pp2a')[0],
            (row['tm_mean'], row['ebind_mean']),
            fontsize=5, ha='left', va='bottom')
    axes[0].set_xlabel('TM-score (structural similarity)')
    axes[0].set_ylabel('ebind_mean (kcal/mol)')
    axes[0].set_title(f'TM vs binding energy (α={alpha})')
    axes[0].axhline(0, color='grey', linestyle='--', alpha=0.5)

    # alpha sweep — how ranking changes
    alphas  = np.linspace(0, 1, 11)
    top5_at_alpha = []
    for a in alphas:
        z_tm    =  zscore(df['tm_mean'])
        z_e     = -zscore(df['ebind_mean'])
        scores  = a * z_tm + (1-a) * z_e
        top5    = df['sample'].iloc[np.argsort(-scores)[:5]].tolist()
        top5_at_alpha.append(top5)

    # show which samples are in top5 across alpha sweep
    all_top = sorted(set(s for t in top5_at_alpha for s in t))
    mat     = np.zeros((len(all_top), len(alphas)))
    for j, top5 in enumerate(top5_at_alpha):
        for s in top5:
            i = all_top.index(s)
            mat[i, j] = 1

    axes[1].imshow(mat, aspect='auto', cmap='Blues',
                    vmin=0, vmax=1)
    axes[1].set_xticks(range(len(alphas)))
    axes[1].set_xticklabels([f'{a:.1f}' for a in alphas],
                              rotation=45, fontsize=7)
    axes[1].set_yticks(range(len(all_top)))
    axes[1].set_yticklabels(
        [s.split('_pp2a')[0] for s in all_top], fontsize=6)
    axes[1].set_xlabel('Alpha (TM weight)')
    axes[1].set_title('Top-5 membership across α sweep')

    # combined score bar chart
    df_plot = df.sort_values('combined_score', ascending=True)
    colors  = ['steelblue' if '0_pipsr2' not in s else 'coral'
               for s in df_plot['sample']]
    axes[2].barh(range(len(df_plot)), df_plot['combined_score'],
                  color=colors)
    axes[2].set_yticks(range(len(df_plot)))
    axes[2].set_yticklabels(
        [s.split('_pp2a')[0] for s in df_plot['sample']],
        fontsize=6)
    axes[2].set_xlabel('Combined score')
    axes[2].set_title('Ranking')
    axes[2].axvline(0, color='grey', linestyle='--', alpha=0.5)

    plt.suptitle(f'Complex ranking (alpha={alpha})', fontsize=11)
    plt.tight_layout()
    outpng = os.path.join(outdir, f'ranking_alpha{alpha:.1f}.png')
    plt.savefig(outpng, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpng}")


# ----------------------------------------------------------------
# 2. contact_surf (interface residues)
# ----------------------------------------------------------------
def contact_surf(pdbin, pathochain='A', plantchain='B',
                 cutoff=5.0) -> dict:
    parser  = PDBParser(QUIET=True)
    struct  = parser.get_structure(
        os.path.basename(pdbin).replace('.pdb',''), pdbin)

    pathoatoms  = Selection.unfold_entities(struct[0][pathochain], 'A')
    plantatoms  = Selection.unfold_entities(struct[0][plantchain], 'A')
    pathocoords = np.array([a.coord for a in pathoatoms])
    plantcoords = np.array([a.coord for a in plantatoms])

    pairs = cKDTree(pathocoords).query_ball_tree(
        cKDTree(plantcoords), cutoff)

    patho_res, plant_res, pairlist = set(), set(), []
    for pidx, qidxs in enumerate(pairs):
        if not qidxs:
            continue
        pr = pathoatoms[pidx].get_parent().get_id()[1]
        patho_res.add(pr)
        for qi in qidxs:
            qr = plantatoms[qi].get_parent().get_id()[1]
            plant_res.add(qr)
            pairlist.append((pr, qr))

    return {
        'patho_resids': sorted(patho_res),
        'plant_resids': sorted(plant_res),
        'pairs':        list(set(pairlist)),
    }


# ----------------------------------------------------------------
# 3. Merizo domain segmentation
def run_merizo(pdbin: str, outdir: str) -> tuple:
    """Returns (domain_map, domain_ranges_str)"""
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, 'patho_merizo.pdb')

    u = mda.Universe(pdbin)
    u.select_atoms('protein').write(tmp)
    uni = mda.Universe(tmp)

    merizocls = MerizoCluster()
    try:
        result = merizocls.predict_from_universe(uni)
    except Exception as ex:
        print(f"  predict_from_universe failed: {ex}")
        try:
            result = merizocls.predict(tmp)
        except Exception as ex2:
            print(f"  predict failed: {ex2}")
            return {}, ''

    print(f"  result type : {type(result)}")
    print(f"  result repr : {repr(result)[:300]}")

    domain_map   = {}
    raw_str      = ''   # e.g. '1-104,105-263,264-519'

    if isinstance(result, str) and result.strip():
        raw_str  = result.strip()
        segments = raw_str.split(',')
        for dom_id, seg in enumerate(segments):
            seg = seg.strip()
            if '-' in seg:
                try:
                    start, end = seg.split('-')
                    for resid in range(int(start), int(end) + 1):
                        domain_map[resid] = dom_id
                except ValueError:
                    print(f"  [WARN] cannot parse segment: {seg}")
            else:
                try:
                    domain_map[int(seg)] = dom_id
                except ValueError:
                    pass

    elif isinstance(result, dict):
        if all(isinstance(k, int) for k in result.keys()):
            domain_map = {int(k): int(v) for k, v in result.items()}
        elif 'domains' in result:
            for dom_id, resids in enumerate(result['domains']):
                for r in resids:
                    domain_map[int(r)] = dom_id

    elif hasattr(result, 'residues') and len(result.residues) > 0:
        for res in result.residues:
            for attr in ['domain_id','domain','merizo_domain','segment']:
                val = getattr(res, attr, None)
                if val is not None:
                    try:
                        domain_map[int(res.resid)] = int(val)
                        break
                    except (ValueError, TypeError):
                        pass

    # build range string if not already from raw string
    if not raw_str and domain_map:
        dom_summary = {}
        for r, d in domain_map.items():
            dom_summary.setdefault(d, []).append(r)
        parts = []
        for d in sorted(dom_summary.keys()):
            rs = sorted(dom_summary[d])
            parts.append(f"{min(rs)}-{max(rs)}")
        raw_str = ','.join(parts)

    print(f"  Parsed: {len(domain_map)} residues | "
          f"{len(set(domain_map.values()))} domains")
    if domain_map:
        dom_summary = {}
        for r, d in domain_map.items():
            dom_summary.setdefault(d, []).append(r)
        for d, rs in sorted(dom_summary.items()):
            print(f"    domain {d}: residues {min(rs)}-{max(rs)} "
                  f"({len(rs)} residues)")

    return domain_map, raw_str
# ----------------------------------------------------------------
# 4. find best domain + interface overlap
# ----------------------------------------------------------------
# in find_hotspot_domain, also return the range of best domain
def find_hotspot_domain(iface, domain_map, store_feats,
                         pathochain='A') -> dict:
    patho_iface = set(iface['patho_resids'])

    if not domain_map:
        return {}

    domain_overlap = {}
    for resid, dom_id in domain_map.items():
        if resid in patho_iface:
            domain_overlap.setdefault(dom_id, []).append(resid)

    if not domain_overlap:
        print("  [WARN] no domain-interface overlap")
        return {}

    res_feats = store_feats.get('res_feats', pd.DataFrame())
    rew_map   = {}
    if not res_feats.empty and 'e_reweighted' in res_feats.columns:
        agg = res_feats[
            res_feats['chain'] == pathochain
        ].groupby('resid')['e_reweighted'].mean()
        rew_map = agg.to_dict()

    domain_scores = {}
    for dom_id, resids in domain_overlap.items():
        n       = len(resids)
        e_score = np.mean([rew_map.get(r, 0.) for r in resids])
        domain_scores[dom_id] = {
            'n_interface':    n,
            'mean_rew_energy':float(e_score),
            'domain_score':   n * abs(e_score) if e_score != 0 else n,
            'resids':         sorted(resids),
        }

    best_domain = max(domain_scores,
                      key=lambda d: domain_scores[d]['domain_score'])

    # get full range of best domain from domain_map
    best_dom_resids = sorted(
        [r for r, d in domain_map.items() if d == best_domain])
    best_dom_range  = (f"{min(best_dom_resids)}-{max(best_dom_resids)}"
                       if best_dom_resids else 'N/A')

    print(f"  Best domain: {best_domain} | "
          f"range={best_dom_range} | "
          f"n_interface={domain_scores[best_domain]['n_interface']} | "
          f"score={domain_scores[best_domain]['domain_score']:.3f}")

    return {
        'best_domain':       best_domain,
        'best_domain_range': best_dom_range,   # ← new
        'all_domains':       domain_scores,
        'domain_resids':     domain_map,
    }
# ----------------------------------------------------------------
# 5. BindCraft hotspot string
# ----------------------------------------------------------------
def make_bindcraft_hotspots(iface:       dict,
                             domain_info: dict,
                             store_feats: dict,
                             pathochain:  str   = 'A',
                             top_n:       int   = 15) -> dict:
    """
    Select top hotspot residues for BindCraft:
      1. Must be in interface (contact_surf)
      2. Must be in best Merizo domain
      3. Ranked by reweighted energy (most negative = strongest)

    Returns BindCraft-ready hotspot strings for both chains.
    """
    patho_iface = set(iface['patho_resids'])
    plant_iface = set(iface['plant_resids'])

    best_dom    = domain_info.get('best_domain')
    all_doms    = domain_info.get('all_domains', {})

    if best_dom is None:
        # fallback: use all interface residues
        domain_patho = patho_iface
    else:
        domain_patho = set(all_doms[best_dom]['resids'])

    # get reweighted energy per residue
    res_feats = store_feats.get('res_feats', pd.DataFrame())
    rew_map   = {}
    if not res_feats.empty and 'e_reweighted' in res_feats.columns:
        for chain, resid_set in [(pathochain, domain_patho)]:
            agg = res_feats[
                res_feats['chain'] == chain
            ].groupby('resid')['e_reweighted'].mean()
            rew_map.update(agg.to_dict())

    # select hotspots: domain ∩ interface, ranked by energy
    hotspot_resids = sorted(
        domain_patho & patho_iface,
        key=lambda r: rew_map.get(r, 0.))[:top_n]

    # plant side hotspots (all interface, no domain filter)
    plant_resids = sorted(
        plant_iface,
        key=lambda r: rew_map.get(r, 0.))[:top_n]

    # BindCraft format: chain+resid comma-separated
    patho_str = ','.join(f"{pathochain}{r}" for r in hotspot_resids)
    plant_str = ','.join(f"B{r}"            for r in plant_resids)

    return {
        'patho_hotspots':      hotspot_resids,
        'plant_hotspots':      plant_resids,
        'bindcraft_patho':     patho_str,
        'bindcraft_plant':     plant_str,
        'bindcraft_combined':  f"{patho_str},{plant_str}",
        'n_patho':             len(hotspot_resids),
        'n_plant':             len(plant_resids),
    }


# ----------------------------------------------------------------
# 6. main pipeline
# ----------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.store, 'rb') as f:
        store = pickle.load(f)

    # step 1: rank
    rank_df = rank_complexes(store, args.alpha, args.gnn_clusters)
    rank_df.to_csv(os.path.join(args.outdir, 'rankings.csv'), index=False)
    plot_ranking(rank_df, args.outdir, args.alpha)

    # step 2-5: Merizo + hotspots on top N
    top_samples = rank_df.head(args.top_n)['sample'].tolist()
    print(f"\nRunning Merizo on top {args.top_n}: {top_samples}")

    results = {}
    summary_rows = []

    for sample in top_samples:
        print(f"\n{'='*60}")
        print(f"Processing: {sample}")
        rank_row = rank_df[rank_df['sample']==sample].iloc[0]

        # find PDB
        pdbs = glob.glob(
            os.path.join(sample, 'abonly', 'frames', 'frame_0000.pdb'))
        if not pdbs:
            print(f"  [SKIP] no PDB found")
            continue
        pdbin    = pdbs[0]
        samp_out = os.path.join(args.outdir, sample)
        os.makedirs(samp_out, exist_ok=True)

        # interface residues
        try:
            iface = contact_surf(pdbin, args.pathochain,
                                  args.plantchain, args.cutoff)
            print(f"  Interface: {len(iface['patho_resids'])} patho | "
                  f"{len(iface['plant_resids'])} plant residues")
        except Exception as ex:
            print(f"  [ERROR contact_surf] {ex}")
            continue

        # Merizo
        try:
            domain_map, domain_ranges = run_merizo(
                pdbin, samp_out)
        except Exception as ex:
            print(f"  [ERROR Merizo] {ex}")
            domain_map = {}

        # domain-interface overlap
        domain_info = find_hotspot_domain(
            iface, domain_map,
            store[sample], args.pathochain)

        # BindCraft hotspots
        hotspots = make_bindcraft_hotspots(
            iface, domain_info, store[sample],
            args.pathochain, top_n=15)

        # save per-sample
        pd.DataFrame({
            'resid':     hotspots['patho_hotspots'],
            'chain':     args.pathochain,
            'type':      'patho_hotspot',
        }).to_csv(os.path.join(samp_out, 'hotspots.csv'), index=False)

        results[sample] = {
            'rank':         int(rank_row['rank']),
            'tm_mean':      float(rank_row['tm_mean']),
            'ebind_mean':   float(rank_row['ebind_mean']),
            'score':        float(rank_row['combined_score']),
            'iface':        iface,
            'domain_map':   domain_map,
            'domain_info':  domain_info,
            'hotspots':     hotspots,
        }

        summary_rows.append({
            'rank':              int(rank_row['rank']),
            'sample':            sample,
            'tm_mean':           float(rank_row['tm_mean']),
            'ebind_mean':        float(rank_row['ebind_mean']),
            'combined_score':    float(rank_row['combined_score']),
            'n_iface_patho':     len(iface['patho_resids']),
            'n_iface_plant':     len(iface['plant_resids']),
            'best_domain':       domain_info.get('best_domain', 'N/A'),
            'best_domain_range':  domain_info.get('best_domain_range', 'N/A'), 
            'all_domain_ranges':  domain_ranges,
            'n_hotspots':        hotspots['n_patho'],
            'bindcraft_patho':   hotspots['bindcraft_patho'],
            'bindcraft_plant':   hotspots['bindcraft_plant'],
            'bindcraft_combined':hotspots['bindcraft_combined'],
        })

        print(f"  BindCraft patho:  {hotspots['bindcraft_patho'][:60]}...")
        print(f"  BindCraft plant:  {hotspots['bindcraft_plant'][:60]}...")

    # save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, 'hotspot_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    # save full results
    with open(os.path.join(args.outdir, 'hotspot_store.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — BindCraft inputs:")
    print(f"{'='*60}")
    for row in summary_rows:
        print(f"\nRank {row['rank']}: {row['sample']}")
        print(f"  TM={row['tm_mean']:.3f} | "
              f"ebind={row['ebind_mean']:.2f} | "
              f"score={row['combined_score']:.3f}")
        print(f"  Domain: {row['best_domain']} | "
              f"hotspots: {row['n_hotspots']}")
        print(f"  Patho:  {row['bindcraft_patho'][:80]}")
        print(f"  Plant:  {row['bindcraft_plant'][:80]}")


if __name__ == '__main__':
    main()
