# update_and_classify.py
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from classify_interface import (
    add_boltzmann_features,
    classifier,
)


def get_args():
    p = ArgumentParser()
    p.add_argument('--store',       default='feature_store.pkl')
    p.add_argument('--store_out',   default='feature_store_boltzmann.pkl')
    p.add_argument('--classifier',  default='RF',
                   choices=['DT','RF','SVM','GB','LR'])
    p.add_argument('--target',      default='is_interface')
    p.add_argument('--temp_tm',     default=0.1,   type=float)
    p.add_argument('--temp_e',      default=0.593, type=float)
    p.add_argument('--alpha',       default=0.5,   type=float)
    p.add_argument('--test_size',   default=0.3,   type=float)
    p.add_argument('--n_splits',    default=5,     type=int)
    p.add_argument('--outdir',      default='classifier_results')
    return p.parse_args()


def update_store(store:   dict,
                 temp_tm: float = 0.1,
                 temp_e:  float = 0.593,
                 alpha:   float = 0.5) -> dict:
    """
    Add Boltzmann features to res_feats in each sample of the store.
    Operates in-place on a copy.
    """
    updated = 0
    for sample, feats in store.items():
        rf = feats['res_feats']
        if rf.empty:
            print(f"[SKIP] {sample} — empty res_feats")
            continue
        store[sample]['res_feats'] = add_boltzmann_features(
            rf, temp_tm=temp_tm, temp_e=temp_e, alpha=alpha)
        updated += 1
        print(f"[OK] {sample:45s} | "
              f"rows={len(rf)} | "
              f"new cols: w_tm, w_ebind, w_combined, "
              f"e_combined_boltzmann, binding_score")

    print(f"\nUpdated {updated}/{len(store)} samples")
    return store


def build_df_from_store(store: dict) -> pd.DataFrame:
    records = []
    for sample, feats in store.items():
        rf = feats['res_feats']
        if rf.empty:
            continue

        # exclude groupby keys from aggregation cols
        groupby_keys = ['sample', 'resid', 'chain', 'restype']
        num_cols = [c for c in rf.select_dtypes(include=np.number).columns
                    if c not in groupby_keys]

        agg = rf.groupby(groupby_keys)[num_cols].mean().reset_index()
        records.append(agg)

    df = pd.concat(records, ignore_index=True)
    print(f"\nFeature DataFrame: {df.shape} | "
          f"{df['sample'].nunique()} samples | "
          f"pos={int(df['is_interface'].sum())} | "
          f"neg={int((df['is_interface']==0).sum())}")
    return df

def main():
    args = get_args()

    # 1. load existing pkl
    print(f"Loading {args.store}...")
    with open(args.store, 'rb') as f:
        store = pickle.load(f)
    print(f"Loaded: {len(store)} samples")

    # 2. add Boltzmann features to each sample's res_feats
    print(f"\nAdding Boltzmann features "
          f"(temp_tm={args.temp_tm}, "
          f"temp_e={args.temp_e}, "
          f"alpha={args.alpha})...")
    store = update_store(store,
                          temp_tm=args.temp_tm,
                          temp_e=args.temp_e,
                          alpha=args.alpha)

    # 3. save updated pkl
    print(f"\nSaving updated store → {args.store_out}")
    with open(args.store_out, 'wb') as f:
        pickle.dump(store, f)
    print(f"Saved: {args.store_out}")

    # 4. build flat DataFrame for classifier
    df = build_df_from_store(store)

    # verify new columns exist
    new_cols = ['w_tm','w_contact','w_rew','w_ebind',
                'w_combined','e_tm_boltzmann','e_ew_boltzmann',
                'e_combined_boltzmann','binding_score']
    present  = [c for c in new_cols if c in df.columns]
    missing  = [c for c in new_cols if c not in df.columns]
    print(f"\nNew Boltzmann cols present: {present}")
    if missing:
        print(f"Missing: {missing}")

    # 5. define feature sets
    base_feats = [
        #'e_mean','e_std','e_min','e_freq','e_reweighted',
        'bfactor','tm1','ebind',
        'charge','hydrophobicity','mol_weight',
        'h_donors','h_acceptors',
    ]
    boltzmann_feats = [
        'w_tm','w_contact','w_rew','w_ebind','w_combined',
        'e_tm_boltzmann','e_ew_boltzmann',
        'e_combined_boltzmann','binding_score',
    ]
    all_feats = [f for f in base_feats + boltzmann_feats
                 if f in df.columns]

    print(f"\nTotal features for classifier: {len(all_feats)}")
    print(all_feats)

    # 6. run classifier
    result = classifier(
        df        = df,
        feats     = all_feats,
        target    = args.target,
        clf_name  = args.classifier,
        test_size = args.test_size,
        n_splits  = args.n_splits,
        outdir    = args.outdir,
    )

    # 7. print top features
    clf = result['model'].named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'feature':    all_feats,
            'importance': clf.feature_importances_,
        }).sort_values('importance', ascending=False)

        print("\nTop 10 features:")
        print(imp_df.head(10).to_string(index=False))

        imp_df.to_csv(
            f"{args.outdir}/feature_importance.csv",
            index=False)


if __name__ == '__main__':
    main()
