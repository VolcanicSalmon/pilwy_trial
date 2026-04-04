import torch
import torch.nn as nn
import pandas as pd

class AAEmbedding(nn.Module):
    
    # map 3-letter to 1-letter for lookup
    AA3TO1 = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
        'UNK':'-',
    }

    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY#-'
    AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}

    def __init__(self, feat_dim: int, infeat_dim: int = 123):
        super(AAEmbedding, self).__init__()

        hydropathy = {'-': 0, '#': 0,
                      'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,
                      'M':1.9,'A':1.8,'W':-0.9,'G':-0.4,'T':-0.7,
                      'S':-0.8,'Y':-1.3,'P':-1.6,'H':-3.2,'N':-3.5,
                      'D':-3.5,'Q':-3.5,'E':-3.5,'K':-3.9,'R':-4.5}
        volume     = {'-': 0, '#': 0,
                      'G':60.1,'A':88.6,'S':89.0,'C':108.5,'D':111.1,
                      'P':112.7,'N':114.1,'T':116.1,'E':138.4,'V':140.0,
                      'Q':143.8,'H':153.2,'M':162.9,'I':166.7,'L':166.7,
                      'K':168.6,'R':173.4,'F':189.9,'Y':193.6,'W':227.8}
        charge     = {**{'R':1,'K':1,'D':-1,'E':-1,'H':0.1},
                      **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#-'}}
        polarity   = {**{x: 1 for x in 'RNDQEHKSTY'},
                      **{x: 0 for x in 'ACGILMFPWV#-'}}
        acceptor   = {**{x: 1 for x in 'DENQHSTY'},
                      **{x: 0 for x in 'RKWACGILMFPV#-'}}
        donor      = {**{x: 1 for x in 'RKWNQHSTY'},
                      **{x: 0 for x in 'DEACGILMFPV#-'}}

        self.register_buffer('embedding', torch.tensor([
            [hydropathy[self.ALPHABET[i]],
             volume[self.ALPHABET[i]] / 100.,
             charge[self.ALPHABET[i]],
             polarity[self.ALPHABET[i]],
             acceptor[self.ALPHABET[i]],
             donor[self.ALPHABET[i]]]
            for i in range(len(self.ALPHABET))
        ], dtype=torch.float))   # (22, 6)

        # MLP: infeat_dim → feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim),   nn.ReLU(),
            nn.Linear(feat_dim,     feat_dim),   nn.ReLU(),
            nn.Linear(feat_dim,     feat_dim),
        )
        self.feat_dim    = feat_dim
        self.infeat_dim  = infeat_dim

    
    def _rbf(self, D: torch.Tensor,D_min: float, D_max: float,
              stride: float) -> torch.Tensor:
        
        #returns: (N, K) RBF expansion
        
        D=torch.nan_to_num(D,nan=0.0)
        D=D.clamp(D_min-stride,D_max+stride)
        D_count = int((D_max - D_min) / stride)
        D_mu    = torch.linspace(D_min, D_max, D_count,
                                  device=D.device)   # (K,)
        D_exp   = D.unsqueeze(-1)                    # (N, 1)
        return torch.exp(-((D_exp - D_mu) / stride) ** 2)  # (N, K)

    def _transform(self, aa_vecs: torch.Tensor) -> torch.Tensor:
        """
        aa_vecs: (N, 6)
        returns: (N, infeat_dim=123)
        """
        return torch.cat([
            self._rbf(aa_vecs[:, 0], -4.5, 4.5,  0.1),   # hydropathy → 90
            self._rbf(aa_vecs[:, 1],  0.0, 2.2,  0.1),   # volume     → 22
            self._rbf(aa_vecs[:, 2], -1.0, 1.0,  0.25),  # charge     →  8
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),        # pol/acc/don →  3
        ], dim=-1)   # (N, 123)

   
    @classmethod
    def resname_to_idx(cls, resname3: str) -> int:
        aa1 = cls.AA3TO1.get(resname3.upper(), '-')
        return cls.AA_TO_IDX.get(aa1, cls.AA_TO_IDX['-'])
  
    def forward(self, x: torch.Tensor,
                raw: bool = False) -> torch.Tensor:
        """
        x   : (N,) long tensor of AA indices (from resname_to_idx)
        raw : if True return (N, 6) physicochemical vecs before RBF
        returns: (N, feat_dim)
        """
        aa_vecs  = self.embedding[x]          # (N, 6)
        if raw:
            return aa_vecs
        rbf_vecs = self._transform(aa_vecs)   # (N, 123)
        return self.mlp(rbf_vecs)             # (N, feat_dim)

    def soft_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 22) soft assignment over alphabet (e.g. from Gumbel-softmax)
        returns: (N, infeat_dim) RBF features without MLP
        """
        aa_vecs  = torch.matmul(x, self.embedding)   # (N, 6)
        rbf_vecs = torch.cat([
            self._rbf(aa_vecs[:, 0], -4.5, 4.5,  0.1),
            self._rbf(aa_vecs[:, 1],  0.0, 2.2,  0.1),
            self._rbf(aa_vecs[:, 2], -1.0, 1.0,  0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)
        return rbf_vecs

    def dim(self) -> int:
        return self.feat_dim



class NodeEncoder(nn.Module):
    """
    Full node encoder combining:
      AAEmbedding (feat_dim)     ← residue identity via RBF
      + structural/dynamic feats ← dSASA, BSE energy, B-factor,
                                   xyz, TM-score, chain, interface flag

    Output: (N, feat_dim + n_extra_feats)
    """
    # columns from res_feat_df that are NOT AA identity
    EXTRA_FEAT_COLS = [
        'pos_norm',       # sequence position
        'bfactor_norm',   # B-factor
        'x_norm', 'y_norm', 'z_norm',   # coordinates
        #'e_mean', 'e_std',
        'e_min',
        #'e_freq', 
        'e_reweighted',  # BSE
        'lambda_B',       # Bjerrum length
        'tm1',            # per-frame TM-score
        'ebind_norm',     # per-frame GBSA
        'chain_flag',     # 0=patho, 1=plant
        'w_tm', 'w_ebind', 'w_combined', 'binding_score',
    ]   # 15 extra dims

    def __init__(self, feat_dim: int = 32):
        super(NodeEncoder, self).__init__()
        self.aa_embed   = AAEmbedding(feat_dim=feat_dim)
        self.extra_proj = nn.Sequential(
            nn.Linear(len(self.EXTRA_FEAT_COLS), feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.feat_dim = feat_dim

    def forward(self, aa_idx: torch.Tensor,
                extra:  torch.Tensor) -> torch.Tensor:
        """
        aa_idx : (N,)      long  — AA index from AAEmbedding.resname_to_idx
        extra  : (N, 15)   float — structural/dynamic features
        returns: (N, feat_dim)
        """
        aa_emb    = self.aa_embed(aa_idx)    # (N, feat_dim)
        extra_emb = self.extra_proj(extra)   # (N, feat_dim)
        fused     = self.fuse(
            torch.cat([aa_emb, extra_emb], dim=-1))  # (N, feat_dim)
        return fused


# ----------------------------------------------------------------
# updated store_to_graphs — adds aa_idx to each Data object
# ----------------------------------------------------------------

def resdf_to_aa_idx(res_df: pd.DataFrame) -> torch.Tensor:
    """Convert restype column → integer index tensor for AAEmbedding."""
    return torch.tensor(
        [AAEmbedding.resname_to_idx(r)
         for r in res_df['restype'].values],
        dtype=torch.long)
