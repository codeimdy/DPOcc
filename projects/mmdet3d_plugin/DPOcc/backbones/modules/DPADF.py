import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Positional Encoding (sincos)
# ---------------------------
def _sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """
    pos: (...,) float tensor
    return: (..., dim) sin/cos encoding
    """
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = pos[..., None] * omega[None, ...]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


def posenc_2d(h: int, w: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    return: (H*W, dim)
    """
    assert dim % 2 == 0
    dim_half = dim // 2
    assert dim_half % 2 == 0
    y = torch.linspace(0, 1, steps=h, device=device, dtype=dtype)
    x = torch.linspace(0, 1, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H,W)
    pe_y = _sincos_1d(yy.reshape(-1), dim_half)   # (HW, dim/2)
    pe_x = _sincos_1d(xx.reshape(-1), dim_half)   # (HW, dim/2)
    return torch.cat([pe_y, pe_x], dim=-1)        # (HW, dim)


def posenc_3d(x: int, y: int, z: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    return: (X*Y*Z, dim)
    """
    assert dim % 3 == 0, "为了简单，这里要求 dim 能被3整除（每个轴一份）。"
    dim_each = dim // 3
    assert dim_each % 2 == 0

    xs = torch.linspace(0, 1, steps=x, device=device, dtype=dtype)
    ys = torch.linspace(0, 1, steps=y, device=device, dtype=dtype)
    zs = torch.linspace(0, 1, steps=z, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")  # (X,Y,Z)

    pex = _sincos_1d(xx.reshape(-1), dim_each)
    pey = _sincos_1d(yy.reshape(-1), dim_each)
    pez = _sincos_1d(zz.reshape(-1), dim_each)
    return torch.cat([pex, pey, pez], dim=-1)  # (S, dim)


# ---------------------------
# Voxel Pooling (scatter reduce)
# ---------------------------
class VoxelPooling(nn.Module):
    """
    把 (B, N, C) 的点/像素特征，按整数体素坐标 (B, N, 3) scatter 到体素网格中。
    输出:
      - voxel_feats: (B, S, C)  S = X*Y*Z
      - counts:     (B, S, 1)
      - occ_mask:   (B, S) bool
    """
    def __init__(self, grid_size: Tuple[int, int, int], reduce: str = "mean"):
        super().__init__()
        assert reduce in ("sum", "mean")
        self.grid_size = grid_size
        self.reduce = reduce

    def forward(self, feats: torch.Tensor, coords: torch.Tensor):
        """
        feats:  (B, N, C)
        coords: (B, N, 3) int64 in [0..X-1],[0..Y-1],[0..Z-1]
        """
        B, N, C = feats.shape
        X, Y, Z = self.grid_size
        S = X * Y * Z

        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        flat = (x * (Y * Z) + y * Z + z).long()  # (B,N)

        voxel_sum = feats.new_zeros((B, S, C))
        counts = feats.new_zeros((B, S, 1))

        voxel_sum.scatter_add_(1, flat[..., None].expand(-1, -1, C), feats)
        counts.scatter_add_(1, flat[..., None, None], torch.ones((B, N, 1), device=feats.device, dtype=feats.dtype))

        if self.reduce == "mean":
            voxel_feats = voxel_sum / counts.clamp_min(1.0)
        else:
            voxel_feats = voxel_sum

        occ_mask = (counts.squeeze(-1) > 0)
        return voxel_feats, counts, occ_mask


# ---------------------------
# Transformer block with self-attn + cross-attn + FFN
# ---------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,              # (B, S, C)
        memory: torch.Tensor,         # (B, L, C)
        q_pos: Optional[torch.Tensor] = None,      # (S, C) or (B,S,C)
        m_pos: Optional[torch.Tensor] = None,      # (L, C) or (B,L,C)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B,L) bool, True=pad
    ):
        # add position embedding (common trick)
        q_in = q + (q_pos if q_pos is not None else 0)
        m_in = memory + (m_pos if m_pos is not None else 0)

        # self-attn
        x, _ = self.self_attn(q_in, q_in, q_in)
        q = self.norm1(q + x)

        # cross-attn (q attends to memory)
        q_in = q + (q_pos if q_pos is not None else 0)
        x, _ = self.cross_attn(q_in, m_in, m_in, key_padding_mask=key_padding_mask)
        q = self.norm2(q + x)

        # ffn
        x = self.ffn(q)
        q = self.norm3(q + x)
        return q


class VoxelQueryTransformer(nn.Module):
    """
    图中的 Voxel Query Q + (Self-Attn, Cross-Attn, FFN, AddNorm)
    输出 O_I ∈ R^{X×Y×Z×C}
    """
    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_ff: int = 1024,
        dropout: float = 0.0,
        use_learnable_queries: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        X, Y, Z = grid_size
        self.S = X * Y * Z
        self.d_model = d_model

        if use_learnable_queries:
            self.query_embed = nn.Parameter(torch.randn(self.S, d_model) * 0.02)
        else:
            self.query_embed = None

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, feat2d: torch.Tensor) -> torch.Tensor:
        """
        feat2d: (B, C, H, W)  作为 memory
        return: O_I tokens (B, S, C)
        """
        B, C, H, W = feat2d.shape
        assert C == self.d_model

        memory = feat2d.flatten(2).transpose(1, 2)  # (B, L=HW, C)

        # 2D pos
        m_pos = posenc_2d(H, W, C, device=feat2d.device, dtype=feat2d.dtype)  # (L,C)
        m_pos = m_pos.unsqueeze(0).expand(B, -1, -1)  # (B,L,C)

        # 3D voxel query init
        if self.query_embed is None:
            q = torch.zeros((B, self.S, C), device=feat2d.device, dtype=feat2d.dtype)
        else:
            q = self.query_embed.unsqueeze(0).expand(B, -1, -1).to(feat2d.dtype)

        q_pos = posenc_3d(*self.grid_size, C, device=feat2d.device, dtype=feat2d.dtype)  # (S,C)
        q_pos = q_pos.unsqueeze(0).expand(B, -1, -1)  # (B,S,C)

        for layer in self.layers:
            q = layer(q, memory, q_pos=q_pos, m_pos=m_pos)

        return q  # (B,S,C)


# ---------------------------
# KNN Retrieve from O_I
# ---------------------------
class KNNRetriever(nn.Module):
    """
    对每个非空体素坐标，从 O_I (dense voxel tokens) 中找 K 个最近邻体素特征。
    注：这里用 torch.cdist 做“最直观”的参考实现，网格很大时会慢。
    """
    def __init__(self, grid_size: Tuple[int, int, int], k: int):
        super().__init__()
        self.grid_size = grid_size
        self.k = k

        X, Y, Z = grid_size
        xs = torch.arange(X)
        ys = torch.arange(Y)
        zs = torch.arange(Z)
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (S,3)
        self.register_buffer("grid_coords", coords, persistent=False)  # (S,3)

    @torch.no_grad()
    def forward(self, nonempty_coords: torch.Tensor, OI: torch.Tensor):
        """
        nonempty_coords: (M,3) int64 for one sample
        OI: (S,C) for one sample
        return:
          nn_feats: (M,K,C)
          nn_idx:   (M,K)
        """
        # (M,3) -> float
        q = nonempty_coords.to(OI.device).to(torch.float32)
        g = self.grid_coords.to(OI.device).to(torch.float32)  # (S,3)

        # distances (M,S)
        dist = torch.cdist(q, g, p=2.0)
        nn_idx = dist.topk(self.k, largest=False, dim=1).indices  # (M,K)
        nn_feats = OI[nn_idx]  # (M,K,C)
        return nn_feats, nn_idx


# ---------------------------
# CDFusion: MLP -> ReLU -> weights -> weighted sum -> concat
# ---------------------------
class CDFusion(nn.Module):
    """
    输入:
      - center_feat (来自 O_D 的非空体素特征): (M, C_d)
      - neigh_feat  (来自 O_I 的 KNN 特征):   (M, K, C_i)

    输出:
      - fused_feat: (M, C_i)  (加权融合的邻域信息)
      - out_feat:   (M, C_d + C_i)  与 center 进行 concat
    """
    def __init__(self, c_d: int, c_i: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_d + c_i, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),  # 输出每个邻居的打分
        )

    def forward(self, center_feat: torch.Tensor, neigh_feat: torch.Tensor):
        """
        center_feat: (M, C_d)
        neigh_feat:  (M, K, C_i)
        """
        M, K, C_i = neigh_feat.shape
        C_d = center_feat.shape[-1]

        center_exp = center_feat[:, None, :].expand(-1, K, -1)  # (M,K,C_d)
        x = torch.cat([center_exp, neigh_feat], dim=-1)         # (M,K,C_d+C_i)

        logits = self.mlp(x).squeeze(-1)                        # (M,K)
        weights = F.softmax(logits, dim=1)                      # (M,K)

        fused = torch.sum(weights[..., None] * neigh_feat, dim=1)  # (M,C_i)
        out = torch.cat([center_feat, fused], dim=-1)              # (M,C_d+C_i)
        return fused, out, weights


class DiagramNet(nn.Module):
    """
    结构对应：
      F_dep ⊗ F_con -> VoxelPooling -> O_D + NonEmpty coords C_D
      F_con -> VoxelQueryTransformer -> O_I
      C_D + O_I -> KNN Retrieve -> neigh_feat
      (center_feat from O_D) + neigh_feat -> CDFusion
      concat(O_D, fused_dense) -> O
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.grid_size = cfg.grid_size
        X, Y, Z = cfg.grid_size
        self.S = X * Y * Z

        # 2D -> d_model
        # 如果你的 F_con 通道 != d_model，可在这里用 1x1 conv 适配
        assert cfg.c_con == cfg.c_i, "为简单起见，这里要求 c_con == c_i（Transformer的d_model）"
        assert cfg.c_d == cfg.c_con, "为简单起见，这里要求 pooled通道 c_d == c_con"

        self.voxel_pool = VoxelPooling(cfg.grid_size, reduce="mean")

        self.voxel_query = VoxelQueryTransformer(
            grid_size=cfg.grid_size,
            d_model=cfg.c_i,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ff=cfg.dim_ff,
        )

        self.knn = KNNRetriever(cfg.grid_size, k=cfg.k)
        self.fusion = CDFusion(c_d=cfg.c_d, c_i=cfg.c_i, hidden=256)

    def forward(
        self,
        F_con: torch.Tensor,              # (B, C, H, W)
        F_dep: torch.Tensor,              # (B, 1, H, W) or (B, C, H, W)
        voxel_coords_map: torch.Tensor,   # (B, H, W, 3) int64
    ):
        B, C, H, W = F_con.shape
        X, Y, Z = self.grid_size
        S = self.S

        # 1) F_dep ⊗ F_con （图左上）
        if F_dep.shape[1] == 1:
            pooled_in = F_con * F_dep  # broadcast
        else:
            pooled_in = F_con * F_dep

        # 2) Voxel Pooling（图上支路）-> O_D, counts, occ
        feats = pooled_in.flatten(2).transpose(1, 2)              # (B, N=HW, C)
        coords = voxel_coords_map.reshape(B, H * W, 3).long()     # (B, HW, 3)

        O_D, counts, occ = self.voxel_pool(feats, coords)         # (B,S,C), (B,S,1), (B,S)

        # 3) Voxel Query Transformer（图下支路）-> O_I
        O_I = self.voxel_query(F_con)                              # (B,S,C_i)

        # 4) Retrieve KNN + CDFusion（对每个 batch 单独处理可变数量 non-empty）
        fused_dense = O_D.new_zeros((B, S, self.cfg.c_i))          # (B,S,C_i)

        all_weights: List[torch.Tensor] = []
        for b in range(B):
            occ_idx = torch.nonzero(occ[b], as_tuple=False).squeeze(1)  # (M,)
            if occ_idx.numel() == 0:
                all_weights.append(torch.empty(0, self.cfg.k, device=O_D.device))
                continue

            # non-empty voxel coords (M,3)
            # idx -> (x,y,z)
            x = occ_idx // (Y * Z)
            yz = occ_idx % (Y * Z)
            y = yz // Z
            z = yz % Z
            nonempty_coords = torch.stack([x, y, z], dim=-1).long()     # (M,3)

            # center features from O_D: (M, C_d)
            center_feat = O_D[b, occ_idx]                                # (M,C_d)

            # knn from O_I: (M,K,C_i)
            neigh_feat, _ = self.knn(nonempty_coords, O_I[b])            # (M,K,C_i)

            # CDFusion
            _, out_feat, w = self.fusion(center_feat, neigh_feat)        # out_feat: (M, C_d+C_i)
            all_weights.append(w)                                        # (M,K)

            # 这里只把融合后的 “邻域聚合结果” 写回 dense（你也可写 out_feat 或其它组合）
            fused_part = out_feat[:, self.cfg.c_d:]                      # (M, C_i)
            fused_dense[b, occ_idx] = fused_part

        # 5) Concat（图右侧 C） -> O
        # 输出 O: (B,S,C_d + C_i)  等价于图中的 O ∈ R^{X×Y×Z×Cf}
        O = torch.cat([O_D, fused_dense], dim=-1)                        # (B,S,C_d+C_i)

        # 你如果要变成 (B, Cf, X, Y, Z) 方便3D卷积：
        O_5d = O.transpose(1, 2).reshape(B, O.shape[-1], X, Y, Z)

        return {
            "O": O,                       # (B, S, Cf)
            "O_5d": O_5d,                 # (B, Cf, X, Y, Z)
            "O_D": O_D,                   # (B, S, C_d)
            "O_I": O_I,                   # (B, S, C_i)
            "occ_mask": occ,              # (B, S)
            "fusion_weights": all_weights # list of (M,K) per batch
        }

