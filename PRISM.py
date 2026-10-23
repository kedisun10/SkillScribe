from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False
    Mamba = None


DEFAULT_DIM_NAMES: Tuple[str, ...] = (
    "tissue", "time", "instrument", "flow", "knowledge"
)


class BiMambaBlock(nn.Module):


    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, bidirectional: bool = True):
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is not installed. Install with:\n"
                "  pip install causal-conv1d>=1.5.0\n"
                "  pip install mamba-ssm>=2.2.0"
            )
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        if bidirectional:
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state,
                                   d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.mamba_fwd(h)
        if self.bidirectional:
            h_rev = torch.flip(h, dims=[1])
            out_rev = self.mamba_bwd(h_rev)
            out = out + torch.flip(out_rev, dims=[1])
        return x + out


class DCABlock(nn.Module):


    def __init__(self, hidden: int, n_heads: int = 8, dropout: float = 0.1,
                 ffn_ratio: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True)
        self.vis_attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True)
        self.txt_attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln_q1 = nn.LayerNorm(hidden)
        self.ln_q2 = nn.LayerNorm(hidden)
        self.ln_q3 = nn.LayerNorm(hidden)
        self.ln_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ffn_ratio, hidden),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        vis_pad_mask: Optional[torch.Tensor] = None,
        txt_pad_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        vis_attn_bias=None,
        return_vis_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:


        q = self.ln_q1(queries)
        q2, _ = self.self_attn(
            q, q, q, attn_mask=self_attn_mask, need_weights=False,
        )
        queries = queries + q2

        merged_mask = None
        if vis_attn_bias is not None:
            B, P, S = vis_attn_bias.shape
            n_heads = self.vis_attn.num_heads
            merged_mask = (
                vis_attn_bias.unsqueeze(1)
                             .expand(B, n_heads, P, S)
                             .reshape(B * n_heads, P, S)
            )


        q = self.ln_q2(queries)
        v2, vis_w = self.vis_attn(
            q, vis_tokens, vis_tokens,
            key_padding_mask=vis_pad_mask,
            attn_mask=merged_mask,
            need_weights=return_vis_attn,
            average_attn_weights=True,
        )
        queries = queries + v2


        q = self.ln_q3(queries)
        t2, _ = self.txt_attn(
            q, txt_tokens, txt_tokens,
            key_padding_mask=txt_pad_mask,
            need_weights=False,
        )
        queries = queries + t2


        queries = queries + self.ffn(self.ln_ffn(queries))

        return queries, vis_w


class DimensionAwarePRISM(nn.Module):


    DIM_NAMES: Tuple[str, ...] = DEFAULT_DIM_NAMES

    def __init__(
        self,
        vis_dim: int,
        hidden: int,

        n_dim_groups: int = 5,
        queries_per_group: int = 6,
        n_free_queries: int = 15,

        n_blocks: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: int = 4,

        use_mamba: bool = True,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        bidirectional: bool = True,
        use_router: bool = True,
        router_heads: int = 4,
        router_temperature: float = 4.0,
        router_query_scale: bool = True,
        router_attn_bias: bool = True,
        router_bias_strength: float = 1.0,
        use_block_diag_mask: bool = True,

        dim_names: Tuple[str, ...] = DEFAULT_DIM_NAMES,
    ):
        super().__init__()

        assert n_dim_groups >= 1, "Need at least 2 dim groups for classification"
        assert queries_per_group >= 0
        assert n_free_queries >= 0
        assert len(dim_names) == n_dim_groups, (
            f"dim_names ({len(dim_names)}) must match n_dim_groups ({n_dim_groups})"
        )

        self.vis_dim = vis_dim
        self.hidden = hidden
        self.n_dim_groups = n_dim_groups
        self.queries_per_group = queries_per_group
        self.n_free_queries = n_free_queries
        self.DIM_NAMES = tuple(dim_names)


        self.use_mamba = bool(use_mamba) and mamba_layers > 0
        if self.use_mamba:
            self.temporal_encoder = nn.ModuleList([
                BiMambaBlock(
                    d_model=vis_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    bidirectional=bidirectional,
                ) for _ in range(mamba_layers)
            ])
            self.post_mamba_norm = nn.LayerNorm(vis_dim)
        else:
            self.temporal_encoder = None
            self.post_mamba_norm = None


        self.proj_v = nn.Linear(vis_dim, hidden, bias=False)


        scale = hidden ** -0.5
        self.dim_queries = nn.ParameterList([
            nn.Parameter(torch.randn(queries_per_group, hidden) * scale)
            for _ in range(n_dim_groups)
        ])
        if n_free_queries > 0:
            self.free_queries = nn.Parameter(
                torch.randn(n_free_queries, hidden) * scale
            )
        else:
            self.free_queries = None


        self.blocks = nn.ModuleList([
            DCABlock(hidden=hidden, n_heads=n_heads,
                     dropout=dropout, ffn_ratio=ffn_ratio)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden)


        self.aux_classifier = nn.Linear(hidden, n_dim_groups)

        self.use_router = bool(use_router)
        self.router_query_scale = bool(router_query_scale)
        self.router_attn_bias = bool(router_attn_bias)
        self.router_bias_strength = float(router_bias_strength)
        if self.use_router:
            self.router = OSATSRouter(
                hidden=hidden,
                n_groups=n_dim_groups,
                n_heads=router_heads,
                temperature=router_temperature,
            )
        else:
            self.router = None


        self.use_block_diag_mask = bool(use_block_diag_mask)
        if self.use_block_diag_mask:
            mask = self._build_block_diag_mask()
        else:
            mask = None
        self.register_buffer("_self_attn_mask", mask, persistent=False)


    @property
    def prefix_tokens(self) -> int:
        return self.n_dim_groups * self.queries_per_group + self.n_free_queries

    def _build_block_diag_mask(self) -> torch.Tensor:

        P = self.prefix_tokens
        mask = torch.full((P, P), float("-inf"))
        K = self.queries_per_group
        for g in range(self.n_dim_groups):
            mask[g * K:(g + 1) * K, g * K:(g + 1) * K] = 0.0
        if self.n_free_queries > 0:
            start = self.n_dim_groups * K
            mask[start:, start:] = 0.0
        return mask

    def _gather_queries(self, B: int, device, dtype) -> torch.Tensor:

        parts = [q.to(dtype) for q in self.dim_queries]
        if self.free_queries is not None:
            parts.append(self.free_queries.to(dtype))
        q = torch.cat(parts, dim=0)
        return q.unsqueeze(0).expand(B, -1, -1).contiguous()


    def forward(
        self,
        vis_seq: torch.Tensor,
        vis_mask: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, S, D = vis_seq.shape
        assert D == self.vis_dim, f"Expected vis_dim={self.vis_dim}, got {D}"
        assert vis_mask.shape == (B, S), f"vis_mask shape mismatch: {vis_mask.shape}"
        device = vis_seq.device
        dtype = vis_seq.dtype


        v = vis_seq
        if self.use_mamba:


            pad_zero = vis_mask.to(dtype).unsqueeze(-1)
            v = v * pad_zero
            for blk in self.temporal_encoder:
                v = blk(v)
            v = self.post_mamba_norm(v)
            v = v * pad_zero


        vis_h = self.proj_v(v)

        per_frame = None
        route_global = None
        if self.use_router:
            per_frame, route_global = self.router(vis_h, vis_mask)


        queries = self._gather_queries(B, device, vis_h.dtype)


        vis_pad_mask = ~vis_mask
        txt_pad_mask = (~txt_mask) if txt_mask is not None else None


        vis_attn_bias = None
        if self.use_router and self.router_attn_bias and per_frame is not None:
            G, K = self.n_dim_groups, self.queries_per_group

            group_idx = torch.arange(G * K, device=device) // K
            bias_dim = torch.log(
                per_frame.index_select(2, group_idx).clamp(min=1e-8)
            )
            bias_dim = bias_dim.transpose(1, 2) * self.router_bias_strength
            if self.n_free_queries > 0:
                bias_free = vis_h.new_zeros(B, self.n_free_queries, S)
                vis_attn_bias = torch.cat([bias_dim, bias_free], dim=1)
            else:
                vis_attn_bias = bias_dim

        last_vis_attn = None
        for i, blk in enumerate(self.blocks):
            want_attn = return_aux and (i == len(self.blocks) - 1)
            queries, vis_w = blk(
                queries, vis_h, txt_tokens,
                vis_pad_mask=vis_pad_mask,
                txt_pad_mask=txt_pad_mask,
                self_attn_mask=self._self_attn_mask,
                vis_attn_bias=vis_attn_bias,
                return_vis_attn=want_attn,
            )
            if want_attn:
                last_vis_attn = vis_w

        queries = self.final_norm(queries)

        if self.use_router and self.router_query_scale and route_global is not None:
            G, K = self.n_dim_groups, self.queries_per_group
            scale = (route_global * G).to(queries.dtype)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            dim_part = queries[:, :G * K, :].reshape(B, G, K, self.hidden)
            dim_part = dim_part * scale
            queries = torch.cat(
                [dim_part.reshape(B, G * K, self.hidden),
                 queries[:, G * K:, :]],
                dim=1,
            )

        prefix = queries

        if not return_aux:
            return prefix, None


        G, K = self.n_dim_groups, self.queries_per_group

        dim_prefix = prefix[:, :G * K, :].reshape(B, G, K, self.hidden)
        dim_features = dim_prefix.mean(dim=2)


        logits = self.aux_classifier(dim_features)


        attn_dim = last_vis_attn[:, :G * K, :].reshape(B, G, K, S)
        attn_maps = attn_dim.mean(dim=2)

        aux = {
            "logits": logits,
            "attn_maps": attn_maps,
            "dim_features": dim_features,
            "route_global": route_global,    
            "per_frame": per_frame,          
        }
        return prefix, aux


    def aux_classification_loss(self, logits: torch.Tensor) -> torch.Tensor:

        B, G, C = logits.shape
        assert G == C == self.n_dim_groups, \
            f"Expected [B, {self.n_dim_groups}, {self.n_dim_groups}], got {logits.shape}"
        labels = torch.arange(G, device=logits.device).unsqueeze(0).expand(B, -1)
        return F.cross_entropy(
            logits.reshape(B * G, C),
            labels.reshape(B * G),
        )

    @torch.no_grad()
    def aux_classification_accuracy(self, logits: torch.Tensor) -> Dict[str, float]:

        B, G, _ = logits.shape
        labels = torch.arange(G, device=logits.device).unsqueeze(0).expand(B, -1)
        pred = logits.argmax(dim=-1)
        correct = (pred == labels)
        per_group = correct.float().mean(dim=0).cpu().tolist()
        out = {"overall": correct.float().mean().item()}
        for name, acc in zip(self.DIM_NAMES, per_group):
            out[f"acc_{name}"] = acc
        return out

    def orth_penalty(self) -> torch.Tensor:


        centroids = torch.stack(
            [q.float().mean(dim=0) for q in self.dim_queries], dim=0,
        )
        c = F.normalize(centroids, dim=-1)
        sim = c @ c.T
        G = sim.size(0)
        eye = torch.eye(G, device=sim.device, dtype=sim.dtype)
        off_diag = (sim * (1.0 - eye)) ** 2
        return off_diag.sum() / (G * (G - 1))

    def attn_diversity_loss(
        self,
        attn_maps: torch.Tensor,
        vis_mask: torch.Tensor,
    ) -> torch.Tensor:

        B, G, S = attn_maps.shape
        mask = vis_mask.to(attn_maps.dtype).unsqueeze(1)
        a = attn_maps * mask
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        a_norm = F.normalize(a, dim=-1)
        sim = torch.bmm(a_norm, a_norm.transpose(1, 2))
        eye = torch.eye(G, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        off_diag = (sim * (1.0 - eye)).sum(dim=(1, 2)) / (G * (G - 1))
        return off_diag.mean()


class OSATSRouter(nn.Module):


    def __init__(self, hidden: int, n_groups: int, n_heads: int = 4,
                 temperature: float = 4.0):
        super().__init__()
        self.n_groups = n_groups
        self.temperature = temperature
        self.dim_probes = nn.Parameter(
            torch.randn(n_groups, hidden) * (hidden ** -0.5)
        )
        self.norm = nn.LayerNorm(hidden)
        self.scorer = nn.MultiheadAttention(
            hidden, n_heads, batch_first=True,
        )

    def forward(
        self,
        vis_h: torch.Tensor,
        vis_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, H = vis_h.shape
        probes = self.dim_probes.to(vis_h.dtype).unsqueeze(0).expand(B, -1, -1)

        _, attn = self.scorer(
            probes, self.norm(vis_h), vis_h,
            key_padding_mask=~vis_mask,
            need_weights=True,
            average_attn_weights=True,
        )


        per_frame = attn.transpose(1, 2)
        per_frame = F.softmax(per_frame * self.temperature, dim=-1)


        mask_f = vis_mask.to(per_frame.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        route_global = (per_frame * mask_f).sum(dim=1) / denom
        return per_frame, route_global

if __name__ == "__main__":
    torch.manual_seed(0)


    B, S, D, H = 3, 25, 768, 3072


    cuda_ok = torch.cuda.is_available()
    use_mamba = _HAS_MAMBA and cuda_ok
    if _HAS_MAMBA and not cuda_ok:
        print("[smoke test] mamba_ssm is installed but no CUDA device is "
              "visible — running smoke test with BiMamba disabled.\n")
    device = torch.device("cuda" if cuda_ok else "cpu")

    prism = DimensionAwarePRISM(
        vis_dim=D,
        hidden=H,
        n_dim_groups=5,
        queries_per_group=6,
        n_free_queries=15,
        n_blocks=2,
        n_heads=8,
        dropout=0.1,
        ffn_ratio=4,
        use_mamba=use_mamba,
        mamba_layers=2 if use_mamba else 0,
    ).to(device)

    n_params = sum(p.numel() for p in prism.parameters() if p.requires_grad)
    print(f"DimensionAwarePRISM trainable parameters: {n_params / 1e6:.2f}M")
    print(f"Prefix length: {prism.prefix_tokens} tokens")
    print(f"BiMamba enabled: {use_mamba}")
    print(f"Device: {device}")
    print()


    vis = torch.randn(B, S, D, device=device)
    vis_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    vis_mask[1, 18:] = False
    vis_mask[2, 22:] = False
    txt = torch.randn(B, 50, H, device=device)

    prefix, aux = prism(vis, vis_mask, txt, return_aux=True)
    print(f"prefix:        {tuple(prefix.shape)}")
    print(f"aux.logits:    {tuple(aux['logits'].shape)}")
    print(f"aux.attn_maps: {tuple(aux['attn_maps'].shape)}")
    print(f"aux.dim_features: {tuple(aux['dim_features'].shape)}")
    print()

    ce = prism.aux_classification_loss(aux["logits"])
    orth = prism.orth_penalty()
    div = prism.attn_diversity_loss(aux["attn_maps"], vis_mask)
    acc = prism.aux_classification_accuracy(aux["logits"])

    print(f"aux CE loss:        {ce.item():.4f}  (random init ~= log({prism.n_dim_groups}) = {torch.log(torch.tensor(float(prism.n_dim_groups))).item():.4f})")
    print(f"orth penalty:       {orth.item():.4f}")
    print(f"attn diversity:     {div.item():.4f}  (low = groups attend differently)")
    print(f"aux accuracy:       {acc}")
    print(f"route_global: {tuple(aux['route_global'].shape)}")   
    print(f"per_frame:    {tuple(aux['per_frame'].shape)}")      
    print(f"route_global[0]: {aux['route_global'][0].tolist()}") 
    print()


    total = ce + 0.05 * orth + 0.05 * div
    total.backward()
    n_grad = sum(1 for p in prism.parameters() if p.requires_grad and p.grad is not None)
    n_trainable = sum(1 for p in prism.parameters() if p.requires_grad)
    print(f"Parameters receiving gradient: {n_grad}/{n_trainable}")
    print("✓ Smoke test passed.")
