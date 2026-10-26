import itertools
import torch
import torch.nn as nn


class SharedGaussianProjector(nn.Module):
    def __init__(self, in_dim=512, latent_dim=512, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU()
        )

        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        h = self.mlp(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class ProductOfExperts(nn.Module):
    def __init__(self, eps=1e-8, add_prior=False):
        super().__init__()
        self.eps = eps
        self.add_prior = add_prior

    def forward(self, mus, logvars):
        vars_ = [torch.exp(logvar) + self.eps for logvar in logvars]
        precisions = [1.0 / var for var in vars_]

        fused_precision = torch.stack(precisions, dim=0).sum(dim=0)
        fused_mu_precision = torch.stack(
            [mu * precision for mu, precision in zip(mus, precisions)], dim=0
        ).sum(dim=0)

        if self.add_prior:
            fused_precision = fused_precision + 1.0

        fused_precision = fused_precision + self.eps
        fused_var = 1.0 / fused_precision
        fused_mu = fused_var * fused_mu_precision
        fused_logvar = torch.log(fused_var + self.eps)

        return fused_mu, fused_logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class MultiModalPoEFusion(nn.Module):
    def __init__(self, in_dim=512, latent_dim=512, hidden_dim=None, dropout=0.1, add_prior=False):
        super().__init__()
        self.projector = SharedGaussianProjector(
            in_dim=in_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.poe = ProductOfExperts(add_prior=add_prior)

    def _encode_single(self, feat_list):
        mus, logvars = [], []
        for feat in feat_list:
            mu, logvar = self.projector(feat)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def _fuse_subset(self, mus, logvars, subset, sample=True):
        sub_mus = [mus[i] for i in subset]
        sub_logvars = [logvars[i] for i in subset]

        fused_mu, fused_logvar = self.poe(sub_mus, sub_logvars)
        z = reparameterize(fused_mu, fused_logvar) if sample else fused_mu

        return {
            "z": z,
            "fused_mu": fused_mu,
            "fused_logvar": fused_logvar,
            "subset": subset
        }

    def forward(self, feat_list, sample=True, mode="all", custom_sizes=(1, 2, 3, 4)):
        """
        feat_list: 当前输入模态列表，每个元素 [B, D]
                   例如:
                   推理时: [nir_feat, cp_feat]
                   或     [nir_feat, cp_feat, sk_feat]
                   训练时: [nir_feat, cp_feat, sk_feat, text_feat]

        mode:
            "all"    -> 当前输入的所有模态一起融合，返回一个结果
            "custom" -> 返回 1/2/3/4 模态的各种组合结果
        """
        num_modalities = len(feat_list)
        assert num_modalities >= 1

        mus, logvars = self._encode_single(feat_list)

        if mode == "all":
            # 当前输入的所有模态一起融合
            subset = tuple(range(num_modalities))
            return self._fuse_subset(mus, logvars, subset, sample=False)

        elif mode == "custom":
            fused_feats_for_one = []
            fused_feats_for_two = []
            fused_feats_for_three = []
            fused_feats_for_four = []

            fused_dists_for_one = []
            fused_dists_for_two = []
            fused_dists_for_three = []
            fused_dists_for_four = []

            for r in custom_sizes:
                if r > num_modalities:
                    continue

                for subset in itertools.combinations(range(num_modalities), r):
                    result = self._fuse_subset(mus, logvars, subset, sample=sample)

                    if r == 1:
                        fused_feats_for_one.append(result["z"])
                        fused_dists_for_one.append(result)
                    elif r == 2:
                        fused_feats_for_two.append(result["z"])
                        fused_dists_for_two.append(result)
                    elif r == 3:
                        fused_feats_for_three.append(result["z"])
                        fused_dists_for_three.append(result)
                    elif r == 4:
                        fused_feats_for_four.append(result["z"])
                        fused_dists_for_four.append(result)

            return {
                "fused_feats_for_one": fused_feats_for_one,
                "fused_feats_for_two": fused_feats_for_two,
                "fused_feats_for_three": fused_feats_for_three,
                "fused_feats_for_four": fused_feats_for_four,

                "fused_dists_for_one": fused_dists_for_one,
                "fused_dists_for_two": fused_dists_for_two,
                "fused_dists_for_three": fused_dists_for_three,
                "fused_dists_for_four": fused_dists_for_four,
            }

        else:
            raise ValueError(f"Unsupported mode: {mode}")