"""Drift loss orchestration (PyTorch port of `train.py:train_step.loss_fn`).

Given a batch of class labels with cached positive (same-class) and
unconditional real samples, this computes the multi-feature drifting loss:

  1. Extract features (a raw "global" feature plus multi-scale MAE features)
     for positive + unconditional samples (no gradient).
  2. Generate `gen_per_label` samples per label and extract their features
     (with gradient through the generator and the feature encoder's input).
  3. For each feature (per scale / per spatial location) compute the drift
     loss of Eq. (26), summing across features.

Classifier-free guidance (Appendix A.7) is realized at training time only:
each label samples a CFG scale `alpha` from a power-law distribution, the
unconditional reals act as extra negatives weighted by `w`, and `alpha` is fed
to the generator as a conditioning input. Inference stays 1-NFE.
"""

import torch
from torch_utils import persistence
from training.drift_field import drift_loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class DriftLoss:
    def __init__(
        self,
        gen_per_label=64,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
        R_list=(0.02, 0.05, 0.2),
        use_norm_x=False,
        cfg_preserve_old=False,
        activation_kwargs=None,
    ):
        self.gen_per_label = gen_per_label
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max
        self.neg_cfg_pw = neg_cfg_pw
        self.no_cfg_frac = no_cfg_frac
        self.cfg_preserve_old = cfg_preserve_old
        self.R_list = tuple(R_list)
        self.use_norm_x = use_norm_x
        if activation_kwargs is None:
            activation_kwargs = dict(
                patch_mean_size=[2, 4], patch_std_size=[2, 4],
                use_std=True, use_mean=True, every_k_block=2,
            )
        self.activation_kwargs = dict(activation_kwargs)

    # -- CFG-scale sampling (power-law p(alpha) ~ alpha^-neg_cfg_pw) ----------

    def sample_cfg(self, n, device, generator=None):
        frac = torch.rand(n, device=device, generator=generator)
        pw = 1.0 - self.neg_cfg_pw
        cmin, cmax = self.cfg_min, self.cfg_max
        if abs(pw) < 1e-6:
            import math
            cfg = torch.exp(math.log(cmin) + frac * (math.log(cmax) - math.log(cmin)))
        else:
            cfg = (cmin ** pw + frac * (cmax ** pw - cmin ** pw)) ** (1.0 / pw)
        if self.no_cfg_frac > 0:
            frac2 = torch.rand(n, device=device, generator=generator)
            cfg = torch.where(frac2 < self.no_cfg_frac, torch.ones_like(cfg), cfg)
        return cfg

    # -- feature extraction --------------------------------------------------

    def _features(self, feature_encoder, x):
        out = {'global': x.reshape(x.shape[0], 1, -1)}
        if self.use_norm_x:
            out['norm_x'] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]
        if feature_encoder is not None:
            out.update(feature_encoder.get_activations(x, **self.activation_kwargs))
        return out

    @staticmethod
    def _group_by_token(feat):
        # [B, N, T, D] -> [(B*T), N, D]
        B, N, T, D = feat.shape
        return feat.permute(0, 2, 1, 3).reshape(B * T, N, D)

    # -- main call -----------------------------------------------------------

    def __call__(self, model, feature_encoder, labels, pos_images, uncond_images, cfg, old_neg_images=None):
        """
        Args:
            model: generator (possibly DDP-wrapped); ``model(class_idx, cfg)`` -> images.
            feature_encoder: frozen MAE (or None for raw-pixel-only drift).
            labels: [Nc] long class indices.
            pos_images: [Nc, n_pos, C, H, W] positive reals.
            uncond_images: [Nc, n_uncond, C, H, W] unconditional reals (CFG negatives).
            cfg: [Nc] per-label CFG scale.
            old_neg_images: optional [Nc, n_old, C, H, W] cached past-generated samples,
                used as extra (frozen, weight-1) negatives representing q(.|c).
        """
        device = pos_images.device
        Nc, n_pos = pos_images.shape[0], pos_images.shape[1]
        n_uncond = uncond_images.shape[1]
        n_old = old_neg_images.shape[1] if old_neg_images is not None else 0
        g = self.gen_per_label

        # CFG weight balances the unconditional reals (p(emptyset)) against the
        # generated negatives (q(.|c)). With cfg_preserve_old, the cached old-gen
        # negatives are counted into the q-side so the p(emptyset):q ratio (hence
        # guidance strength) stays fixed regardless of n_old.
        q_count = (g - 1 + n_old) if self.cfg_preserve_old else (g - 1)
        uncond_w = (cfg - 1.0) * q_count / max(1, n_uncond)  # [Nc]

        # Frozen features for positives + unconditional negatives (+ cached old-gen negatives).
        parts = [pos_images, uncond_images]
        if n_old > 0:
            parts.append(old_neg_images)
        n_neg_block = n_pos + n_uncond + n_old
        neg_input = torch.cat(parts, dim=1)
        neg_input = neg_input.reshape(Nc * n_neg_block, *neg_input.shape[2:])
        with torch.no_grad():
            sg_feats = self._features(feature_encoder, neg_input)
        sg_feats = {k: v.reshape(Nc, n_neg_block, *v.shape[1:]) for k, v in sg_feats.items()}

        # Generated samples (with gradient).
        class_idx = labels.repeat_interleave(g)
        cfg_rep = cfg.repeat_interleave(g)
        gen_images = model(class_idx, cfg_rep)
        gen_feats = self._features(feature_encoder, gen_images)
        gen_feats = {k: v.reshape(Nc, g, *v.shape[1:]) for k, v in gen_feats.items()}

        total_loss = gen_images.new_zeros(())
        stats = {}
        for key, gen_f in gen_feats.items():
            sg = sg_feats[key]                          # [Nc, n_neg_block, T, D]
            pos_f = sg[:, :n_pos]
            uncond_f = sg[:, n_pos:n_pos + n_uncond]
            old_f = sg[:, n_pos + n_uncond:]
            T = gen_f.shape[2]

            gen_r = self._group_by_token(gen_f)         # [(Nc*T), g, D]
            pos_r = self._group_by_token(pos_f)

            # Combined fixed negatives: unconditional reals (CFG-weighted) + old-gen (weight 1).
            neg_feats, neg_weights = [], []
            if n_uncond > 0:
                neg_feats.append(self._group_by_token(uncond_f))
                neg_weights.append(uncond_w[:, None, None].expand(Nc, T, n_uncond).reshape(Nc * T, n_uncond))
            if n_old > 0:
                neg_feats.append(self._group_by_token(old_f))
                neg_weights.append(gen_r.new_ones(Nc * T, n_old))
            fixed_neg = torch.cat(neg_feats, dim=1) if neg_feats else None
            w_neg = torch.cat(neg_weights, dim=1) if neg_weights else None

            loss_k, info_k = drift_loss(
                gen=gen_r, fixed_pos=pos_r, fixed_neg=fixed_neg,
                weight_neg=w_neg, R_list=self.R_list,
            )
            total_loss = total_loss + loss_k.mean()
            stats[f'drift/{key}'] = loss_k.mean().detach()

        out_stats = dict(loss=total_loss.detach(), cfg=cfg.mean().detach(),
                         gen_images=gen_images.detach(), gen_labels=class_idx.detach())
        return total_loss, out_stats

#----------------------------------------------------------------------------
