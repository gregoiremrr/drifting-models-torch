"""Drifting field and drift loss (PyTorch port of the JAX `drift_loss.py`).

Implements the training objective of "Generative Modeling via Drifting"
(Deng et al., 2026). The drifting field V governs how generated samples move
at training time; the loss pushes each generated sample toward its (frozen)
drifted target ``stopgrad(x + V)``.

The field is the anti-symmetric, attraction/repulsion mean-shift field of
Eq. (10)-(11), estimated within a batch using the doubly-normalized softmax
kernel of Alg. 2 (softmax over both the x-axis and y-axis of the pairwise
distance matrix). Multiple temperatures are aggregated as in Appendix A.6.

All tensors operate per "row group" B: each group holds the generated,
negative and positive samples that interact with one another. In the
generator training loop, B = (#class_labels x #feature_locations), N is the
number of samples in a group, and D is the feature dimensionality.
"""

import torch

#----------------------------------------------------------------------------
# Batched pairwise Euclidean distance.
# x: [B, N, D], y: [B, M, D] -> [B, N, M]

def cdist(x, y, eps=1e-8):
    xydot = torch.einsum('bnd,bmd->bnm', x, y)
    xnorms = torch.einsum('bnd,bnd->bn', x, x)
    ynorms = torch.einsum('bmd,bmd->bm', y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(sq_dist.clamp(min=eps))

#----------------------------------------------------------------------------
# Drift loss for a single feature.
#
# Args:
#     gen:        [B, C_g, D] generated samples (with gradient).
#     fixed_pos:  [B, C_p, D] positive (real, same-class) samples.
#     fixed_neg:  [B, C_n, D] extra negatives (e.g. unconditional reals for CFG).
#                 May be None.
#     weight_gen/weight_pos/weight_neg: per-sample kernel weights [B, C_*].
#                 None => all ones.
#     R_list:     temperatures tau; one normalized drift per tau, then summed.
#
# Returns:
#     loss: [B] per-group MSE between the generated feature and its frozen
#           drifted target, computed in the normalized feature space.
#     info: dict of scalars (feature scale + per-temperature drift magnitude).

def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
):
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros((B, 0, S))
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    # Order matters: [generated | negatives | positives]; negatives are the
    # first (C_g + C_n) columns, positives the last C_p columns.
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # The entire target (goal) is computed without gradients.
    with torch.no_grad():
        info = {}
        dist = cdist(old_gen, targets)                       # [B, C_g, C_g+C_n+C_p]
        weighted_dist = dist * targets_w[:, None, :]
        # Feature normalization scale S_j: mean (weighted) pairwise distance.
        scale = weighted_dist.mean() / targets_w.mean()
        info['scale'] = scale

        # Normalize coordinates to order 1, and distances to order 1.
        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / scale.clamp(min=1e-3)

        # Mask self-interaction of generated samples against themselves
        # (the [gen, gen] diagonal block).
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = torch.nn.functional.pad(diag_mask, (0, C_n + C_p))[None]
        dist_normed = dist_normed + block_mask * mask_val

        force_across_R = torch.zeros_like(old_gen_scaled)
        for R in R_list:
            logits = -dist_normed / R
            # Doubly-normalized kernel A = sqrt(softmax_y * softmax_x).
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * aff_transpose).clamp(min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]            # generated + extra negatives
            aff_pos = affinity[:, :, split_idx:]            # positives

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos                # W_neg
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg                 # W_pos
            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum('biy,byx->bix', R_coeff, targets_scaled)
            # Subtract the self term so the field is invariant to translation;
            # the total coefficient is ~0 by anti-symmetry.
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f'loss_{R}'] = f_norm_val

            # Drift normalization per temperature, then sum.
            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))
    return loss, info

#----------------------------------------------------------------------------
