# SPDX-License-Identifier: Apache-2.0
# Formula adaptations from NVIDIA video_to_data (2026 NVIDIA CORPORATION & AFFILIATES).
# See NOTICE and configs/upstream.json for provenance.
"""Contact rewards with [hand, environment, body] weights before aggregation.

Original denominators are preserved: renormalizing by confidence would cancel
the intended attenuation. All-one weights reproduce the upstream formulas.
"""
import torch


def contact_terms(cmd_active, cur_active, cmd_support, cur_support, weights,
                  tolerance=.1, var=.1):
    if cmd_active.shape != cur_active.shape or cmd_active.shape != cmd_support.shape or cmd_active.shape != cur_support.shape:
        raise ValueError("contact tensors must share [hand, environment, body, direction] shape")
    if cmd_active.ndim != 4 or weights.shape != cmd_active.shape[:-1]:
        raise ValueError("weights must align to hand/environment/body axes")
    if var <= 0 or tolerance < 0:
        raise ValueError("invalid reward parameters")
    active_body = cmd_active.any(-1)
    current_body = cur_active.any(-1)
    basis_count = cmd_active.sum(-1).float().clamp(min=1e-6)
    body_count = active_body.sum(-1).float().clamp(min=1e-6)
    hand_count = (body_count > 1e-3).float().sum(0).clamp(min=1e-6)
    loss = ((1 - tolerance) * cmd_support - cur_support).clamp(min=0).square()
    loss = loss + (cur_support - (1 + tolerance) * cmd_support).clamp(min=0).square()
    per_body = ((cmd_active & cur_active).float() * torch.exp(-loss / var)).sum(-1) / basis_count
    wrench = ((per_body * weights).sum(-1) / body_count).sum(0) / hand_count
    missing = (cmd_active & ~cur_active).sum(-1).float() / basis_count
    missed = ((missing * active_body.float() * weights).sum(-1) / body_count).sum(0) / hand_count
    unintended = (~active_body & current_body).float()
    excess = (~active_body).float() * cur_support.clamp(min=0).square().mean(-1)
    inactive_count = (~active_body).sum(-1).float().clamp(min=1e-3)
    unwanted = ((unintended * weights).mean(-1) + (excess * weights).sum(-1) / inactive_count).sum(0)
    return {"contact_wrench_support_reward": wrench, "missed_contact_penalty": missed,
            "unintended_contact_penalty": unwanted}
