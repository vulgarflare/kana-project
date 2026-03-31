# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from semilearn.algorithms.fixmatch.fixmatch import FixMatch
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('fixmatch_Shape_Consistency_Loss_Overlay2Img')
class Fixmatch_Shape_Consistency_Loss_Overlay2Img(FixMatch):
    """
    FixMatch + Shape Consistency Prior Loss
    - prior loss 作用于 strong augmentation + overlay mask 图像（x_ulb_s_overlay）
    - 兼容 semilearn 0.3.2：model(x) 可能返回 dict/tuple，而不是纯 logits Tensor
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        
        # 兼容：不同版本 FixMatch 里无标签 loss 权重名字不一致
        self.ulb_loss_ratio = getattr(args, "ulb_loss_ratio", 1.0)
        if hasattr(self, "lambda_u") and (self.lambda_u is not None):
            # 有些版本叫 lambda_u
            self.ulb_loss_ratio = self.lambda_u

        # prior 参数
        self.prior_lambda = getattr(args, "prior_lambda", 0.05)
        self.blue_k = getattr(args, "blue_k", 8.0)
        self.blue_t = getattr(args, "blue_t", -0.04)

        # ImageNet normalization mean/std
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--prior_lambda', float, 0.05),
            SSL_Argument('--blue_k', float, 8.0),
            SSL_Argument('--blue_t', float, -0.04),
        ]

    # -----------------------------
    # Helpers: forward logits safely
    # -----------------------------
    def _as_logits(self, out):
        """semilearn 有些 net wrapper 会返回 dict/tuple；这里统一取 logits Tensor"""
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            # 常见: (logits, feats) 或 (logits,)
            for v in out:
                if torch.is_tensor(v):
                    return v
            raise TypeError(f"Model output tuple/list has no Tensor element: {type(out)}")
        if isinstance(out, dict):
            for k in ("logits", "logit", "pred", "output"):
                if k in out and torch.is_tensor(out[k]):
                    return out[k]
            # 如果 dict 里只有一个 tensor value
            for v in out.values():
                if torch.is_tensor(v):
                    return v
            raise TypeError(f"Model output dict has no Tensor logits: keys={list(out.keys())}")
        raise TypeError(f"Unsupported model output type: {type(out)}")

    def forward_logits(self, x):
        return self._as_logits(self.model(x))

    # -----------------------------
    # Denorm / mask / overlay
    # -----------------------------
    def denorm(self, x):
        mean = self.imagenet_mean.to(x.device)
        std = self.imagenet_std.to(x.device)
        return (x * std + mean).clamp(0, 1)

    def blue_score(self, x01):
        r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
        return b - 0.5 * (r + g)

    def build_soft_mask(self, x_norm):
        """
        x_norm: normalized tensor [B,3,H,W]
        return: mask ∈ [0,1], [B,1,H,W]
        """
        x01 = self.denorm(x_norm)
        score = self.blue_score(x01)
        return torch.sigmoid(self.blue_k * (score - self.blue_t))

    def overlay_mask(self, x_norm, mask):
        """
        在[0,1]空间把 mask 区域红色高亮，再还原回 normalized 空间
        x_norm: [B,3,H,W] normalized
        mask:   [B,1,H,W] in [0,1]
        """
        x01 = self.denorm(x_norm)

        overlay01 = x01.clone()
        # 红通道增强
        overlay01[:, 0:1] = (overlay01[:, 0:1] * (1 - 0.6 * mask) + 0.6 * mask).clamp(0, 1)

        mean = self.imagenet_mean.to(x_norm.device)
        std = self.imagenet_std.to(x_norm.device)
        overlay_norm = (overlay01 - mean) / std
        return overlay_norm

    # -----------------------------
    # Prior loss
    # -----------------------------
    def prior_loss(self, logits, mask):
        """
        logits: [B,C]
        mask:   [B,1,H,W]
        用 entropy minimization + mask weighting（按每张图 mask 平均强度加权）
        """
        probs = torch.softmax(logits, dim=-1)              # [B,C]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B]
        mask_w = mask.mean(dim=[2, 3]).squeeze(1)          # [B]
        return (entropy * mask_w).mean()

    # -----------------------------
    # Train step (semilearn 0.3.2)
    # -----------------------------
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        """
        注意：Trainer.fit 会调用：
        algorithm.train_step(**algorithm.process_batch(**data_lb, **data_ulb))
        所以这里的入参应与 FixMatch 版本一致：x_lb/y_lb/x_ulb_w/x_ulb_s
        """

        # (1) supervised
        logits_lb = self.forward_logits(x_lb)
        sup_loss = F.cross_entropy(logits_lb, y_lb)

        # (2) pseudo label from weak
        with torch.no_grad():
            logits_ulb_w = self.forward_logits(x_ulb_w)
            probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
            max_probs, pseudo_label = torch.max(probs_ulb_w, dim=-1)
            mask_conf = max_probs.ge(self.p_cutoff).float()

        # (3) unsup loss on strong
        logits_ulb_s = self.forward_logits(x_ulb_s)
        unsup_loss = (F.cross_entropy(logits_ulb_s, pseudo_label, reduction="none") * mask_conf).mean()

        # (4) prior loss on overlay(strong)
        soft_mask = self.build_soft_mask(x_ulb_s)              # [B,1,H,W]
        x_ulb_s_overlay = self.overlay_mask(x_ulb_s, soft_mask)
        logits_overlay = self.forward_logits(x_ulb_s_overlay)
        prior = self.prior_loss(logits_overlay, soft_mask)

        total_loss = sup_loss + self.ulb_loss_ratio * unsup_loss + self.prior_lambda * prior

        # 返回 dict：loss 必须是 Tensor
        return {
            "loss": total_loss,
            "sup_loss": sup_loss.detach(),
            "unsup_loss": unsup_loss.detach(),
            "prior_loss": prior.detach(),
            "mask_conf_ratio": mask_conf.mean().detach(),
            "mask_mean": soft_mask.mean().detach(),
        }
