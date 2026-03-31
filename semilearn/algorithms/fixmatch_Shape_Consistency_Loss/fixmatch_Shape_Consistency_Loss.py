import torch
import torch.nn.functional as F

from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.fixmatch import FixMatch
from semilearn.algorithms.utils import SSL_Argument, str2bool


# -----------------------------
# small differentiable priors
# -----------------------------
def tv_loss_2d(m: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss for a 2D map.
    m: [B, 1, H, W] or [B, H, W]
    """
    if m.dim() == 3:
        m = m.unsqueeze(1)
    dh = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
    dw = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
    return dh + dw


def isotropy_loss(m: torch.Tensor) -> torch.Tensor:
    """
    Encourage activated region to be roughly isotropic.
    m: [B,1,H,W] or [B,H,W], non-negative
    """
    if m.dim() == 3:
        m = m.unsqueeze(1)
    B, _, H, W = m.shape
    eps = 1e-6

    m = m.clamp_min(0)
    Z = m.sum(dim=(2, 3), keepdim=True) + eps
    p = m / Z

    ys = torch.linspace(0, 1, H, device=m.device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=m.device).view(1, 1, 1, W)

    mx = (p * xs).sum(dim=(2, 3), keepdim=True)
    my = (p * ys).sum(dim=(2, 3), keepdim=True)

    vx = (p * (xs - mx) ** 2).sum(dim=(2, 3))
    vy = (p * (ys - my) ** 2).sum(dim=(2, 3))
    vxy = (p * (xs - mx) * (ys - my)).sum(dim=(2, 3))

    return ((vx - vy) ** 2 + 4.0 * (vxy ** 2)).mean()


# -----------------------------
# main algorithm
# -----------------------------
@ALGORITHMS.register('fixmatch_Shape_Consistency_Loss')
class FixMatch_Shape_Consistency_Loss(FixMatch):
    """
    FixMatch + Color/Area/Depth prior (EMA prototypes) on unlabeled samples.

    Prior is based on a differentiable "blue-ish" soft mask:
      - mask = sigmoid(k * (blue_score - t))
      - area = mean(mask)
      - depth = mean(mask * blue_score)

    Then:
      - maintain per-class EMA prototypes of (area, depth) from labeled data
      - for high-confidence unlabeled (mask==1), enforce its (area,depth) close to prototype of pseudo-label class

    Also optionally add very small TV/isotropy regularizers on the soft mask.
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

        # prior weights / hyperparams
        self.prior_lambda = args.prior_lambda
        self.proto_m = args.proto_m
        self.blue_k = args.blue_k
        self.blue_t = args.blue_t

        # optional tiny shape reg on mask
        self.mask_tv_lambda = args.mask_tv_lambda
        self.mask_iso_lambda = args.mask_iso_lambda

        # how to interpret input range
        self.assume_imagenet_norm = args.assume_imagenet_norm

        # per-class EMA prototypes: [K, 2] => (area, depth)
        K = self.args.num_classes
        self.model.register_buffer("proto_ad", torch.zeros(K, 2))
        self.model.register_buffer("proto_cnt", torch.zeros(K, dtype=torch.float))  # for warm start / debug

    # -------- utilities --------
    def _denorm_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        Return approx in [0,1] if it looks like ImageNet normalized.
        """
        if not self.assume_imagenet_norm:
            return x.clamp(0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x01 = x * std + mean
        return x01.clamp(0, 1)

    def _blue_soft_mask(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build a differentiable "blue-ish" score and soft mask.
        x01: [B,3,H,W] in [0,1]
        Returns:
          blue_score: [B,1,H,W] ~ higher => more blue
          mask: [B,1,H,W] in (0,1)
        """
        r = x01[:, 0:1]
        g = x01[:, 1:2]
        b = x01[:, 2:3]

        # simple differentiable "blueness" score
        # you can tweak: b - 0.5*(r+g) tends to highlight blue-ish region vs white paper
        blue_score = b - 0.5 * (r + g)  # roughly in [-1,1]

        # sigmoid to get soft mask
        mask = torch.sigmoid(self.blue_k * (blue_score - self.blue_t))
        return blue_score, mask

    @torch.no_grad()
    def _update_prototypes_from_labeled(self, x_lb: torch.Tensor, y_lb: torch.Tensor):
        proto_ad = self.model.proto_ad
        proto_cnt = self.model.proto_cnt

        x01 = self._denorm_if_needed(x_lb)
        blue_score, mask = self._blue_soft_mask(x01)

        area = mask.mean(dim=(2, 3))
        depth = (mask * blue_score).mean(dim=(2, 3))
        ad = torch.cat([area, depth], dim=1)

        for c in range(self.args.num_classes):
            idx = (y_lb == c)
            if idx.any():
                cur = ad[idx].mean(dim=0)
                proto_ad[c] = self.proto_m * proto_ad[c] + (1 - self.proto_m) * cur
                proto_cnt[c] += idx.float().sum()



    def _prior_loss_on_unlabeled(self, x_ulb_s: torch.Tensor, pseudo_label: torch.Tensor, mask_conf: torch.Tensor):
        """
        Compute prior loss only on high-confidence unlabeled samples.
        mask_conf: [B] float/bool (from FixMatch masking hook)
        pseudo_label: [B] (hard) or [B,K] (soft). We handle both.
        """
        if x_ulb_s is None:
            return torch.tensor(0.0, device=self.model.proto_ad.device)

        x01 = self._denorm_if_needed(x_ulb_s)
        blue_score, smask = self._blue_soft_mask(x01)

        # area/depth
        area = smask.mean(dim=(2, 3))                 # [B,1]
        depth = (smask * blue_score).mean(dim=(2, 3)) # [B,1]
        ad = torch.cat([area, depth], dim=1)          # [B,2]

        # choose class index
        if pseudo_label.dim() == 2:
            cls = pseudo_label.argmax(dim=1)
        else:
            cls = pseudo_label

        # select high-confidence
        m = mask_conf.float()
        if m.dim() == 2:
            m = m.squeeze(1)
        keep = (m > 0.5)
        if not keep.any():
            # still allow tiny mask regularization to shape smask smoothly (optional)
            reg = torch.tensor(0.0, device=x_ulb_s.device)
            if self.mask_tv_lambda > 0:
                reg = reg + self.mask_tv_lambda * tv_loss_2d(smask)
            if self.mask_iso_lambda > 0:
                reg = reg + self.mask_iso_lambda * isotropy_loss(smask)
            return reg

        ad_k = ad[keep]         # [Nk,2]
        cls_k = cls[keep]       # [Nk]
        proto = self.model.proto_ad[cls_k]  # [Nk,2]

        # main prototype matching loss
        # smooth L1 is robust for small data / noisy mask
        loss_proto = F.smooth_l1_loss(ad_k, proto, reduction='mean')

        # tiny regularizers on soft mask map (optional)
        loss_reg = torch.tensor(0.0, device=x_ulb_s.device)
        if self.mask_tv_lambda > 0:
            loss_reg = loss_reg + self.mask_tv_lambda * tv_loss_2d(smask)
        if self.mask_iso_lambda > 0:
            loss_reg = loss_reg + self.mask_iso_lambda * isotropy_loss(smask)

        return loss_proto + loss_reg

    # -------- training --------
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # update prototypes from labeled (EMA) - no grad
        self._update_prototypes_from_labeled(x_lb, y_lb)

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats = outputs.get('feat', None)
                feats_x_lb = feats[:num_lb] if feats is not None else None
                feats_x_ulb_w, feats_x_ulb_s = (feats[num_lb:].chunk(2) if feats is not None else (None, None))
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb.get('feat', None)

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s.get('feat', None)

                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w.get('feat', None)

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            # supervised
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # FixMatch pseudo label pipeline
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            pseudo_label = self.call_hook(
                "gen_ulb_targets", "PseudoLabelingHook",
                logits=probs_x_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T,
                softmax=False
            )

            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            # ---- new: color/area/depth prior loss ----
            prior_loss = self._prior_loss_on_unlabeled(x_ulb_s, pseudo_label, mask)
            # prior_loss = self._prior_loss_on_unlabeled(x_ulb_w, pseudo_label, mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.prior_lambda * prior_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            prior_loss=float(prior_loss.detach().cpu().item()),
            total_loss=total_loss.item(),
            util_ratio=mask.float().mean().item(),
            proto_cnt_min=float(self.model.proto_cnt.min().item()),
            proto_cnt_max=float(self.model.proto_cnt.max().item()),
            proto_cnt_sum=float(self.model.proto_cnt.sum().item()),
        )
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            # FixMatch
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),

            # Prior
            SSL_Argument('--prior_lambda', float, 0.2),   # 总先验权重（建议 0.05~0.3）
            SSL_Argument('--proto_m', float, 0.95),       # EMA 原型动量（0.9~0.99）
            SSL_Argument('--blue_k', float, 12.0),        # sigmoid 斜率（8~20）
            SSL_Argument('--blue_t', float, 0.02),        # 阈值（根据数据调：-0.05~0.1）
            SSL_Argument('--assume_imagenet_norm', str2bool, True),

            # tiny mask regularization (optional, keep small)
            SSL_Argument('--mask_tv_lambda', float, 0.02),
            SSL_Argument('--mask_iso_lambda', float, 0.01),
        ]
