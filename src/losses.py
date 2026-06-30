from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    valid = (targets >= 0) & (targets < num_classes)
    targets_safe = targets.clone()
    targets_safe[~valid] = 0
    targets_one_hot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1).float()
    intersection = (probs * targets_one_hot * valid_mask).sum(dim=(0, 2, 3))
    union = (probs * valid_mask).sum(dim=(0, 2, 3)) + (targets_one_hot * valid_mask).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return 1.0 - dice.mean()


class SegmentationCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ce_weight: float,
        dice_weight: float,
        feat_weight: float,
        pred_weight: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.feat_weight = feat_weight
        self.pred_weight = pred_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(
        self,
        student_out: dict,
        teacher_out: dict | None,
        targets: torch.Tensor,
        distill_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        logits = student_out["logits"]
        loss_ce = self.ce(logits, targets)
        loss_dice = dice_loss(logits, targets, self.num_classes)

        loss_feat = torch.tensor(0.0, device=logits.device)
        loss_pred = torch.tensor(0.0, device=logits.device)

        if teacher_out is not None and distill_weight > 0.0:
            for s_feat, t_feat in zip(student_out["fused_features"], teacher_out["fused_features"]):
                loss_feat = loss_feat + F.mse_loss(s_feat, t_feat.detach())

            t_prob = F.softmax(teacher_out["logits"].detach(), dim=1)
            s_log_prob = F.log_softmax(student_out["logits"], dim=1)
            valid = ((targets >= 0) & (targets < self.num_classes)).float()
            pixel_kl = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=1)
            loss_pred = (pixel_kl * valid).sum() / valid.sum().clamp_min(1.0)

        total = (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + distill_weight * self.feat_weight * loss_feat
            + distill_weight * self.pred_weight * loss_pred
        )
        return {
            "total": total,
            "ce": loss_ce,
            "dice": loss_dice,
            "feat": loss_feat,
            "pred": loss_pred,
        }
