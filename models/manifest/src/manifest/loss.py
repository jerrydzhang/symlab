import torch
import torch.nn.functional as F
from .tokenizer import PAD_ID, NUM_ID


def compute_loss(
    logits: torch.Tensor,
    num_preds: torch.Tensor,
    token_targets: torch.Tensor,
    num_targets: torch.Tensor,
    lambda_: float,
    ignore_index: int = PAD_ID,
    num_index: int = NUM_ID,
) -> torch.Tensor:
    ce, mse = decomposed_loss(
        logits, num_preds, token_targets, num_targets, ignore_index, num_index
    )
    return ce + lambda_ * mse


def decomposed_loss(
    logits: torch.Tensor,
    num_preds: torch.Tensor,
    token_targets: torch.Tensor,
    num_targets: torch.Tensor,
    ignore_index: int = PAD_ID,
    num_index: int = NUM_ID,
    max_const: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ce_loss, mse_loss) separately for logging.

    Constant targets are clipped to [-max_const, max_const] before MSE to
    prevent extreme values from destabilizing training. The stats token
    carries normalization params for recovering real constants at inference.
    """
    logits = logits.flatten(0, 1)
    token_targets = token_targets.flatten(0, 1)
    ce_loss = F.cross_entropy(logits, token_targets, ignore_index=ignore_index)

    num_preds = num_preds.squeeze(-1).flatten(0, 1)
    num_targets = num_targets.flatten(0, 1)

    num_mask = token_targets == num_index
    num_preds = num_preds[num_mask]
    num_targets = num_targets[num_mask].clamp(-max_const, max_const)
    if num_preds.numel() == 0:
        mse_loss = torch.tensor(0.0, device=logits.device)
    else:
        mse_loss = F.mse_loss(num_preds, num_targets)

    return ce_loss, mse_loss
