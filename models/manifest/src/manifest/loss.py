import torch
import torch.nn.functional as F
from .tokenizer import PAD_ID, NUM_ID


def transform_constant(c: torch.Tensor) -> torch.Tensor:
    """Sign-preserving log transform: sign(c) * log1p(|c|).

    Compresses [-100, 100] to [-4.6, 4.6] so MSE behaves well across
    the full constant range. Invert with inverse_transform_constant.
    """
    return torch.sign(c) * torch.log1p(c.abs())


def inverse_transform_constant(t: torch.Tensor) -> torch.Tensor:
    """Inverse of transform_constant: sign(t) * expm1(|t|)."""
    return torch.sign(t) * torch.expm1(t.abs())


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ce_loss, mse_loss) separately for logging.

    Constant targets are log-transformed (sign-preserving log1p) before
    MSE so the loss operates in a bounded range regardless of the real
    constant magnitude. Predictions are inverse-transformed at generation
    time via inverse_transform_constant.
    """
    logits = logits.flatten(0, 1)
    token_targets = token_targets.flatten(0, 1)
    ce_loss = F.cross_entropy(logits, token_targets, ignore_index=ignore_index)

    num_preds = num_preds.squeeze(-1).flatten(0, 1)
    num_targets = num_targets.flatten(0, 1)

    num_mask = token_targets == num_index
    num_preds = num_preds[num_mask]
    num_targets = transform_constant(num_targets[num_mask])
    if num_preds.numel() == 0:
        mse_loss = torch.tensor(0.0, device=logits.device)
    else:
        mse_loss = F.mse_loss(num_preds, num_targets)

    return ce_loss, mse_loss
