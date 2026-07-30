import torch
import torch.nn.functional as F

from manifest.loss import compute_loss
from manifest.tokenizer import NUM_ID, PAD_ID

V = 8  # vocab size for synthetic logits


class TestCrossEntropy:
    def test_ce_ignores_pad_positions(self):
        # Perturbing logits only at PAD positions must not move the loss: PAD
        # rows are dropped by cross_entropy(ignore_index=PAD_ID).
        torch.manual_seed(0)
        B, L = 2, 5
        logits = torch.randn(B, L, V, requires_grad=True)
        # real classes avoid PAD (0) and NUM (3) so MSE stays empty
        targets = torch.tensor([[1, PAD_ID, 4, 5, PAD_ID], [6, 7, PAD_ID, 2, 1]])
        num_preds = torch.randn(B, L, 1, requires_grad=True)
        num_targets = torch.zeros(B, L)
        loss_ref = compute_loss(logits, num_preds, targets, num_targets, lambda_=1.0)

        logits2 = logits.detach().clone()
        with torch.no_grad():
            pad_rows = (targets == PAD_ID).sum().item()
            logits2[targets == PAD_ID] = torch.randn(pad_rows, V) * 100.0
        logits2.requires_grad_(True)
        loss_pert = compute_loss(logits2, num_preds, targets, num_targets, lambda_=1.0)
        assert torch.allclose(loss_ref, loss_pert, atol=1e-5)


class TestMSE:
    def test_mse_ignores_non_num_positions(self):
        # Perturbing num_preds only at non-NUM positions must not move the loss.
        torch.manual_seed(0)
        B, L = 2, 5
        logits = torch.zeros(B, L, V, requires_grad=True)  # constant CE term
        targets = torch.tensor([[1, NUM_ID, 4, PAD_ID, 6], [2, 5, NUM_ID, 7, 1]])
        num_targets = torch.randn(B, L)
        base = torch.randn(B, L, 1)
        num_preds = base.clone().detach().requires_grad_(True)
        loss_ref = compute_loss(logits, num_preds, targets, num_targets, lambda_=1.0)

        perturbed = base.clone()
        non_num = ~(targets == NUM_ID)
        perturbed.squeeze(-1)[non_num] = torch.randn(non_num.sum()) * 50.0
        num_preds2 = perturbed.detach().requires_grad_(True)
        loss_pert = compute_loss(logits, num_preds2, targets, num_targets, lambda_=1.0)
        assert torch.allclose(loss_ref, loss_pert, atol=1e-5)

    def test_mse_value_matches_manual_computation(self):
        torch.manual_seed(0)
        B, L = 2, 5
        logits = torch.zeros(B, L, V, requires_grad=True)
        targets = torch.tensor([[1, NUM_ID, 4, PAD_ID, 6], [2, 5, NUM_ID, 7, 1]])
        num_preds = torch.randn(B, L, 1, requires_grad=True)
        num_targets = torch.randn(B, L)
        loss_l0 = compute_loss(logits, num_preds, targets, num_targets, lambda_=0.0)
        loss_l1 = compute_loss(logits, num_preds, targets, num_targets, lambda_=1.0)
        mask = targets == NUM_ID
        from manifest.loss import transform_constant
        transformed_targets = transform_constant(num_targets[mask])
        manual_mse = F.mse_loss(num_preds.squeeze(-1)[mask], transformed_targets)
        assert torch.allclose(loss_l1 - loss_l0, manual_mse, atol=1e-6)


class TestLambdaWeighting:
    def test_loss_scales_linearly_with_lambda(self):
        torch.manual_seed(0)
        B, L = 2, 5
        logits = torch.randn(B, L, V, requires_grad=True)
        targets = torch.tensor([[1, NUM_ID, 4, PAD_ID, 6], [2, 5, NUM_ID, 7, 1]])
        num_preds = torch.randn(B, L, 1, requires_grad=True)
        num_targets = torch.randn(B, L)
        common = dict(
            logits=logits,
            num_preds=num_preds,
            token_targets=targets,
            num_targets=num_targets,
        )
        loss_l0 = compute_loss(**common, lambda_=0.0)
        loss_l1 = compute_loss(**common, lambda_=1.0)
        loss_l2 = compute_loss(**common, lambda_=2.5)
        mask = targets == NUM_ID
        from manifest.loss import transform_constant
        transformed = transform_constant(num_targets[mask])
        mse = F.mse_loss(num_preds.squeeze(-1)[mask], transformed)
        assert torch.allclose(loss_l1 - loss_l0, mse, atol=1e-6)
        assert torch.allclose(loss_l2 - loss_l0, 2.5 * mse, atol=1e-6)


class TestEmptyNumGuard:
    def _no_num(self):
        torch.manual_seed(0)
        B, L = 2, 5
        logits = torch.randn(B, L, V, requires_grad=True)
        targets = torch.tensor([[1, 2, 4, PAD_ID, 6], [5, 6, PAD_ID, 7, 2]])  # no NUM
        num_preds = torch.randn(B, L, 1, requires_grad=True)
        num_targets = torch.randn(B, L)
        return logits, num_preds, targets, num_targets

    def test_returns_ce_only_when_no_num_positions(self):
        logits, num_preds, targets, num_targets = self._no_num()
        loss = compute_loss(logits, num_preds, targets, num_targets, lambda_=1.0)
        ce_only = compute_loss(logits, num_preds, targets, num_targets, lambda_=0.0)
        assert torch.allclose(loss, ce_only, atol=1e-6)

    def test_backward_does_not_crash_without_num(self):
        logits, num_preds, targets, num_targets = self._no_num()
        loss = compute_loss(logits, num_preds, targets, num_targets, lambda_=1.0)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0
        # With no NUM positions the numeric head is disconnected from the graph:
        # the empty-NUM guard yields a grad-free constant 0.0 for the MSE term.
        assert num_preds.grad is None


class TestGradientFlow:
    def test_gradients_reach_logits_and_num_preds(self):
        torch.manual_seed(0)
        B, L = 2, 5
        logits = torch.randn(B, L, V, requires_grad=True)
        targets = torch.tensor([[1, NUM_ID, 4, PAD_ID, 6], [2, 5, NUM_ID, 7, 1]])
        num_preds = torch.randn(B, L, 1, requires_grad=True)
        num_targets = torch.randn(B, L)
        loss = compute_loss(logits, num_preds, targets, num_targets, lambda_=1.0)
        loss.backward()
        assert logits.grad.abs().sum() > 0
        assert num_preds.grad.abs().sum() > 0
        # num_preds receives gradient only at NUM positions
        non_num = ~(targets == NUM_ID)
        assert num_preds.grad.squeeze(-1)[non_num].abs().sum() == 0
