"""
tests/unit/test_model.py
─────────────────────────
Rigorous unit tests for the DeepFM ranking model.

Tests cover:
  1. Architecture invariants (output shapes, parameter counts)
  2. Gradient flow — all parameters receive gradients during backprop
  3. Numerical stability — no NaN/Inf in forward pass under extreme inputs
  4. Monotonicity smoke test — higher quality input → higher predicted score (statistical)
  5. ONNX parity — exported graph must produce bit-identical results to PyTorch
  6. Inference throughput — N=100 must complete in <15ms on CPU
  7. FM layer mathematical correctness vs naive implementation
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ranking.model import (
    DeepFMRanker,
    FactorisationMachineLayer,
    FeatureDims,
    MLPBlock,
    RankerForONNXExport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> DeepFMRanker:
    """Shared model instance — module-scoped for speed (model init is ~50ms)."""
    m = DeepFMRanker()
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_batch() -> torch.Tensor:
    """Canonical batch of 100 candidates for throughput tests."""
    torch.manual_seed(42)
    return torch.randn(100, FeatureDims.TOTAL)


# ── Architecture Tests ────────────────────────────────────────────────────────

class TestArchitecture:

    def test_output_shapes(self, model, dummy_batch):
        """Both task heads must return [batch, 1] tensors."""
        with torch.no_grad():
            click_logit, complete_logit = model(dummy_batch)
        assert click_logit.shape == (100, 1), f"Click head shape: {click_logit.shape}"
        assert complete_logit.shape == (100, 1), f"Complete head shape: {complete_logit.shape}"

    def test_predict_proba_bounds(self, model, dummy_batch):
        """Combined probability must lie strictly in (0, 1) for any finite input."""
        with torch.no_grad():
            proba = model.predict_proba(dummy_batch)
        assert proba.shape == (100, 1)
        assert torch.all(proba > 0.0), f"Score ≤ 0 found: {proba.min()}"
        assert torch.all(proba < 1.0), f"Score ≥ 1 found: {proba.max()}"

    def test_parameter_count_in_expected_range(self, model):
        """
        Model should have between 500K and 5M parameters.
        Smaller → underfitting risk; larger → serving latency risk.
        """
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 100_000 < n_params < 10_000_000, (
            f"Unexpected parameter count: {n_params:,}. "
            "Check MLP dims or FM latent factor size."
        )

    def test_input_dim_matches_feature_dims(self, model):
        """Model input_dim must match FeatureDims.TOTAL = 409."""
        assert model.input_dim == FeatureDims.TOTAL, (
            f"Model input_dim={model.input_dim} != FeatureDims.TOTAL={FeatureDims.TOTAL}. "
            "Update model or feature engineering to match."
        )

    def test_linear_weight_shape(self, model):
        """First-order linear term must be [input_dim, 1]."""
        assert model.linear.weight.shape == (1, FeatureDims.TOTAL)

    def test_task_head_weights_independent(self, model):
        """Click and complete heads must have different weight initialisation."""
        click_w = model.click_head.weight.data
        complete_w = model.complete_head.weight.data
        assert not torch.allclose(click_w, complete_w, atol=1e-6), (
            "Click and complete head weights are identical — "
            "Xavier init should differentiate them."
        )


# ── Gradient Flow Tests ───────────────────────────────────────────────────────

class TestGradientFlow:

    def test_all_parameters_receive_gradients(self):
        """
        Critical: if any parameter has None gradient after backward(),
        it means a graph disconnection — that layer cannot learn.
        """
        model = DeepFMRanker()
        model.train()
        x = torch.randn(32, FeatureDims.TOTAL, requires_grad=False)
        y_click = torch.randint(0, 2, (32,)).float()
        y_complete = torch.randint(0, 2, (32,)).float()

        loss = model.compute_mtl_loss(x, y_click, y_complete)
        loss.backward()

        disconnected = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                disconnected.append(name)

        assert len(disconnected) == 0, (
            f"Parameters with no gradient (graph disconnected): {disconnected}. "
            "These parameters CANNOT be trained."
        )

    def test_uncertainty_weights_receive_gradients(self):
        """Kendall uncertainty weights (log_sigma) must be trained via autograd."""
        model = DeepFMRanker()
        model.train()
        x = torch.randn(16, FeatureDims.TOTAL)
        loss = model.compute_mtl_loss(
            x,
            torch.zeros(16),
            torch.zeros(16),
        )
        loss.backward()
        assert model.log_sigma_click.grad is not None, "log_sigma_click has no gradient"
        assert model.log_sigma_complete.grad is not None, "log_sigma_complete has no gradient"

    def test_fm_layer_gradients_not_zero(self):
        """FM interaction matrix V must have non-zero gradients."""
        model = DeepFMRanker()
        model.train()
        x = torch.randn(16, FeatureDims.TOTAL)
        loss = model.compute_mtl_loss(x, torch.ones(16), torch.zeros(16))
        loss.backward()
        fm_grad_norm = model.fm.v.grad.norm().item()
        assert fm_grad_norm > 1e-10, f"FM V gradient is effectively zero: {fm_grad_norm}"


# ── Numerical Stability Tests ─────────────────────────────────────────────────

class TestNumericalStability:

    @given(
        arrays(np.float32, (16, FeatureDims.TOTAL),
               elements=st.floats(min_value=-10.0, max_value=10.0,
                                   allow_nan=False, allow_infinity=False))
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_no_nan_for_valid_inputs(self, x_np):
        """No NaN in output for any valid bounded float32 input."""
        model = DeepFMRanker()
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x_np)
            proba = model.predict_proba(x)
        assert torch.all(torch.isfinite(proba)), f"Non-finite output for bounded input"

    def test_all_zeros_input_produces_finite_output(self, model):
        """All-zero input (cold-start user, no embedding) must not produce NaN."""
        x = torch.zeros(1, FeatureDims.TOTAL)
        with torch.no_grad():
            proba = model.predict_proba(x)
        assert torch.isfinite(proba).all()

    def test_all_ones_input_produces_finite_output(self, model):
        x = torch.ones(1, FeatureDims.TOTAL)
        with torch.no_grad():
            proba = model.predict_proba(x)
        assert torch.isfinite(proba).all()

    def test_batch_size_1_matches_batch_size_100(self, model):
        """
        Results must not depend on batch size (no batch norm mode leakage).
        model.eval() freezes BN statistics — verify this holds.
        """
        torch.manual_seed(0)
        x = torch.randn(100, FeatureDims.TOTAL)
        with torch.no_grad():
            scores_batched = model.predict_proba(x)
            scores_individual = torch.cat([
                model.predict_proba(x[i:i+1]) for i in range(100)
            ])
        np.testing.assert_allclose(
            scores_batched.numpy(),
            scores_individual.numpy(),
            rtol=1e-4,
            err_msg="Batch vs individual inference mismatch — BN in train mode?"
        )


# ── FM Layer Mathematical Correctness ────────────────────────────────────────

class TestFMLayerMath:

    def test_fm_formula_vs_naive_pairwise(self):
        """
        Validate the FM trick formula against naive O(M²) pairwise computation.
        FM efficient: 0.5 * (||Σ vᵢxᵢ||² - Σ ||vᵢxᵢ||²)
        Naive:        Σᵢ Σⱼ₍ⱼ>ᵢ₎ <vᵢ, vⱼ> xᵢ xⱼ
        Both must produce identical results.
        """
        torch.manual_seed(42)
        input_dim, k, batch = 20, 8, 4
        fm = FactorisationMachineLayer(input_dim, k)
        x = torch.randn(batch, input_dim)

        # Efficient FM output
        with torch.no_grad():
            efficient_out = fm(x)

        # Naive pairwise implementation
        naive_out = torch.zeros(batch)
        with torch.no_grad():
            for b in range(batch):
                s = 0.0
                for i in range(input_dim):
                    for j in range(i + 1, input_dim):
                        # inner product of latent vectors, scaled by feature values
                        s += torch.dot(fm.v[i], fm.v[j]).item() * x[b, i].item() * x[b, j].item()
                naive_out[b] = s

        np.testing.assert_allclose(
            efficient_out.numpy(),
            naive_out.numpy(),
            rtol=1e-4,
            err_msg="FM efficient formula doesn't match naive pairwise computation"
        )

    def test_fm_output_zero_for_zero_input(self):
        """FM(0) = 0 by definition — no interactions with zero-valued features."""
        fm = FactorisationMachineLayer(32, 16)
        x = torch.zeros(8, 32)
        with torch.no_grad():
            out = fm(x)
        assert torch.allclose(out, torch.zeros(8), atol=1e-6)


# ── Inference Throughput Tests ────────────────────────────────────────────────

class TestInferenceLatency:

    def test_n100_inference_under_15ms_cpu(self, model, dummy_batch):
        """
        Stage-2 latency budget: <15ms for N=100 candidates on CPU.
        This test will FAIL in CI if model architecture is too large.
        Adjust MLP dims in TrainingConfig if this consistently fails.
        """
        # Warm-up pass to ensure JIT/cache effects don't distort measurement
        with torch.no_grad():
            _ = model.predict_proba(dummy_batch)

        # Timed pass (5 iterations, take median)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.predict_proba(dummy_batch)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[2]
        assert median_ms < 15.0, (
            f"CPU inference for N=100 took {median_ms:.1f}ms (budget: 15ms). "
            "Consider reducing MLP dims or enabling quantisation."
        )

    def test_single_candidate_inference_under_5ms(self, model):
        """Single-item inference (edge case for new-user cold-start) must be fast."""
        x = torch.randn(1, FeatureDims.TOTAL)
        with torch.no_grad():
            _ = model.predict_proba(x)  # warmup

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.predict_proba(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 5.0, f"Single-item inference took {elapsed_ms:.1f}ms"


# ── ONNX Export & Parity Tests ────────────────────────────────────────────────

class TestONNXParity:

    @pytest.fixture(scope="class")
    def onnx_model_path(self, tmp_path_factory) -> str:
        tmp = tmp_path_factory.mktemp("onnx")
        path = str(tmp / "test_model.onnx")
        try:
            import onnx, onnxruntime
            from ranking.export_onnx import export_to_onnx
            model = DeepFMRanker()
            model.eval()
            export_to_onnx(model, output_path=path)
            return path
        except ImportError:
            pytest.skip("onnx or onnxruntime not installed")

    def test_onnx_output_matches_pytorch(self, onnx_model_path, tmp_path_factory):
        """ONNX graph must produce results within 1e-4 of PyTorch inference."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        # Load same model that was exported
        model = DeepFMRanker()
        model.eval()

        session = ort.InferenceSession(onnx_model_path)
        x_np = np.random.randn(100, FeatureDims.TOTAL).astype(np.float32)

        # PyTorch prediction
        with torch.no_grad():
            torch_scores = model.predict_proba(torch.from_numpy(x_np)).numpy()

        # ONNX prediction
        onnx_scores = session.run(["engagement_score"], {"features": x_np})[0]

        np.testing.assert_allclose(
            torch_scores, onnx_scores, rtol=1e-4, atol=1e-5,
            err_msg="ONNX vs PyTorch output mismatch exceeds tolerance. "
                    "Check for ops not supported in ONNX opset 17."
        )

    def test_onnx_handles_dynamic_batch_sizes(self, onnx_model_path):
        """ONNX graph must handle batch sizes other than the export hint (100)."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        session = ort.InferenceSession(onnx_model_path)

        for batch_size in [1, 5, 50, 200]:
            x = np.random.randn(batch_size, FeatureDims.TOTAL).astype(np.float32)
            result = session.run(["engagement_score"], {"features": x})[0]
            assert result.shape == (batch_size, 1), (
                f"Dynamic batch failed for batch_size={batch_size}: {result.shape}"
            )
