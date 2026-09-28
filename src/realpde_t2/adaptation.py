"""Test-time adaptation predictors that honor the Track 2 stream contract."""

from __future__ import annotations

import torch

from realpde_t2.stream_eval import EXPECTED_SHAPE, Normalizer
from realpde_t2.training import physical_relative_uv_mse


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def parameter_is_selected(name: str, prefixes: tuple[str, ...]) -> bool:
    """True if `name` equals a prefix or starts with one (`fc1.` matches `fc1.weight`)."""
    if not prefixes:
        return True
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def select_trainable_parameters(
    model: torch.nn.Module, prefixes: tuple[str, ...]
) -> list[torch.nn.Parameter]:
    """Enable grads only on selected names and return those parameters."""
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        allowed = parameter_is_selected(name, prefixes)
        parameter.requires_grad = allowed
        if allowed:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"no parameters matched prefixes {prefixes}")
    return selected


def _assert_window(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != {EXPECTED_SHAPE}")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"Non-finite {name}")
    return tensor


class GradientTTTPredictor:
    """One previous-window gradient step, then a no-grad current prediction.

    Adaptation uses the cached previous input with the revealed previous
    target. The current target is never consumed. BatchNorm modules stay in
    eval mode so a batch-one update cannot overwrite running statistics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        normalizer: Normalizer,
        learning_rate: float = 1e-4,
        momentum: float = 0.0,
        adapt_steps: int = 1,
        measured_channels: int = 2,
        denominator_epsilon: float = 1e-12,
        trainable_prefixes: tuple[str, ...] = (),
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if adapt_steps < 1:
            raise ValueError("adapt_steps must be at least 1")
        self.model = model.to(device)
        self.device = device
        self.normalizer = normalizer.to(device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adapt_steps = adapt_steps
        self.measured_channels = measured_channels
        self.denominator_epsilon = denominator_epsilon
        self.trainable_prefixes = tuple(trainable_prefixes)
        self._init_state = _clone_state_dict(self.model)
        self._optimizer: torch.optim.Optimizer | None = None
        self._prev_input: torch.Tensor | None = None
        self.last_adapt_loss: float | None = None
        self.adapt_losses: list[float] = []
        self.reset()

    def _make_optimizer(self) -> torch.optim.SGD:
        return torch.optim.SGD(
            select_trainable_parameters(self.model, self.trainable_prefixes),
            lr=self.learning_rate,
            momentum=self.momentum,
        )

    def _set_eval_mode(self) -> None:
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def reset(self) -> None:
        self.model.load_state_dict(self._init_state)
        self._set_eval_mode()
        self._optimizer = self._make_optimizer()
        self._prev_input = None
        self.last_adapt_loss = None

    def _adapt(self, prev_target_norm: torch.Tensor) -> float:
        if self._prev_input is None:
            raise RuntimeError("Adaptation requires a cached previous input")
        if self._optimizer is None:
            raise RuntimeError("Optimizer missing; call reset() first")
        prev_input = self._prev_input
        prev_target = prev_target_norm.to(self.device, dtype=torch.float32)
        last_loss = None
        self._set_eval_mode()
        for _ in range(self.adapt_steps):
            self._optimizer.zero_grad(set_to_none=True)
            prediction = self.model(prev_input)
            loss = physical_relative_uv_mse(
                prediction,
                prev_target,
                self.normalizer,
                measured_channels=self.measured_channels,
                denominator_epsilon=self.denominator_epsilon,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite adaptation loss: {loss}")
            loss.backward()
            self._optimizer.step()
            last_loss = float(loss.detach().cpu())
        self._set_eval_mode()
        assert last_loss is not None
        return last_loss

    def predict(
        self, input_norm: torch.Tensor, prev_target_norm: torch.Tensor | None
    ) -> torch.Tensor:
        input_norm = _assert_window(input_norm, "input_norm").to(
            self.device, dtype=torch.float32
        )
        self.last_adapt_loss = None
        if prev_target_norm is not None:
            if self._prev_input is None:
                raise ValueError("Received a previous target before a cached input")
            prev_target_norm = _assert_window(prev_target_norm, "prev_target_norm")
            self.last_adapt_loss = self._adapt(prev_target_norm)
            self.adapt_losses.append(self.last_adapt_loss)
        elif self._prev_input is not None:
            raise ValueError("Missing previous target after the first trajectory step")

        with torch.inference_mode():
            prediction = self.model(input_norm)
        prediction = _assert_window(prediction.detach(), "prediction")
        self._prev_input = input_norm.detach()
        return prediction


def bn_running_stats(
    model: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Snapshot BatchNorm running mean/var for reset-isolation tests."""
    snapshots = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            mean = module.running_mean.detach().clone() if module.running_mean is not None else None
            var = module.running_var.detach().clone() if module.running_var is not None else None
            if mean is not None:
                snapshots.append((mean, var))
    return snapshots
