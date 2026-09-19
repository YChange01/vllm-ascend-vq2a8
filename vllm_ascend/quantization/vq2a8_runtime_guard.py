# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled host-metadata guards for an already captured V4 MoE runtime.

This caches the check plan, never a check result. Every invocation inspects the
current runtime/config/bank/root fields and tensor addresses/layouts. It does
not read tensor payloads, synchronize a device, or permit implicit recapture.
"""

from dataclasses import dataclass

import torch

RUNTIME_GUARD_MODES = ("signature", "planned", "native")
RUNTIME_FIELDS = (
    ("v4_compute_backend", "v1"),
    ("v4_activation_preparation", "rowwise"),
    ("v4_activation_reorder", "scalar"),
    ("v4_b1_schedule", "baseline"),
    ("v4_validity_mode", "torch"),
    ("v4_route_mapping", "torch"),
    ("v4_select_sign", "separate"),
    ("v4_activation_tail", "torch"),
    ("v4_runtime_guard", "signature"),
)
CONFIG_FIELDS = ("top_k", "hidden_size", "num_shared", "renormalize", "routed_scale", "swiglu_limit")
GEOMETRY_FIELDS = ("rows", "columns", "rht_true_columns", "rht_block_size")
CONTRACT_ERROR = "V4 MoE graph runtime/root/geometry signature changed; no implicit recapture."


def _changed(detail):
    raise RuntimeError(f"{CONTRACT_ERROR} ({detail})")


def _same_scalar(actual, expected):
    # A malformed runtime must not smuggle a device scalar into Python's truth
    # conversion. The supported configuration consists only of host scalars.
    return type(actual) is type(expected) and actual == expected


def _host_scalar(value, label):
    if type(value) not in (str, int, float, bool, type(None)):
        _changed(f"{label} must be a host scalar")
    return value


@dataclass(frozen=True)
class _TensorPlan:
    owner: torch.Tensor
    pointer: int
    shape: torch.Size
    stride: tuple
    offset: int
    dtype: torch.dtype
    device: torch.device
    label: str

    @classmethod
    def compile(cls, tensor, label):
        if not isinstance(tensor, torch.Tensor):
            _changed(f"{label} is not a tensor")
        return cls(
            tensor,
            tensor.data_ptr(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
            label,
        )

    def check(self, tensor):
        if (
            tensor is not self.owner
            or tensor.data_ptr() != self.pointer
            or tensor.shape != self.shape
            or tensor.stride() != self.stride
            or tensor.storage_offset() != self.offset
            or tensor.dtype != self.dtype
            or tensor.device != self.device
        ):
            _changed(self.label)


class PlannedRuntimeGuard:
    """Strongly-owned, per-invocation equivalent of the runtime signature gate.

    The captured root dictionary can be rewrapped with the same tensor objects;
    changing a key, tensor identity, storage offset or layout is rejected. Bank
    and config owners are retained and must stay identical. Projection-spec
    objects can be rewrapped only when all captured geometry values stay equal.
    The guard does not claim to detect in-place changes to immutable weights.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.config = runtime.config
        self.banks = runtime._device_route_banks
        self.runtime_fields = tuple(
            (name, default, _host_scalar(getattr(runtime, name, default), name)) for name, default in RUNTIME_FIELDS
        )
        self.config_fields = tuple((name, _host_scalar(getattr(self.config, name), name)) for name in CONFIG_FIELDS)
        self.projections = tuple(
            (
                kind,
                self.banks[kind][0],
                tuple((name, _host_scalar(getattr(self.banks[kind][1], name), name)) for name in GEOMETRY_FIELDS),
            )
            for kind in ("gate_up", "down")
        )
        self.root_keys = frozenset(runtime.root)
        self.roots = tuple((name, _TensorPlan.compile(tensor, f"root.{name}")) for name, tensor in runtime.root.items())
        self.lookup = _TensorPlan.compile(self.banks["lookup"], "banks.lookup")

    def _check_host(self, runtime):
        # No sorted(), signature reconstruction, or cross-invocation validation
        # memo. Retained owners also prevent id reuse from hiding replacements.
        if (
            runtime is not self.runtime
            or runtime.config is not self.config
            or runtime._device_route_banks is not self.banks
        ):
            _changed("runtime/config/bank owner")
        for name, default, expected in self.runtime_fields:
            if not _same_scalar(getattr(runtime, name, default), expected):
                _changed(name)
        for name, expected in self.config_fields:
            if not _same_scalar(getattr(self.config, name, None), expected):
                _changed(f"config.{name}")
        for kind, bank, geometry in self.projections:
            current = self.banks.get(kind)
            if not isinstance(current, (tuple, list)) or len(current) != 2 or current[0] is not bank:
                _changed(f"banks.{kind}")
            for name, expected in geometry:
                if not _same_scalar(getattr(current[1], name, None), expected):
                    _changed(f"banks.{kind}.{name}")
        root = runtime.root
        if not isinstance(root, dict) or root.keys() != self.root_keys:
            _changed("root keys")
        return root

    def check(self, runtime):
        root = self._check_host(runtime)
        for name, plan in self.roots:
            plan.check(root[name])
        self.lookup.check(self.banks.get("lookup"))


def native_runtime_guard_factory(*, native_ops=None, native_factory=None):
    """Require the independent host-only ABI; never fall back to Python."""
    native_ops = torch.ops.vq2a8_ascendc_v4_v2 if native_ops is None else native_ops
    try:
        version = native_ops.runtime_guard_version()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("Native runtime guard ABI missing; no fallback.") from error
    if type(version) is not int or version != 1:
        raise RuntimeError(f"Native runtime guard requires independent ABI 1, got {version!r}.")
    if native_factory is None:
        native_factory = torch.classes.vq2a8_ascendc_v4_v2.RuntimeTensorGuard
    return native_factory


class NativeRuntimeGuard(PlannedRuntimeGuard):
    """Live host structure checks plus compiled, strongly owned tensor checks.

    Python checks object identity as well as live container/scalar/geometry
    fields. C++ receives the current tensors, not a captured owner list, and
    compares their metadata with immutable startup snapshots on every call.
    In-place tensor payload changes remain outside this metadata-only contract.
    """

    def __init__(self, runtime, *, native_ops=None, native_factory=None):
        super().__init__(runtime)
        self.native_factory = native_runtime_guard_factory(native_ops=native_ops, native_factory=native_factory)
        plans = tuple(plan for _, plan in self.roots) + (self.lookup,)
        self.native_plan = self.native_factory([plan.owner for plan in plans], [plan.label for plan in plans])

    def collect(self, runtime, tensors):
        """Append checked live fields; no metadata properties or device reads."""
        try:
            root = self._check_host(runtime)
            for name, plan in self.roots:
                tensor = root[name]
                if tensor is not plan.owner:
                    _changed(plan.label)
                tensors.append(tensor)
            lookup = runtime._device_route_banks.get("lookup")
            if lookup is not self.lookup.owner:
                _changed("banks.lookup")
            tensors.append(lookup)
        except (AttributeError, KeyError, TypeError) as error:
            raise RuntimeError(CONTRACT_ERROR) from error

    def check(self, runtime):
        tensors = []
        self.collect(runtime, tensors)
        self.native_plan.check(tensors)


class NativeRuntimeGuardBatch:
    """One C++ metadata call per decoder replay, with no cached pass results."""

    def __init__(self, computes):
        self.computes = tuple(computes)
        if not self.computes:
            raise ValueError("Native runtime guard batch requires captured computes.")
        self.plans = tuple(getattr(compute, "_runtime_guard_plan", None) for compute in self.computes)
        if any(
            getattr(compute, "_runtime_guard_mode", None) != "native" or not isinstance(plan, NativeRuntimeGuard)
            for compute, plan in zip(self.computes, self.plans)
        ):
            raise ValueError("Native decoder runtime guard requires native plans for every captured layer.")
        self.native_plan = self.plans[0].native_factory([], [])
        for plan in self.plans:
            # Copy the existing C++ snapshots, not current metadata: mutation
            # between per-layer construction and aggregation must still fail.
            self.native_plan.append(plan.native_plan)

    def check(self, computes):
        if len(computes) != len(self.computes):
            _changed("decoder compute count")
        tensors = []
        for current, captured, plan in zip(computes, self.computes, self.plans):
            if (
                current is not captured
                or current._runtime_guard_plan is not plan
                or current._runtime_guard_mode != "native"
            ):
                _changed("decoder compute/guard owner")
            plan.collect(current.runtime, tensors)
        self.native_plan.check(tensors)
