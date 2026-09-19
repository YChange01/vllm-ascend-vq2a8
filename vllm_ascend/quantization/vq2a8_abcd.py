# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only option contracts for independently selectable ABCD candidates."""

ABCD_DEFAULTS = {"runtime_guard": "signature", "select_sign": "separate", "activation_tail": "torch"}


def add_candidate_arguments(parser):
    parser.add_argument("--runtime-guard", choices=("signature", "planned"), default="signature")
    parser.add_argument("--select-sign", choices=("separate", "fused"), default="separate")
    parser.add_argument("--activation-tail", choices=("torch", "fused_reorder"), default="torch")


def validate_candidates(
    runtime_guard="signature",
    select_sign="separate",
    activation_tail="torch",
    *,
    backend="v2",
    policy="ascendc_v4",
    preparation="sign_fused_direct",
    reorder="vectorized",
    device_route=True,
    graph_mode=None,
):
    for name, value, choices in (
        ("runtime_guard", runtime_guard, ("signature", "planned")),
        ("select_sign", select_sign, ("separate", "fused")),
        ("activation_tail", activation_tail, ("torch", "fused_reorder")),
    ):
        if value not in choices:
            raise ValueError(f"Invalid v4_{name}={value!r}; require {choices}.")
    enabled = runtime_guard != "signature" or select_sign != "separate" or activation_tail != "torch"
    if enabled and (backend != "v2" or policy != "ascendc_v4" or device_route is not True):
        raise ValueError("ABCD candidates require V4 v2 device-route decode.")
    if runtime_guard == "planned" and graph_mode == "none":
        raise ValueError("Planned runtime guard requires MoE or decoder graphs.")
    if (select_sign == "fused" or activation_tail == "fused_reorder") and preparation != "sign_fused_direct":
        raise ValueError("Select/sign and tail candidates require sign_fused_direct preparation.")
    if activation_tail == "fused_reorder" and reorder != "vectorized":
        raise ValueError("Fused tail requires vectorized activation reorder.")


def validate_candidate_args(args, *, graph_mode, device_route):
    validate_candidates(
        args.runtime_guard,
        args.select_sign,
        args.activation_tail,
        backend=args.compute_backend,
        preparation=args.activation_preparation,
        reorder=args.activation_reorder,
        graph_mode=graph_mode,
        device_route=device_route,
    )
