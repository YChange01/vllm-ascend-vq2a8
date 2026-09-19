# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only option contracts for independently selectable ABCD candidates."""

ABCD_DEFAULTS = {"runtime_guard": "signature", "select_sign": "separate", "activation_tail": "torch"}


def add_candidate_arguments(parser):
    parser.add_argument("--runtime-guard", choices=("signature", "planned", "native"), default="signature")
    parser.add_argument("--select-sign", choices=("separate", "fused"), default="separate")
    parser.add_argument("--activation-tail", choices=("torch", "fused_reorder"), default="torch")
    parser.add_argument("--decoder-input-mode", choices=("general", "b1_packed"), default="general")
    parser.add_argument("--b1-schedule", choices=("baseline", "tile_major"), default="baseline")
    parser.add_argument("--swiglu-mode", choices=("torch", "fused_select_sign"), default="torch")


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
    decoder_input_mode="general",
    b1_schedule="baseline",
    swiglu_mode="torch",
):
    for name, value, choices in (
        ("runtime_guard", runtime_guard, ("signature", "planned", "native")),
        ("select_sign", select_sign, ("separate", "fused")),
        ("activation_tail", activation_tail, ("torch", "fused_reorder")),
        ("decoder_input_mode", decoder_input_mode, ("general", "b1_packed")),
        ("b1_schedule", b1_schedule, ("baseline", "tile_major")),
        ("swiglu_mode", swiglu_mode, ("torch", "fused_select_sign")),
    ):
        if value not in choices:
            raise ValueError(f"Invalid v4_{name}={value!r}; require {choices}.")
    enabled = (
        runtime_guard != "signature"
        or select_sign != "separate"
        or activation_tail != "torch"
        or decoder_input_mode != "general"
        or b1_schedule != "baseline"
        or swiglu_mode != "torch"
        or reorder in ("chunk_reuse2", "chunk_reuse4")
    )
    if enabled and (backend != "v2" or policy != "ascendc_v4" or device_route is not True):
        raise ValueError("ABCD candidates require V4 v2 device-route decode.")
    if runtime_guard in ("planned", "native") and graph_mode == "none":
        raise ValueError("Planned runtime guard requires MoE or decoder graphs.")
    if decoder_input_mode != "general" and graph_mode != "decoder":
        raise ValueError("Packed decoder input requires V4 v2 decoder graphs.")
    if (select_sign == "fused" or activation_tail == "fused_reorder") and preparation != "sign_fused_direct":
        raise ValueError("Select/sign and tail candidates require sign_fused_direct preparation.")
    if activation_tail == "fused_reorder" and reorder != "vectorized":
        raise ValueError("Fused tail requires vectorized activation reorder.")
    if b1_schedule != "baseline" or reorder in ("chunk_reuse2", "chunk_reuse4"):
        if graph_mode not in (None, "decoder"):
            raise ValueError("Chunk reuse and B1 schedule candidates require decoder graphs.")
        if activation_tail != "torch" or reorder not in ("vectorized", "chunk_reuse2", "chunk_reuse4"):
            raise ValueError("B1 candidates require Torch tail and vectorized/chunk_reuse2/chunk_reuse4 reorder.")
    if swiglu_mode != "torch":
        if graph_mode not in (None, "decoder"):
            raise ValueError("Fused SwiGLU/select/sign requires decoder graphs.")
        if (
            select_sign != "fused"
            or preparation != "sign_fused_direct"
            or activation_tail != "torch"
            or reorder != "vectorized"
            or b1_schedule != "baseline"
        ):
            raise ValueError(
                "Fused SwiGLU/select/sign requires fused select/sign, sign_fused_direct preparation, "
                "Torch tail, vectorized reorder and baseline B1 schedule."
            )


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
        decoder_input_mode=getattr(args, "decoder_input_mode", "general"),
        b1_schedule=getattr(args, "b1_schedule", "baseline"),
        swiglu_mode=getattr(args, "swiglu_mode", "torch"),
    )
