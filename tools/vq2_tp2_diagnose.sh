#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Diagnostic only: no model, device reset, package install or system configuration writes.
set -o pipefail

usage() {
    printf '%s\n' \
        'Usage: bash tools/vq2_tp2_diagnose.sh [--allow-busy]' \
        '  --allow-busy  Run tiny communication checks on occupied physical NPU 0,1.' \
        '                May compete for HBM/communication resources and affect existing jobs.' \
        '                Does not bypass failed/unknown device snapshots; not a performance test.' \
        '  -h, --help    Show this help without collecting diagnostics or starting tests.' \
        'Default: skip communication tests unless both physical NPU 0,1 are reported idle.'
}

VQ2_ALLOW_BUSY=0
while (($#)); do
    case "$1" in
        --allow-busy) VQ2_ALLOW_BUSY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
readonly VQ2_ALLOW_BUSY

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
command -v timeout >/dev/null || { echo 'ERROR: timeout command missing'; exit 1; }
command -v python >/dev/null || exit 1
VQ2_DIAG_DIR=$(mktemp -d /tmp/vq2-tp2-diag.XXXXXX) || exit 1
echo "REPORT=$VQ2_DIAG_DIR"
VQ2_TEST_PID=

cleanup() {
    if [[ -n "$VQ2_TEST_PID" ]]; then
        kill -TERM "$VQ2_TEST_PID" 2>/dev/null || true
        wait "$VQ2_TEST_PID" 2>/dev/null || true
        VQ2_TEST_PID=
    fi
}

check() {
    printf '\n===== %s =====\n' "$1"
    shift
    timeout -k 5s 20s "$@"
    printf 'CHECK_EXIT=%s\n' "$?"
}

idle() {
    timeout -k 5s 20s npu-smi info > "$VQ2_DIAG_DIR/$1-idle.log" 2>&1 || {
        echo "SKIP: NPU snapshot query failed; LOG=$VQ2_DIAG_DIR/$1-idle.log"
        return 2
    }
    python - "$VQ2_DIAG_DIR/$1-idle.log" <<'PY'
import re, sys
from pathlib import Path
try:
    text = Path(sys.argv[1]).read_text(errors="replace")
except OSError as error:
    print(f"SKIP: cannot read NPU snapshot: {error}")
    sys.exit(2)
lines = text.splitlines()
start = next((i for i, s in enumerate(lines) if "Process id" in s and "NPU ID" in s), None)
if start is None:
    print("SKIP: cannot recognize NPU process table")
    sys.exit(2)
table = "\n".join(lines[start:])
empty = {int(x) for x in re.findall(r"No running processes found in NPU\s+(\d+)\s*\|", table)}
busy = {int(x) for x in re.findall(r"(?m)^\s*\|\s*(\d+)\s*\|\s*\d+\s*\|", table)}
targets = {0, 1}
if not targets <= empty | busy or targets & empty & busy:
    print("IDLE_SNAPSHOT: unknown or contradictory device state; skip test")
    sys.exit(2)
if targets & busy:
    print("IDLE_SNAPSHOT: occupied physical NPUs=" + ",".join(map(str, sorted(targets & busy))))
    # Distinct from Python's ordinary error exit code: only this may be overridden.
    sys.exit(10)
print("IDLE_SNAPSHOT: 0,1 clear; not an exclusive reservation")
sys.exit(0)
PY
}

run_test() {
    local label=$1 pid rc isolation=IDLE_SNAPSHOT
    shift
    idle "$label"
    rc=$?
    if ((rc != 0)); then
        if ((rc == 10 && VQ2_ALLOW_BUSY == 1)); then
            isolation=SHARED_DEVICE
            echo "WARNING: --allow-busy permits $label on occupied NPU 0,1; existing jobs may be affected."
            echo "WARNING: NPU/HCCL runtime memory exceeds the tiny tensor size; resource contention can cause failure."
        else
            echo "$label=SKIPPED_BUSY_OR_UNKNOWN SNAPSHOT_EXIT=$rc ALLOW_BUSY=$VQ2_ALLOW_BUSY"
            return
        fi
    fi
    echo "TEST_ISOLATION=$isolation TEST=$label TIMING_VALID=False"
    printf '\n===== %s: only physical NPU 0,1; timeout 180s =====\n' "$label"
    timeout -k 10s 180s env ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_LAUNCH_BLOCKING=1 \
        ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1 "$@" \
        > "$VQ2_DIAG_DIR/$label.log" 2>&1 &
    pid=$!
    VQ2_TEST_PID=$pid
    while kill -0 "$pid" 2>/dev/null; do
        echo "WAIT=$label LOG=$VQ2_DIAG_DIR/$label.log"
        sleep 5
    done
    wait "$pid"
    rc=$?
    VQ2_TEST_PID=
    echo "$label EXIT=$rc"
    grep -nE 'RAW_STAGE=|RAW_PASS=|TP2_SMOKE_(STAGE|STATUS|RESULT)=' "$VQ2_DIAG_DIR/$label.log"
    echo 'FIRST_ERRORS:'
    grep -n -m 6 -B 3 -A 6 -E '\[ERROR\]|\[Error\]|Traceback|RuntimeError|Error:|Exception:' "$VQ2_DIAG_DIR/$label.log"
    echo 'LOADED_LIBRARIES:'
    grep '^LOADED_LIB=' "$VQ2_DIAG_DIR/$label.log" | sort -u
}

{
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ((VQ2_ALLOW_BUSY)); then
    echo 'DIAG_POLICY=ALLOW_BUSY PHYSICAL_NPUS=0,1 TIMING_VALID=False'
else
    echo 'DIAG_POLICY=REQUIRE_IDLE PHYSICAL_NPUS=0,1 TIMING_VALID=False'
fi
check system uname -a
check container-security grep -E '^(Cap|NoNewPrivs|Seccomp)' /proc/self/status
check packages python -c 'from importlib.metadata import version; print({p: version(p) for p in ("torch","torch-npu","vllm")})'
check npu npu-smi info
check topology npu-smi info -t topo
for i in 0 1; do
    check "signature-enable-$i" npu-smi info -t custom-op-secverify-enable -i "$i"
    check "signature-mode-$i" npu-smi info -t custom-op-secverify-mode -i "$i"
done
check ub-devices ls -l /dev/uburma /dev/ummu
check npu-devices ls -l /dev/davinci0 /dev/davinci1 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
check ub-libraries bash -c 'ls -ld /usr/lib64/urma; ls -l /usr/lib64/liburma* /usr/lib64/libummu*'
check urma-version urma_admin --version
check urma-devices urma_admin show
check configs ls -ld /lib/route.conf /etc/hccl_rootinfo.json /etc/hixlep /usr/local/Ascend/driver/topo
check driver-version cat /usr/local/Ascend/driver/version.info
check cann-paths readlink -f /usr/local/Ascend/cann-9.1.0 /usr/local/Ascend/ascend-toolkit/latest
check hccl-dependencies ldd /usr/local/Ascend/ascend-toolkit/latest/lib64/libhccl_v2.so
check mounts grep -E '/dev/uburma|/dev/ummu|/usr/lib64|/usr/local/Ascend|/etc/hccl|/etc/hixlep|/lib/route.conf' /proc/self/mountinfo
check environment bash -c 'env | grep -E "^(LD_LIBRARY_PATH|ASCEND_HOME_PATH|ASCEND_OPP_PATH|ASCEND_RT_VISIBLE_DEVICES|ASCEND_UB_DRV_MOUNT|ASCEND_LOCAL_COMM_RES|ASCEND_LOCAL_COMM_RES_PATH|ASCEND_GLOBAL_RESOURCE_CONFIG|HCCL_[A-Z0-9_]+|GLOO_SOCKET_IFNAME|VLLM_DISTRIBUTED_USE_SPLIT_GROUP)="'

run_test torch_hccl python -u -m torch.distributed.run --standalone --nproc-per-node=2 --no-python python -u -c '
import os, traceback
from datetime import timedelta
from pathlib import Path
rank = int(os.environ["LOCAL_RANK"])
def stage(name):
    print(f"RAW_STAGE={name} rank={rank}", flush=True)
try:
    import torch
    import torch_npu
    assert rank in (0, 1) and torch.npu.device_count() == 2
    stage("set_device")
    torch.npu.set_device(rank)
    x = torch.full((1, 8), 3.0 * (rank + 1), dtype=torch.float32, device=f"npu:{rank}")
    assert torch.equal(x.cpu(), torch.full((1, 8), 3.0 * (rank + 1), dtype=torch.float32))
    stage("single_device_pass")
    torch.distributed.init_process_group("hccl", timeout=timedelta(seconds=120))
    stage("allreduce")
    torch.distributed.all_reduce(x)
    assert torch.equal(x.cpu(), torch.full((1, 8), 9.0, dtype=torch.float32))
    print(f"RAW_PASS=allreduce rank={rank}", flush=True)
    torch.distributed.destroy_process_group()
except Exception:
    traceback.print_exc()
    raise
finally:
    for path in sorted({s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines() if "/" in s and any(k in s for k in ("libhccl", "libhcomm", "liburma", "libummu", "libascend_hal", "libascendcl"))}):
        print("LOADED_LIB=" + path, flush=True)
'

run_test vllm_hccl python -u -m torch.distributed.run --standalone --nproc-per-node=2 \
    tools/validate_vq2a8_tp2_collective.py --communication-only --timeout-s 120
echo "DONE REPORT=$VQ2_DIAG_DIR"
} 2>&1 | tee "$VQ2_DIAG_DIR/summary.log"
if tar -czf "$VQ2_DIAG_DIR.tar.gz" -C "$VQ2_DIAG_DIR" .; then
    echo "UPLOAD=$VQ2_DIAG_DIR.tar.gz"
else
    echo "ARCHIVE_FAILED: upload $VQ2_DIAG_DIR/summary.log and test logs instead"
fi
