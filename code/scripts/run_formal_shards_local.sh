#!/usr/bin/env bash
# 本地"每 GPU 一 shard 一进程"编排器（无 Slurm 环境的 E1/E2/E3-A 执行预案载体）。
#
# 约定：
# - 一卡一 shard 一进程：CUDA_VISIBLE_DEVICES 绑定，进程级失败互不影响；
# - 每 shard 的 stdout/stderr 落到 <run>/orchestration_logs/shard-XXXX.log；
# - 任一 shard 非零退出不中断其余 shard；最后打印汇总并以非零退出；
# - 断点恢复直接重跑同一命令即可（terminal marker 六条件匹配才跳过）。
#
# 用法：
#   run_formal_shards_local.sh --run <run_dir> --gpus 0,1,2,3,4,5,6,7 \
#       --episode-executor softsim_genesis_runner:run_episode \
#       [--target-provider module:function] [--workers 1] [--timeout-s 3600]
#
# 环境变量：GENESIS_PYTHON（Genesis 1.3.3 venv 解释器）、PYOPENGL_PLATFORM=egl、
# __EGL_VENDOR_LIBRARY_FILENAMES 必须在调用侧导出（headless 必需）。
set -u

RUN=""
GPUS="0,1,2,3,4,5,6,7"
EXECUTOR="softsim_genesis_runner:run_episode"
TARGET_PROVIDER=""
WORKERS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --episode-executor) EXECUTOR="$2"; shift 2;;
    --target-provider) TARGET_PROVIDER="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [[ -z "$RUN" ]]; then
  echo "--run <run_dir> is required" >&2
  exit 2
fi
if [[ ! -f "$RUN/run_manifest.json" ]]; then
  echo "run manifest not found: $RUN/run_manifest.json" >&2
  exit 2
fi

SHARD_COUNT=$(python3 -c "import json;print(json.load(open('$RUN/run_manifest.json'))['shard_count'])")
LOG_DIR="$RUN/orchestration_logs"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NGPU=${#GPU_LIST[@]}
echo "run=$RUN shards=$SHARD_COUNT gpus=${GPU_LIST[*]} executor=$EXECUTOR"

PIDS=()
SHARD_OF_PID=()
for ((i = 0; i < SHARD_COUNT; i++)); do
  GPU=${GPU_LIST[$((i % NGPU))]}
  CMD=(softsim-exp run --run "$RUN" --shard-index "$i" --workers "$WORKERS"
       --episode-executor "$EXECUTOR" --process-isolation)
  if [[ -n "$TARGET_PROVIDER" ]]; then
    CMD+=(--target-provider "$TARGET_PROVIDER")
  fi
  echo "launch shard $i on GPU $GPU -> $LOG_DIR/shard-$(printf '%04d' "$i").log"
  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "$LOG_DIR/shard-$(printf '%04d' "$i").log" 2>&1 &
  PIDS+=($!)
  SHARD_OF_PID+=("$i")
done

FAILED=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "shard ${SHARD_OF_PID[$idx]} FAILED (see $LOG_DIR/shard-$(printf '%04d' "${SHARD_OF_PID[$idx]}").log)" >&2
    FAILED=$((FAILED + 1))
  fi
done

softsim-exp status --run "$RUN" || true
if [[ $FAILED -gt 0 ]]; then
  echo "$FAILED shard(s) failed; resume with the same command (terminal markers are hash-checked)" >&2
  exit 1
fi
echo "all $SHARD_COUNT shard(s) completed"
