#!/usr/bin/env bash

# Run a read-only Lite3 static bag through FAST-LIO, CLIP and GA-BSVM on the
# development computer.  This script never starts navigation or publishes a
# robot motion command.

set -Eeuo pipefail

SCRIPT_DIR="$(
  CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)"
PROJECT_ROOT="$(
  CDPATH='' cd -- "$SCRIPT_DIR/.." &&
    pwd
)"
WORKSPACE_ROOT="$(
  CDPATH='' cd -- "$PROJECT_ROOT/../.." &&
    pwd
)"

CAPTURE_VALIDATOR="$SCRIPT_DIR/verify_lite3_capture.py"
TIMESTAMP_AUDITOR="$SCRIPT_DIR/audit_lite3_timestamps.py"
SMOKE_VALIDATOR="$SCRIPT_DIR/verify_lite3_offline_smoke.py"
FAST_LIO_CONFIG="$PROJECT_ROOT/config/fast_lio_lite3_offline.yaml"
SEMANTIC_CONFIG="$PROJECT_ROOT/config/semantic_mapping_lite3_real.yaml"

ROS_SETUP="${LITE3_OFFLINE_ROS_SETUP:-/opt/ros/humble/setup.bash}"
WORKSPACE_SETUP="${LITE3_OFFLINE_WS_SETUP:-$WORKSPACE_ROOT/install/setup.bash}"
WORK_ROOT="${LITE3_OFFLINE_WORK_ROOT:-$HOME/lite3_offline_runs}"
RATE="${LITE3_OFFLINE_RATE:-0.10}"
DOMAIN_ID="${LITE3_OFFLINE_DOMAIN_ID:-42}"
QUERY="${LITE3_OFFLINE_QUERY:-road}"
CLIP_DEVICE="${LITE3_OFFLINE_CLIP_DEVICE:-cpu}"
MINIMUM_FREE_GIB="${LITE3_OFFLINE_MINIMUM_FREE_GIB:-8}"
READ_AHEAD="${LITE3_OFFLINE_READ_AHEAD:-10000}"

INPUT_SPEC=''
PREPARE_ONLY=0
RUN_DIR=''
INPUT_RUN=''
INPUT_IS_REMOTE=0
OVERALL_STATUS='RUNNING'
SIGNALLED=0
declare -A PROCESS_PIDS=()

usage() {
  cat <<'USAGE'
Usage:
  run_lite3_offline_smoke.sh --input INPUT [options]

INPUT may be either an absolute/local directory or:
  user@host:/absolute/lite3_concurrent_directory

Options:
  --input INPUT          Split Lite3 capture containing imu/lidar/camera
  --work-root DIRECTORY  Output parent (default: ~/lite3_offline_runs)
  --rate RATE            Playback rate in (0, 1], default 0.10
  --domain-id ID         Isolated ROS domain, default 42
  --query TEXT           CLIP query, default "road"
  --clip-device DEVICE   CLIP device, default cpu
  --prepare-only         Copy/reference, hash, audit and merge; do not run ROS nodes
  -h, --help             Show this help

Safety:
  The runner sets ROS_LOCALHOST_ONLY=1 and never starts Nav2,
  active_perception_node, a motion bridge, or a /cmd_vel publisher.
USAGE
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_argument() {
  local option="$1"
  local value="${2-}"
  [[ -n "$value" ]] || fail "$option requires a value"
}

while (($#)); do
  case "$1" in
    --input)
      require_argument "$1" "${2-}"
      INPUT_SPEC="$2"
      shift 2
      ;;
    --work-root)
      require_argument "$1" "${2-}"
      WORK_ROOT="$2"
      shift 2
      ;;
    --rate)
      require_argument "$1" "${2-}"
      RATE="$2"
      shift 2
      ;;
    --domain-id)
      require_argument "$1" "${2-}"
      DOMAIN_ID="$2"
      shift 2
      ;;
    --query)
      require_argument "$1" "${2-}"
      QUERY="$2"
      shift 2
      ;;
    --clip-device)
      require_argument "$1" "${2-}"
      CLIP_DEVICE="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -n "$INPUT_SPEC" ]] || fail '--input is required'
[[ "$DOMAIN_ID" =~ ^[0-9]+$ ]] ||
  fail '--domain-id must be a non-negative integer'
((DOMAIN_ID <= 232)) ||
  fail '--domain-id must be in the ROS 2 range 0..232'
[[ "$MINIMUM_FREE_GIB" =~ ^[0-9]+$ ]] ||
  fail 'LITE3_OFFLINE_MINIMUM_FREE_GIB must be an integer'
[[ "$READ_AHEAD" =~ ^[0-9]+$ ]] ||
  fail 'LITE3_OFFLINE_READ_AHEAD must be an integer'
[[ "$QUERY" != *$'\n'* ]] || fail '--query must not contain a newline'

python3 - "$RATE" <<'PY' ||
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 < value <= 1.0 else 1)
PY
  fail '--rate must be a finite number in (0, 1]'

for required_file in \
  "$CAPTURE_VALIDATOR" \
  "$TIMESTAMP_AUDITOR" \
  "$SMOKE_VALIDATOR" \
  "$FAST_LIO_CONFIG" \
  "$SEMANTIC_CONFIG" \
  "$ROS_SETUP" \
  "$WORKSPACE_SETUP"
do
  [[ -f "$required_file" ]] || fail "required file not found: $required_file"
done

for required_command in \
  awk \
  date \
  df \
  diff \
  find \
  git \
  grep \
  mktemp \
  mv \
  cp \
  ps \
  python3 \
  sha256sum \
  sort \
  stat \
  stdbuf \
  tee \
  timeout \
  xargs
do
  command -v "$required_command" >/dev/null ||
    fail "required command not found: $required_command"
done

if [[ "$INPUT_SPEC" =~ ^([^:]+):(/.*)$ ]]; then
  INPUT_IS_REMOTE=1
  REMOTE_HOST="${BASH_REMATCH[1]}"
  REMOTE_PATH="${BASH_REMATCH[2]}"
  [[ "$REMOTE_HOST" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    fail 'remote input must use a safe user@host destination'
  [[ "$REMOTE_PATH" =~ ^/[A-Za-z0-9._/+@=-]+$ ]] ||
    fail 'remote path contains unsupported shell characters'
  for remote_command in flock scp ssh; do
    command -v "$remote_command" >/dev/null ||
      fail "required remote-copy command not found: $remote_command"
  done
else
  [[ "$INPUT_SPEC" == /* ]] ||
    INPUT_SPEC="$PWD/$INPUT_SPEC"
  [[ -d "$INPUT_SPEC" ]] ||
    fail "local input is not a directory: $INPUT_SPEC"
  INPUT_SPEC="$(
    CDPATH='' cd -- "$INPUT_SPEC" &&
      pwd -P
  )"
  WORK_ROOT_CHECK="$(
    python3 -c \
      'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
      "$WORK_ROOT"
  )"
  case "$WORK_ROOT_CHECK/" in
    "$INPUT_SPEC/"*)
      fail '--work-root must not be inside the source capture'
      ;;
  esac
fi

mkdir -p "$WORK_ROOT"
WORK_ROOT="$(
  CDPATH='' cd -- "$WORK_ROOT" &&
    pwd -P
)"

AVAILABLE_KIB="$(
  df -Pk "$WORK_ROOT" |
    awk 'NR == 2 {print $4}'
)"
REQUIRED_KIB=$((MINIMUM_FREE_GIB * 1024 * 1024))
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] ||
  fail 'could not determine free disk space'
((AVAILABLE_KIB >= REQUIRED_KIB)) ||
  fail "less than ${MINIMUM_FREE_GIB} GiB free under $WORK_ROOT"

RUN_DIR="$(
  mktemp -d \
    "$WORK_ROOT/lite3_clip_smoke_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX"
)"
LOGS_DIR="$RUN_DIR/logs"
GENERATED_DIR="$RUN_DIR/generated"
ROS_LOG_DIR="$LOGS_DIR/ros"
mkdir -p "$LOGS_DIR" "$GENERATED_DIR" "$ROS_LOG_DIR"
FAST_LIO_SOURCE_ROOT="${LITE3_OFFLINE_FAST_LIO_SOURCE_ROOT:-$WORKSPACE_ROOT/src/fast_lio}"
LIVOX_SOURCE_ROOT="${LITE3_OFFLINE_LIVOX_SOURCE_ROOT:-$WORKSPACE_ROOT/src/livox_ros_driver2}"
FAST_LIO_DEBUG_DIR="$FAST_LIO_SOURCE_ROOT/Log"
FAST_LIO_DEBUG_BACKUP="$RUN_DIR/preexisting_fast_lio_debug"
FAST_LIO_DEBUG_OUTPUT="$RUN_DIR/fast_lio_debug"
FAST_LIO_DEBUG_PREPARED=0
OVERALL_FILE="$RUN_DIR/OVERALL"
printf '%s\n' "$OVERALL_STATUS" >"$OVERALL_FILE"
printf 'export RUN_DIR=%q\n' "$RUN_DIR" \
  >"$WORK_ROOT/lite3_offline_current_run.env"

write_status() {
  OVERALL_STATUS="$1"
  printf '%s\n' "$OVERALL_STATUS" >"$OVERALL_FILE"
}

group_is_alive() {
  local pid="$1"
  ps -eo pgid=,stat= |
    awk -v expected_pgid="$pid" '
      $1 == expected_pgid && $2 !~ /^Z/ {
        alive = 1
      }
      END {
        exit(alive ? 0 : 1)
      }
    '
}

wait_group_gone() {
  local pid="$1"
  local seconds="$2"
  local attempt
  for ((attempt = 0; attempt < seconds; attempt++)); do
    group_is_alive "$pid" || return 0
    sleep 1
  done
  ! group_is_alive "$pid"
}

stop_group() {
  local name="$1"
  local pid="${PROCESS_PIDS[$name]-}"
  [[ -n "$pid" ]] || return 0

  if group_is_alive "$pid"; then
    kill -INT -- "-$pid" 2>/dev/null || true
    wait_group_gone "$pid" 15 || true
  fi
  if group_is_alive "$pid"; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    wait_group_gone "$pid" 5 || true
  fi
  if group_is_alive "$pid"; then
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait_group_gone "$pid" 2 || true
  fi
  wait "$pid" 2>/dev/null || true
  unset 'PROCESS_PIDS[$name]'
}

cleanup_processes() {
  local name
  for name in player query recorder ga clip fast_lio; do
    stop_group "$name"
  done
}

prepare_fast_lio_debug_files() {
  local filename
  [[ -d "$FAST_LIO_DEBUG_DIR" ]] || {
    echo "WARN: FAST-LIO debug directory was not found: $FAST_LIO_DEBUG_DIR"
    return 0
  }
  mkdir -p "$FAST_LIO_DEBUG_BACKUP" "$FAST_LIO_DEBUG_OUTPUT"
  FAST_LIO_DEBUG_PREPARED=1
  for filename in pos_log.txt mat_pre.txt mat_out.txt dbg.txt; do
    if [[ -e "$FAST_LIO_DEBUG_DIR/$filename" ]]; then
      cp -a -- \
        "$FAST_LIO_DEBUG_DIR/$filename" \
        "$FAST_LIO_DEBUG_BACKUP/$filename"
    fi
  done
}

collect_restore_fast_lio_debug_files() {
  local filename
  ((FAST_LIO_DEBUG_PREPARED)) || return 0
  for filename in pos_log.txt mat_pre.txt mat_out.txt dbg.txt; do
    if [[ -e "$FAST_LIO_DEBUG_DIR/$filename" ]]; then
      mv -- \
        "$FAST_LIO_DEBUG_DIR/$filename" \
        "$FAST_LIO_DEBUG_OUTPUT/$filename"
    fi
    if [[ -e "$FAST_LIO_DEBUG_BACKUP/$filename" ]]; then
      cp -a -- \
        "$FAST_LIO_DEBUG_BACKUP/$filename" \
        "$FAST_LIO_DEBUG_DIR/$filename"
    fi
  done
  FAST_LIO_DEBUG_PREPARED=0
}

write_manifest() {
  local exit_status="$1"
  local git_sha='unknown'
  local git_dirty='unknown'
  if git -C "$PROJECT_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    git_sha="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
      git_dirty='true'
    else
      git_dirty='false'
    fi
  fi

  python3 - \
    "$RUN_DIR" \
    "$INPUT_SPEC" \
    "$INPUT_RUN" \
    "$RATE" \
    "$DOMAIN_ID" \
    "$QUERY" \
    "$CLIP_DEVICE" \
    "$OVERALL_STATUS" \
    "$exit_status" \
    "$git_sha" \
    "$git_dirty" \
    "${ROS_DISTRO-unknown}" \
    >"$RUN_DIR/manifest.json" <<'PY'
import datetime
import json
import sys

(
    run_dir,
    input_spec,
    input_run,
    rate,
    domain_id,
    query,
    clip_device,
    overall,
    exit_status,
    git_sha,
    git_dirty,
    ros_distro,
) = sys.argv[1:]

json.dump(
    {
        'run_directory': run_dir,
        'input_spec': input_spec,
        'resolved_input': input_run,
        'playback_rate': float(rate),
        'ros_domain_id': int(domain_id),
        'ros_localhost_only': True,
        'query': query,
        'clip_device': clip_device,
        'overall': overall,
        'exit_status': int(exit_status),
        'git_sha': git_sha,
        'git_dirty': git_dirty == 'true',
        'ros_distro': ros_distro,
        'finished_at_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'motion_ready': False,
        'calibration_status': (
            'INTRINSICS_FROM_BAG_EXTRINSICS_UNVERIFIED'
        ),
    },
    sys.stdout,
    indent=2,
    sort_keys=True,
)
sys.stdout.write('\n')
PY
}

write_artifact_hashes() {
  (
    CDPATH='' cd -- "$RUN_DIR"
    find . -type f \
      ! -name artifact_sha256.txt \
      -print0 |
      sort -z |
      xargs -0 sha256sum
  ) >"$RUN_DIR/artifact_sha256.txt"
}

handle_signal() {
  SIGNALLED=1
  write_status 'ABORTED'
  echo '收到中断信号，正在停止本次离线进程……' >&2
  exit 130
}

handle_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  cleanup_processes
  collect_restore_fast_lio_debug_files
  if ((SIGNALLED)); then
    status=130
    write_status 'ABORTED'
  elif ((status != 0)) && [[ "$OVERALL_STATUS" == 'RUNNING' ]]; then
    write_status 'FAIL'
  fi
  write_manifest "$status" || true
  write_artifact_hashes || true
  echo "RUN_DIR=$RUN_DIR"
  echo "OVERALL=$OVERALL_STATUS"
  exit "$status"
}

trap handle_signal HUP INT TERM
trap handle_exit EXIT

{
  printf 'export RUN_DIR=%q\n' "$RUN_DIR"
  printf 'export ROS_DOMAIN_ID=%q\n' "$DOMAIN_ID"
  printf 'export ROS_LOCALHOST_ONLY=1\n'
  printf 'export ROS_LOG_DIR=%q\n' "$ROS_LOG_DIR"
  printf 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n'
  printf 'export LITE3_INPUT_SPEC=%q\n' "$INPUT_SPEC"
} >"$RUN_DIR/run.env"
printf '%s\n' "$INPUT_SPEC" >"$RUN_DIR/source_path.txt"

cp -- \
  "$FAST_LIO_CONFIG" \
  "$GENERATED_DIR/fast_lio_lite3_offline.yaml"
cp -- \
  "$SEMANTIC_CONFIG" \
  "$GENERATED_DIR/semantic_mapping_lite3_real.source.yaml"
FAST_LIO_RUNTIME_CONFIG="$GENERATED_DIR/fast_lio_lite3_offline.yaml"

(
  CDPATH='' cd -- "$PROJECT_ROOT"
  sha256sum \
    config/fast_lio_lite3_offline.yaml \
    config/semantic_mapping_lite3_real.yaml \
    scripts/audit_lite3_timestamps.py \
    scripts/run_lite3_offline_smoke.sh \
    scripts/verify_lite3_capture.py \
    scripts/verify_lite3_offline_smoke.py \
    semantic_mapping/clip_node.py \
    semantic_mapping/ga_bsvm_node.py
) >"$RUN_DIR/implementation_sha256.txt"
if (
  [[ -f "$FAST_LIO_SOURCE_ROOT/src/laserMapping.cpp" ]] &&
  [[ -f "$FAST_LIO_SOURCE_ROOT/config/mid360.yaml" ]] &&
  [[ -f "$LIVOX_SOURCE_ROOT/msg/CustomMsg.msg" ]]
); then
  sha256sum \
    "$FAST_LIO_SOURCE_ROOT/src/laserMapping.cpp" \
    "$FAST_LIO_SOURCE_ROOT/config/mid360.yaml" \
    "$LIVOX_SOURCE_ROOT/msg/CustomMsg.msg" \
    >>"$RUN_DIR/implementation_sha256.txt"
fi

git -C "$PROJECT_ROOT" status \
  --porcelain=v1 \
  --untracked-files=all \
  >"$RUN_DIR/git-status.txt" || true
git -C "$PROJECT_ROOT" diff --binary \
  >"$RUN_DIR/git-working-tree.patch" || true
git -C "$PROJECT_ROOT" diff --cached --binary \
  >"$RUN_DIR/git-index.patch" || true
for dependency_name in fast_lio livox_ros_driver2; do
  dependency_root="$WORKSPACE_ROOT/src/$dependency_name"
  if git -C "$dependency_root" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$dependency_root" status \
      --porcelain=v1 \
      --untracked-files=all \
      >"$RUN_DIR/git-status-${dependency_name}.txt" || true
    git -C "$dependency_root" diff --binary \
      >"$RUN_DIR/git-working-tree-${dependency_name}.patch" || true
    git -C "$dependency_root" diff --cached --binary \
      >"$RUN_DIR/git-index-${dependency_name}.patch" || true
  fi
done

echo "RUN_DIR=$RUN_DIR"
echo '=== 准备源 Bag ==='

if ((INPUT_IS_REMOTE)); then
  ssh "$REMOTE_HOST" sh -s -- "$REMOTE_PATH" <<'REMOTE_CHECK'
set -eu
root=$1
test -d "$root/imu"
test -d "$root/lidar"
test -d "$root/camera"
for bag in imu lidar camera; do
  test -f "$root/$bag/metadata.yaml"
  find "$root/$bag" -maxdepth 1 -type f -name '*.db3' | grep -q .
done
REMOTE_CHECK

  TRANSFER_KEY="$(
    printf '%s' "$INPUT_SPEC" |
      sha256sum |
      awk '{print $1}'
  )"
  TRANSFER_ROOT="$WORK_ROOT/.transfers"
  RAW_PARTIAL="$TRANSFER_ROOT/${TRANSFER_KEY}.partial"
  TRANSFER_LOCK="$TRANSFER_ROOT/${TRANSFER_KEY}.lock"
  mkdir -p "$TRANSFER_ROOT"
  exec {TRANSFER_LOCK_FD}>"$TRANSFER_LOCK"
  flock "$TRANSFER_LOCK_FD"
  mkdir -p "$RAW_PARTIAL"
  if (
    command -v rsync >/dev/null &&
    ssh "$REMOTE_HOST" 'command -v rsync >/dev/null'
  ); then
    rsync \
      -a \
      --partial \
      --info=progress2 \
      "$REMOTE_HOST:$REMOTE_PATH/" \
      "$RAW_PARTIAL/" \
      2>&1 | tee "$LOGS_DIR/transfer.log"
  else
    echo 'WARN: rsync unavailable; falling back to non-resumable scp' |
      tee "$LOGS_DIR/transfer.log"
    scp -pr \
      "$REMOTE_HOST:$REMOTE_PATH/." \
      "$RAW_PARTIAL/" \
      2>&1 | tee -a "$LOGS_DIR/transfer.log"
  fi
  INPUT_RUN="$RAW_PARTIAL"

  ssh "$REMOTE_HOST" sh -s -- "$REMOTE_PATH" \
    >"$RUN_DIR/source_sha256.remote.txt" <<'REMOTE_HASH'
set -eu
cd -- "$1"
find imu lidar camera -type f \
  \( -name '*.db3' -o -name 'metadata.yaml' \) \
  -print0 |
  sort -z |
  xargs -0 sha256sum
REMOTE_HASH
else
  INPUT_RUN="$INPUT_SPEC"
fi

for bag_name in imu lidar camera; do
  [[ -f "$INPUT_RUN/$bag_name/metadata.yaml" ]] ||
    fail "missing $bag_name/metadata.yaml under $INPUT_RUN"
  compgen -G "$INPUT_RUN/$bag_name/*.db3" >/dev/null ||
    fail "no db3 found under $INPUT_RUN/$bag_name"
done

(
  CDPATH='' cd -- "$INPUT_RUN"
  find imu lidar camera -type f \
    \( -name '*.db3' -o -name 'metadata.yaml' \) \
    -print0 |
    sort -z |
    xargs -0 sha256sum
) >"$RUN_DIR/source_sha256.txt"

if [[ -f "$RUN_DIR/source_sha256.remote.txt" ]]; then
  diff -u \
    "$RUN_DIR/source_sha256.remote.txt" \
    "$RUN_DIR/source_sha256.txt" \
    >"$RUN_DIR/source_sha256.diff" ||
    fail 'source and transferred SHA256 manifests differ'
fi

if ((INPUT_IS_REMOTE)); then
  RAW_DIR="$RUN_DIR/raw"
  mv "$INPUT_RUN" "$RAW_DIR"
  INPUT_RUN="$RAW_DIR"
fi

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$WORKSPACE_SETUP"
set -u

export ROS_DOMAIN_ID="$DOMAIN_ID"
export ROS_LOCALHOST_ONLY=1
export ROS2CLI_DISABLE_DAEMON=1
export ROS_LOG_DIR
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

{
  uname -a
  python3 --version
  printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO-unknown}"
  printf 'ROS_LOG_DIR=%s\n' "$ROS_LOG_DIR"
  printf 'RMW_IMPLEMENTATION=%s\n' "$RMW_IMPLEMENTATION"
} >"$RUN_DIR/environment.txt"

for required_command in ros2 setsid xargs; do
  command -v "$required_command" >/dev/null ||
    fail "required command not found: $required_command"
done

python3 - <<'PY'
import rclpy
import rosbag2_py
import yaml

from livox_ros_driver2.msg import CustomMsg

assert CustomMsg is not None
PY

for package in fast_lio livox_ros_driver2 semantic_mapping; do
  ros2 pkg prefix "$package" >/dev/null ||
    fail "ROS package is unavailable: $package"
done

echo '=== 输入 Bag 验收 ==='
python3 "$CAPTURE_VALIDATOR" \
  "$INPUT_RUN" \
  --json-out "$RUN_DIR/input_validation.json" \
  2>&1 | tee "$RUN_DIR/input_validation.txt"

echo '=== 消息时间与标定结构审计 ==='
python3 -u "$TIMESTAMP_AUDITOR" "$INPUT_RUN" \
  2>&1 | tee "$RUN_DIR/timestamp_audit.txt"
grep -q '^SENSOR_HEADER_ALIGNMENT=PASS$' \
  "$RUN_DIR/timestamp_audit.txt" ||
  fail 'sensor header alignment did not meet the offline smoke gate'

MERGED_BAG="$RUN_DIR/merged"
MERGE_OPTIONS="$GENERATED_DIR/merge_options.yaml"
MERGED_URI="$(
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' \
    "$MERGED_BAG"
)"
cat >"$MERGE_OPTIONS" <<EOF
output_bags:
- uri: $MERGED_URI
  storage_id: sqlite3
  all: false
  topics:
  - /timefix/imu
  - /timefix/lidar
  - /camera/color/image_raw
  - /camera/color/camera_info
EOF

echo '=== 合并为单一回放 Bag ==='
ros2 bag convert \
  -i "$INPUT_RUN/imu" \
  -i "$INPUT_RUN/lidar" \
  -i "$INPUT_RUN/camera" \
  -o "$MERGE_OPTIONS" \
  >"$LOGS_DIR/merge.log" 2>&1

python3 "$SMOKE_VALIDATOR" \
  --input-run "$INPUT_RUN" \
  --merged-bag "$MERGED_BAG" \
  --json-out "$RUN_DIR/merge_validation.json" \
  --text-out "$RUN_DIR/merge_validation.txt"

SEMANTIC_RUNTIME_CONFIG="$GENERATED_DIR/semantic_runtime.yaml"
CAMERA_INFO_REPORT="$RUN_DIR/camera_intrinsics.json"
python3 - \
  "$MERGED_BAG" \
  "$SEMANTIC_CONFIG" \
  "$SEMANTIC_RUNTIME_CONFIG" \
  "$CAMERA_INFO_REPORT" \
  "$CLIP_DEVICE" <<'PY'
import json
import math
import pathlib
import sqlite3
import sys

import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo


bag_dir = pathlib.Path(sys.argv[1])
source_config = pathlib.Path(sys.argv[2])
runtime_config = pathlib.Path(sys.argv[3])
report_path = pathlib.Path(sys.argv[4])
clip_device = sys.argv[5]

fingerprints = {}
first_message = None
for database in sorted(bag_dir.glob('*.db3')):
    uri = 'file:{}?mode=ro'.format(database)
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("""
            SELECT messages.data
            FROM messages
            JOIN topics ON topics.id = messages.topic_id
            WHERE topics.name = '/camera/color/camera_info'
            ORDER BY messages.timestamp
        """)
        for (serialized,) in rows:
            message = deserialize_message(bytes(serialized), CameraInfo)
            if first_message is None:
                first_message = message
            roi = message.roi
            fingerprint = (
                int(message.width),
                int(message.height),
                str(message.distortion_model),
                tuple(round(float(value), 12) for value in message.d),
                tuple(round(float(value), 12) for value in message.k),
                tuple(round(float(value), 12) for value in message.r),
                tuple(round(float(value), 12) for value in message.p),
                int(message.binning_x),
                int(message.binning_y),
                (
                    int(roi.x_offset),
                    int(roi.y_offset),
                    int(roi.height),
                    int(roi.width),
                    bool(roi.do_rectify),
                ),
                str(message.header.frame_id),
            )
            fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1

if first_message is None:
    raise SystemExit('ERROR: CameraInfo is missing from merged bag')
if len(fingerprints) != 1:
    raise SystemExit(
        'ERROR: CameraInfo changed during capture: {} fingerprints'.format(
            len(fingerprints)))

k = [float(value) for value in first_message.k]
if (
    len(k) != 9
    or not all(math.isfinite(value) for value in k)
    or k[0] <= 0.0
    or k[4] <= 0.0
    or abs(k[8] - 1.0) > 1e-6
    or not 0.0 <= k[2] < float(first_message.width)
    or not 0.0 <= k[5] < float(first_message.height)
):
    raise SystemExit('ERROR: CameraInfo K is not geometrically valid')

configuration = yaml.safe_load(source_config.read_text(encoding='utf-8'))
clip_parameters = configuration['clip_node']['ros__parameters']
ga_parameters = configuration['ga_bsvm_node']['ros__parameters']
clip_parameters['device'] = clip_device
clip_parameters['use_sim_time'] = True
ga_parameters['use_sim_time'] = True
ga_parameters['semantic_backend'] = 'clip'
ga_parameters['camera_k'] = k
runtime_config.write_text(
    yaml.safe_dump(configuration, sort_keys=False),
    encoding='utf-8',
)

report = {
    'message_count': sum(fingerprints.values()),
    'fingerprint_count': len(fingerprints),
    'width': int(first_message.width),
    'height': int(first_message.height),
    'frame_id': first_message.header.frame_id,
    'distortion_model': first_message.distortion_model,
    'd': [float(value) for value in first_message.d],
    'camera_k': k,
    'status': 'INTRINSICS_FROM_BAG_EXTRINSICS_UNVERIFIED',
    'motion_ready': False,
}
report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY

QOS_FILE="$GENERATED_DIR/playback_qos.yaml"
cat >"$QOS_FILE" <<'QOS_YAML'
/timefix/imu:
  history: keep_last
  depth: 1000
  reliability: reliable
  durability: volatile
/timefix/lidar:
  history: keep_last
  depth: 100
  reliability: reliable
  durability: volatile
/camera/color/image_raw:
  history: keep_last
  depth: 20
  reliability: reliable
  durability: volatile
/camera/color/camera_info:
  history: keep_last
  depth: 20
  reliability: reliable
  durability: volatile
QOS_YAML

QUERY_WATCHER="$GENERATED_DIR/query_after_nonempty_cloud.py"
cat >"$QUERY_WATCHER" <<'PY'
"""Publish one text query after receiving a non-empty semantic cloud."""

import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


def main():
    query = sys.argv[1]
    timeout_seconds = float(sys.argv[2])
    rclpy.init()
    node = rclpy.create_node('lite3_offline_query_once')
    publisher = node.create_publisher(String, '/text_query', 10)
    state = {'nonempty': False}

    def receive_cloud(message):
        if int(message.width) * int(message.height) > 0:
            state['nonempty'] = True

    subscription = node.create_subscription(
        PointCloud2,
        '/semantic_cloud',
        receive_cloud,
        qos_profile_sensor_data,
    )
    deadline = time.monotonic() + timeout_seconds
    while (
        rclpy.ok()
        and time.monotonic() < deadline
        and not state['nonempty']
    ):
        rclpy.spin_once(node, timeout_sec=0.2)

    if not state['nonempty']:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit('timed out waiting for a non-empty semantic cloud')

    while (
        rclpy.ok()
        and time.monotonic() < deadline
        and publisher.get_subscription_count() < 1
    ):
        rclpy.spin_once(node, timeout_sec=0.2)

    if publisher.get_subscription_count() < 1:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit('text query has no matching subscriber')

    message = String()
    message.data = query
    publisher.publish(message)
    for unused_iteration in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)

    print('published_query={!r}'.format(query), flush=True)
    del subscription
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
PY

if ((PREPARE_ONLY)); then
  write_status 'PREPARED'
  exit 0
fi

python3 - \
  "$SEMANTIC_RUNTIME_CONFIG" \
  "$RUN_DIR/model_preflight.json" \
  "$QUERY" <<'PY' \
  2>&1 | tee "$LOGS_DIR/model_preflight.log"
import json
import hashlib
import math
import os
import pathlib
import sys

import cv_bridge
import open_clip
import scipy
import torch
import yaml

configuration = yaml.safe_load(
    pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
parameters = configuration['clip_node']['ros__parameters']
model_name = parameters['model_name']
pretrained = parameters['pretrained']
device = parameters['device']
query = sys.argv[3]
pretrained_configuration = open_clip.get_pretrained_cfg(
    model_name, pretrained)

if device == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('ERROR: requested CUDA for CLIP but CUDA is unavailable')

model, unused_train, unused_preprocess = (
    open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
)
model = model.eval().to(device)
tokenizer = open_clip.get_tokenizer(model_name)
with torch.no_grad():
    feature = model.encode_text(tokenizer([query]).to(device))
    feature = feature.float()
    feature = feature / feature.norm(dim=-1, keepdim=True)
values = feature.detach().cpu().reshape(-1).tolist()
if len(values) != 512 or not all(math.isfinite(value) for value in values):
    raise SystemExit('ERROR: CLIP offline model preflight produced invalid output')

weight_candidates = []
hf_repository = str(
    pretrained_configuration.get('hf_hub', '')).rstrip('/')
if hf_repository:
    hf_cache = pathlib.Path(os.environ.get(
        'HF_HUB_CACHE',
        pathlib.Path(os.environ.get(
            'HF_HOME',
            pathlib.Path.home() / '.cache' / 'huggingface',
        )) / 'hub',
    ))
    repository_cache = (
        hf_cache
        / ('models--' + hf_repository.replace('/', '--'))
    )
    if repository_cache.is_dir():
        for pattern in ('*.safetensors', '*.bin', '*.pt'):
            weight_candidates.extend(
                repository_cache.glob('snapshots/*/' + pattern))

if not weight_candidates:
    checkpoint_directory = pathlib.Path(
        torch.hub.get_dir()) / 'checkpoints'
    for pattern in ('*.safetensors', '*.bin', '*.pt'):
        weight_candidates.extend(checkpoint_directory.glob(pattern))

weight_files = []
for candidate in sorted(set(weight_candidates)):
    resolved = candidate.resolve()
    digest = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    weight_files.append({
        'snapshot_path': str(candidate),
        'resolved_path': str(resolved),
        'size_bytes': resolved.stat().st_size,
        'sha256': digest.hexdigest(),
    })

if not weight_files:
    raise SystemExit(
        'ERROR: CLIP loaded, but its cached weight file could not be located')

report = {
    'model_name': model_name,
    'pretrained': pretrained,
    'device': device,
    'query': query,
    'query_feature_length': len(values),
    'query_feature_norm': math.sqrt(sum(value * value for value in values)),
    'weight_files': weight_files,
    'offline_environment': {
        'HF_HUB_OFFLINE': True,
        'TRANSFORMERS_OFFLINE': True,
    },
    'passed': True,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print('CLIP_OFFLINE_PREFLIGHT=PASS')
PY

echo '=== 隔离域安全检查 ==='
NODE_LIST_OUTPUT="$(
  ros2 node list --no-daemon --spin-time 2 2>&1 || true
)"
EXISTING_NODES="$(
  awk '/^\// {print}' <<<"$NODE_LIST_OUTPUT"
)"
if [[ -n "$EXISTING_NODES" ]]; then
  printf '%s\n' "$EXISTING_NODES"
  fail "ROS domain $DOMAIN_ID is not empty; choose another --domain-id"
fi

topic_publisher_count() {
  local topic_name="$1"
  local info
  info="$(
    ros2 topic info \
      --no-daemon \
      --spin-time 0.5 \
      "$topic_name" 2>&1 || true
  )"
  awk '
    /Publisher count:/ {
      print $3
      found = 1
      exit
    }
    END {
      if (!found) {
        print 0
      }
    }
  ' <<<"$info"
}

[[ "$(topic_publisher_count /cmd_vel)" == '0' ]] ||
  fail '/cmd_vel has a publisher in the isolated domain'

prepare_fast_lio_debug_files

start_group() {
  local name="$1"
  local log_file="$2"
  shift 2
  setsid stdbuf -oL -eL "$@" >"$log_file" 2>&1 &
  PROCESS_PIDS["$name"]=$!
  echo "$name pid=${PROCESS_PIDS[$name]}"
}

wait_for_runtime_endpoint() {
  local process_name="$1"
  local topic_name="$2"
  local expected_publishers="$3"
  local timeout_seconds="$4"
  local attempt
  local pid="${PROCESS_PIDS[$process_name]}"

  for ((attempt = 0; attempt < timeout_seconds; attempt++)); do
    local core_process
    for core_process in fast_lio clip ga; do
      if [[ -n "${PROCESS_PIDS[$core_process]-}" ]]; then
        group_is_alive "${PROCESS_PIDS[$core_process]}" ||
          fail "$core_process exited while waiting for $topic_name"
      fi
    done
    group_is_alive "$pid" ||
      fail "$process_name exited before $topic_name became ready"
    if [[ "$(topic_publisher_count "$topic_name")" == "$expected_publishers" ]]; then
      return 0
    fi
    sleep 1
  done
  fail "timeout waiting for $topic_name publisher"
}

start_group \
  fast_lio \
  "$LOGS_DIR/fast_lio.log" \
  ros2 run fast_lio fastlio_mapping --ros-args \
    --params-file "$FAST_LIO_RUNTIME_CONFIG" \
    -p use_sim_time:=true

start_group \
  clip \
  "$LOGS_DIR/clip.log" \
  ros2 run semantic_mapping clip_node --ros-args \
    --params-file "$SEMANTIC_RUNTIME_CONFIG"

start_group \
  ga \
  "$LOGS_DIR/ga.log" \
  ros2 run semantic_mapping ga_bsvm_node --ros-args \
    --params-file "$SEMANTIC_RUNTIME_CONFIG"

wait_for_runtime_endpoint fast_lio /Odometry 1 60
wait_for_runtime_endpoint fast_lio /tf 1 60
wait_for_runtime_endpoint clip /clip_logits 1 180
wait_for_runtime_endpoint ga /semantic_cloud 1 60

OUTPUT_BAG="$RUN_DIR/output_bag"
start_group \
  recorder \
  "$LOGS_DIR/recorder.log" \
  ros2 bag record \
    --use-sim-time \
    --max-cache-size 0 \
    -s sqlite3 \
    -o "$OUTPUT_BAG" \
    /clock \
    /Odometry \
    /path \
    /tf \
    /tf_static \
    /cloud_registered \
    /clip_logits \
    /clip_features \
    /text_query \
    /query_feature \
    /semantic_cloud \
    /uncertainty_cloud \
    /voxel_entropy_data \
    /semantic_cost_map \
    /query_target_pose \
    /goal_pose

sleep 3
group_is_alive "${PROCESS_PIDS[recorder]}" ||
  fail 'output recorder exited before playback'

read -r MERGED_DURATION_S WATCHDOG_SECONDS < <(
  python3 - "$MERGED_BAG" "$RATE" <<'PY'
import math
import pathlib
import sqlite3
import sys

bag = pathlib.Path(sys.argv[1])
rate = float(sys.argv[2])
first = None
last = None
for database in sorted(bag.glob('*.db3')):
    uri = 'file:{}?mode=ro'.format(database)
    with sqlite3.connect(uri, uri=True) as connection:
        low, high = connection.execute(
            'SELECT MIN(timestamp), MAX(timestamp) FROM messages').fetchone()
    if low is not None:
        first = int(low) if first is None else min(first, int(low))
        last = int(high) if last is None else max(last, int(high))
if first is None or last is None or last <= first:
    raise SystemExit('merged bag has no usable duration')
duration = (last - first) * 1e-9
watchdog = int(math.ceil(duration / rate + 240.0))
print('{:.9f} {}'.format(duration, watchdog))
PY
)
echo "merged_duration=${MERGED_DURATION_S}s watchdog=${WATCHDOG_SECONDS}s"

start_group \
  player \
  "$LOGS_DIR/player.log" \
  timeout \
    --signal=INT \
    --kill-after=20s \
    "${WATCHDOG_SECONDS}s" \
    ros2 bag play "$MERGED_BAG" \
      --rate "$RATE" \
      --clock 100 \
      --read-ahead-queue-size "$READ_AHEAD" \
      --disable-keyboard-controls \
      --qos-profile-overrides-path "$QOS_FILE" \
      --topics \
        /timefix/imu \
        /timefix/lidar \
        /camera/color/image_raw \
        /camera/color/camera_info

start_group \
  query \
  "$LOGS_DIR/query.log" \
  python3 -u "$QUERY_WATCHER" "$QUERY" "$WATCHDOG_SECONDS"

RUNTIME_GRAPH="$RUN_DIR/runtime_graph.json"
AUTHORITY_LOG="$LOGS_DIR/authority_samples.tsv"
printf 'clock\todometry\ttf\tcmd_vel\n' >"$AUTHORITY_LOG"
GRAPH_READY=0
for unused_attempt in {1..20}; do
  CLOCK_PUBLISHERS="$(topic_publisher_count /clock)"
  ODOMETRY_PUBLISHERS="$(topic_publisher_count /Odometry)"
  TF_PUBLISHERS="$(topic_publisher_count /tf)"
  CMD_VEL_PUBLISHERS="$(topic_publisher_count /cmd_vel)"
  if ((
    CLOCK_PUBLISHERS == 1 &&
    ODOMETRY_PUBLISHERS == 1 &&
    TF_PUBLISHERS == 1 &&
    CMD_VEL_PUBLISHERS == 0
  )); then
    GRAPH_READY=1
    break
  fi
  sleep 1
done

((GRAPH_READY)) ||
  fail 'runtime publisher graph did not reach the isolated expected state'

AUTHORITY_SAMPLE_COUNT=0
while group_is_alive "${PROCESS_PIDS[player]}"; do
  for required_process in fast_lio clip ga recorder; do
    group_is_alive "${PROCESS_PIDS[$required_process]}" ||
      fail "$required_process exited during playback"
  done

  CLOCK_PUBLISHERS="$(topic_publisher_count /clock)"
  ODOMETRY_PUBLISHERS="$(topic_publisher_count /Odometry)"
  TF_PUBLISHERS="$(topic_publisher_count /tf)"
  CMD_VEL_PUBLISHERS="$(topic_publisher_count /cmd_vel)"
  group_is_alive "${PROCESS_PIDS[player]}" || break

  printf '%s\t%s\t%s\t%s\n' \
    "$CLOCK_PUBLISHERS" \
    "$ODOMETRY_PUBLISHERS" \
    "$TF_PUBLISHERS" \
    "$CMD_VEL_PUBLISHERS" \
    >>"$AUTHORITY_LOG"
  AUTHORITY_SAMPLE_COUNT=$((AUTHORITY_SAMPLE_COUNT + 1))

  if ! ((
    CLOCK_PUBLISHERS == 1 &&
    ODOMETRY_PUBLISHERS == 1 &&
    TF_PUBLISHERS == 1 &&
    CMD_VEL_PUBLISHERS == 0
  )); then
    fail 'runtime publisher authority changed during playback'
  fi
  sleep 2
done

PLAYER_PID="${PROCESS_PIDS[player]}"
if wait "$PLAYER_PID"; then
  PLAYER_STATUS=0
else
  PLAYER_STATUS=$?
fi
unset 'PROCESS_PIDS[player]'
echo "player_status=$PLAYER_STATUS"
((PLAYER_STATUS == 0)) ||
  fail "ros2 bag play failed or timed out with status $PLAYER_STATUS"

for required_process in fast_lio clip ga recorder; do
  group_is_alive "${PROCESS_PIDS[$required_process]}" ||
    fail "$required_process exited at the end of playback"
done

python3 - \
  "$RUNTIME_GRAPH" \
  "$CLOCK_PUBLISHERS" \
  "$ODOMETRY_PUBLISHERS" \
  "$TF_PUBLISHERS" \
  "$CMD_VEL_PUBLISHERS" \
  "$AUTHORITY_SAMPLE_COUNT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
keys = (
    'clock_publishers',
    'odometry_publishers',
    'tf_publishers',
    'cmd_vel_publishers',
    'authority_sample_count',
)
values = [int(value) for value in sys.argv[2:]]
document = dict(zip(keys, values))
document['authority_violations'] = 0
path.write_text(
    json.dumps(document, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY

sleep 3
stop_group query
stop_group recorder
stop_group ga
stop_group clip
stop_group fast_lio

echo '=== 离线算法输出验收 ==='
python3 "$SMOKE_VALIDATOR" \
  --input-run "$INPUT_RUN" \
  --merged-bag "$MERGED_BAG" \
  --output-bag "$OUTPUT_BAG" \
  --logs-dir "$LOGS_DIR" \
  --runtime-graph "$RUNTIME_GRAPH" \
  --json-out "$RUN_DIR/smoke_validation.json" \
  --text-out "$RUN_DIR/smoke_validation.txt"

write_status 'ALGORITHM_STATIC_PASS_NON_GEOMETRIC'
exit 0
