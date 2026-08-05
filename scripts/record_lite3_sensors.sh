#!/usr/bin/env bash

set -Ee -o pipefail

SCRIPT_DIR="$(
  CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)"
VALIDATOR="${LITE3_VALIDATOR:-$SCRIPT_DIR/verify_lite3_capture.py}"

CAPTURE_SECONDS="${LITE3_CAPTURE_SECONDS:-70}"
BAG_ROOT="${LITE3_BAG_ROOT:-$HOME/lite3_bags}"
MINIMUM_FREE_GIB="${LITE3_MINIMUM_FREE_GIB:-3}"

ROS_SETUP="${LITE3_ROS_SETUP:-/opt/ros/foxy/setup.bash}"
MID360_SETUP="${LITE3_MID360_SETUP:-$HOME/lite_cog_ros2/driver/mid360_ws/install/setup.bash}"
REALSENSE_SETUP="${LITE3_REALSENSE_SETUP:-$HOME/lite_cog_ros2/driver/realsense_ws/install/setup.bash}"

VALIDATE_ONLY=''
SKIP_KERNEL_CHECK=0
RECORDER_PIDS=()

usage() {
  cat <<'USAGE'
Usage:
  record_lite3_sensors.sh [options]
  record_lite3_sensors.sh --validate-only RUN_DIR

Options:
  --duration SECONDS       Recording timeout (default: 70)
  --bag-root DIRECTORY     Parent directory for captures
  --validate-only RUN_DIR  Validate an existing split capture
  --skip-kernel-check      Do not prompt for sudo journal access
  -h, --help               Show this help

The capture path is read-only with respect to robot motion: it never publishes
to /cmd_vel or any other control topic.
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
    --duration)
      require_argument "$1" "${2-}"
      CAPTURE_SECONDS="$2"
      shift 2
      ;;
    --bag-root)
      require_argument "$1" "${2-}"
      BAG_ROOT="$2"
      shift 2
      ;;
    --validate-only)
      require_argument "$1" "${2-}"
      VALIDATE_ONLY="$2"
      shift 2
      ;;
    --skip-kernel-check)
      SKIP_KERNEL_CHECK=1
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

[[ -f "$VALIDATOR" ]] || fail "validator not found: $VALIDATOR"

if [[ -n "$VALIDATE_ONLY" ]]; then
  exec python3 "$VALIDATOR" "$VALIDATE_ONLY"
fi

[[ "$CAPTURE_SECONDS" =~ ^[0-9]+$ ]] ||
  fail '--duration must be an integer'
((CAPTURE_SECONDS > 0)) || fail '--duration must be positive'
[[ "$MINIMUM_FREE_GIB" =~ ^[0-9]+$ ]] ||
  fail 'LITE3_MINIMUM_FREE_GIB must be an integer'

for setup_file in \
  "$ROS_SETUP" \
  "$MID360_SETUP" \
  "$REALSENSE_SETUP"
do
  [[ -f "$setup_file" ]] ||
    fail "setup file not found: $setup_file"
done

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$MID360_SETUP"
# shellcheck disable=SC1090
source "$REALSENSE_SETUP"
set -u

for required_command in \
  awk \
  cat \
  date \
  df \
  du \
  grep \
  hostname \
  lsusb \
  mktemp \
  python3 \
  ros2 \
  setsid \
  sleep \
  sync \
  tee \
  timeout \
  uname
do
  command -v "$required_command" >/dev/null ||
    fail "required command not found: $required_command"
done

python3 -c 'import sqlite3' ||
  fail 'Python sqlite3 module is unavailable'

if ((!SKIP_KERNEL_CHECK)); then
  for required_command in journalctl sudo; do
    command -v "$required_command" >/dev/null ||
      fail "required command not found: $required_command"
  done
fi

mkdir -p "$BAG_ROOT"

AVAILABLE_KIB="$(
  df -Pk "$BAG_ROOT" |
    awk 'NR == 2 {print $4}'
)"
REQUIRED_KIB=$((MINIMUM_FREE_GIB * 1024 * 1024))
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] ||
  fail 'could not determine free disk space'
((AVAILABLE_KIB >= REQUIRED_KIB)) ||
  fail "less than ${MINIMUM_FREE_GIB} GiB free under $BAG_ROOT"

RUN_DIR="$(
  mktemp -d \
    "$BAG_ROOT/lite3_concurrent_$(date +%Y%m%d_%H%M%S)_XXXXXX"
)"
QOS_FILE="$RUN_DIR/lite3_record_qos.yaml"
PREFLIGHT_LOG="$RUN_DIR/preflight.log"
CAPTURE_ENV="$RUN_DIR/capture.env"

printf 'export RUN_DIR=%q\n' "$RUN_DIR" \
  > "$HOME/lite3_current_run.env"

cat >"$QOS_FILE" <<'QOS_YAML'
/timefix/imu:
  history: keep_last
  depth: 1000
  reliability: reliable
  durability: volatile
QOS_YAML

{
  echo "run_directory=$RUN_DIR"
  echo "created_at=$(date -Ins)"
  echo "hostname=$(hostname)"
  echo "capture_seconds=$CAPTURE_SECONDS"
  echo "ros_distro=${ROS_DISTRO-unknown}"
  echo "ros_domain_id=${ROS_DOMAIN_ID-0}"
  echo "rmw_implementation=${RMW_IMPLEMENTATION-default}"
  uname -a
  df -h "$BAG_ROOT"
} >"$RUN_DIR/manifest.txt"

echo "RUN_DIR=$RUN_DIR"
echo '=== 运动安全检查 ==='

if ! CMD_INFO="$(ros2 topic info /cmd_vel 2>&1)"; then
  printf '%s\n' "$CMD_INFO"
  fail 'could not inspect /cmd_vel'
fi
printf '%s\n' "$CMD_INFO" | tee -a "$PREFLIGHT_LOG"
grep -q 'Publisher count: 0' <<<"$CMD_INFO" ||
  fail '/cmd_vel has a publisher; capture is blocked'

echo '=== D435I USB枚举（录制前） ===' | tee -a "$PREFLIGHT_LOG"
if ! USB_BEFORE="$(lsusb -d 8086:0b3a 2>&1)"; then
  printf '%s\n' "$USB_BEFORE" | tee -a "$PREFLIGHT_LOG"
  fail 'D435I 8086:0b3a is not present'
fi
grep -qi '8086:0b3a' <<<"$USB_BEFORE" ||
  fail 'D435I 8086:0b3a is not present'
printf '%s\n' "$USB_BEFORE" | tee -a "$PREFLIGHT_LOG"

echo '=== 传感器发布者检查 ===' | tee -a "$PREFLIGHT_LOG"
for topic in \
  /timefix/imu \
  /timefix/lidar \
  /camera/color/image_raw \
  /camera/color/camera_info
do
  if ! topic_info="$(ros2 topic info "$topic" 2>&1)"; then
    printf '\n=== %s ===\n%s\n' \
      "$topic" "$topic_info" | tee -a "$PREFLIGHT_LOG"
    fail "could not inspect $topic"
  fi

  printf '\n=== %s ===\n%s\n' \
    "$topic" "$topic_info" | tee -a "$PREFLIGHT_LOG"
  grep -q 'Publisher count: 1' <<<"$topic_info" ||
    fail "$topic must have exactly one publisher"
done

echo '=== 实际消息检查（最长15秒） ===' | tee -a "$PREFLIGHT_LOG"
python3 - <<'PY' 2>&1 | tee -a "$PREFLIGHT_LOG"
import sys
import time

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu


rclpy.init()
node = rclpy.create_node('lite3_capture_preflight')
received = {}


def remember(name, validator):
    def callback(message):
        if name not in received:
            received[name] = bool(validator(message))
    return callback


subscriptions = [
    node.create_subscription(
        Imu,
        '/timefix/imu',
        remember(
            'imu',
            lambda message: message.header.stamp.sec > 0,
        ),
        qos_profile_sensor_data,
    ),
    node.create_subscription(
        CustomMsg,
        '/timefix/lidar',
        remember(
            'lidar',
            lambda message: (
                message.point_num > 0
                and len(message.points) > 0
                and message.timebase > 0
                and message.header.stamp.sec > 0
            ),
        ),
        qos_profile_sensor_data,
    ),
    node.create_subscription(
        Image,
        '/camera/color/image_raw',
        remember(
            'image',
            lambda message: (
                message.width > 0
                and message.height > 0
                and len(message.data) > 0
                and message.header.stamp.sec > 0
            ),
        ),
        qos_profile_sensor_data,
    ),
    node.create_subscription(
        CameraInfo,
        '/camera/color/camera_info',
        remember(
            'camera_info',
            lambda message: (
                message.header.stamp.sec > 0
                and any(abs(value) > 0.0 for value in message.k)
            ),
        ),
        qos_profile_sensor_data,
    ),
]

deadline = time.monotonic() + 15.0
while rclpy.ok() and time.monotonic() < deadline:
    if len(received) == 4:
        break
    rclpy.spin_once(node, timeout_sec=0.1)

expected = ('imu', 'lidar', 'image', 'camera_info')
passed = True
for name in expected:
    value = received.get(name)
    print('{}={}'.format(name, value if value is not None else 'MISSING'))
    passed = passed and value is True

node.destroy_node()
rclpy.shutdown()

print('LIVE_PREFLIGHT={}'.format('PASS' if passed else 'FAIL'))
sys.exit(0 if passed else 1)
PY

echo '=== 开始三个独立 recorder ==='
echo "录制时间=${CAPTURE_SECONDS}秒"

IMU_LOG="$RUN_DIR/imu_recorder.log"
LIDAR_LOG="$RUN_DIR/lidar_recorder.log"
CAMERA_LOG="$RUN_DIR/camera_recorder.log"
CAPTURE_START_EPOCH="$(date +%s)"
CAPTURE_START_WALL="$(date '+%Y-%m-%d %H:%M:%S')"

cleanup_recorders() {
  local pid
  local remaining

  signal_recorder_groups INT

  if wait_for_recorder_groups 20; then
    reap_recorders
    return
  fi

  signal_recorder_groups TERM
  if wait_for_recorder_groups 5; then
    reap_recorders
    return
  fi

  signal_recorder_groups KILL
  wait_for_recorder_groups 2 || true
  reap_recorders
}

signal_recorder_groups() {
  local signal_name="$1"
  local pid

  for pid in "${RECORDER_PIDS[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill "-$signal_name" -- "-$pid" 2>/dev/null || true
    fi
  done
}

wait_for_recorder_groups() {
  local seconds="$1"
  local attempt
  local pid
  local remaining

  for ((attempt = 0; attempt < seconds; attempt++)); do
    remaining=0
    for pid in "${RECORDER_PIDS[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        remaining=1
      fi
    done
    ((remaining == 0)) && return 0
    sleep 1
  done
  return 1
}

reap_recorders() {
  local pid
  for pid in "${RECORDER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

handle_signal() {
  echo
  echo '收到中断信号，正在停止本次启动的 recorder……' >&2
  exit 130
}

handle_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  cleanup_recorders
  exit "$status"
}

trap handle_signal HUP INT TERM
trap handle_exit EXIT

setsid timeout \
  --signal=INT \
  --kill-after=20s \
  "${CAPTURE_SECONDS}s" \
  ros2 bag record \
    --max-cache-size 0 \
    --qos-profile-overrides-path "$QOS_FILE" \
    -s sqlite3 \
    -o "$RUN_DIR/imu" \
    /timefix/imu \
    >"$IMU_LOG" 2>&1 &
IMU_PID=$!
RECORDER_PIDS+=("$IMU_PID")

setsid timeout \
  --signal=INT \
  --kill-after=20s \
  "${CAPTURE_SECONDS}s" \
  ros2 bag record \
    --max-cache-size 0 \
    -s sqlite3 \
    -o "$RUN_DIR/lidar" \
    /timefix/lidar \
    /tf \
    /tf_static \
    >"$LIDAR_LOG" 2>&1 &
LIDAR_PID=$!
RECORDER_PIDS+=("$LIDAR_PID")

# Foxy rosbag2 0.3.11 must write raw RGB directly and use the offered camera
# QoS.  A non-zero cache left only metadata after timeout/SIGINT shutdown;
# reliable depth 100 accumulated several seconds of stale RGB messages.
setsid timeout \
  --signal=INT \
  --kill-after=20s \
  "${CAPTURE_SECONDS}s" \
  ros2 bag record \
    --max-cache-size 0 \
    -s sqlite3 \
    -o "$RUN_DIR/camera" \
    /camera/color/image_raw \
    /camera/color/camera_info \
    >"$CAMERA_LOG" 2>&1 &
CAMERA_PID=$!
RECORDER_PIDS+=("$CAMERA_PID")

printf 'imu_pid=%s lidar_pid=%s camera_pid=%s\n' \
  "$IMU_PID" "$LIDAR_PID" "$CAMERA_PID"
echo '正在录制，请不要关闭SSH或按 Ctrl+C……'

if wait "$IMU_PID"; then
  IMU_STATUS=0
else
  IMU_STATUS=$?
fi
if wait "$LIDAR_PID"; then
  LIDAR_STATUS=0
else
  LIDAR_STATUS=$?
fi
if wait "$CAMERA_PID"; then
  CAMERA_STATUS=0
else
  CAMERA_STATUS=$?
fi

RECORDER_PIDS=()
trap - EXIT HUP INT TERM
sync

CAPTURE_END_EPOCH="$(date +%s)"

{
  echo "RUN_DIR=$RUN_DIR"
  echo "CAPTURE_START_EPOCH=$CAPTURE_START_EPOCH"
  echo "CAPTURE_END_EPOCH=$CAPTURE_END_EPOCH"
  echo "CAPTURE_SECONDS=$CAPTURE_SECONDS"
  echo "IMU_STATUS=$IMU_STATUS"
  echo "LIDAR_STATUS=$LIDAR_STATUS"
  echo "CAMERA_STATUS=$CAMERA_STATUS"
} >"$CAPTURE_ENV"

echo
printf 'imu_status=%s lidar_status=%s camera_status=%s\n' \
  "$IMU_STATUS" "$LIDAR_STATUS" "$CAMERA_STATUS"
echo '自然定时结束时三个状态均应为124'

echo '=== recorder日志摘要 ==='
for log_file in \
  "$IMU_LOG" \
  "$LIDAR_LOG" \
  "$CAMERA_LOG"
do
  echo
  echo "=== $log_file ==="
  grep -E \
    'Subscribed to topic|All requested topics|ERROR|WARN|Failed|signal_handler|database is locked' \
    "$log_file" || true
done

echo '=== Bag信息 ==='
for bag_name in imu lidar camera; do
  echo
  echo "=== $bag_name ==="
  ros2 bag info "$RUN_DIR/$bag_name" || true
done

echo '=== SQLite自动验收 ==='
set +e
python3 "$VALIDATOR" \
  "$RUN_DIR" \
  --imu-status "$IMU_STATUS" \
  --lidar-status "$LIDAR_STATUS" \
  --camera-status "$CAMERA_STATUS" \
  --json-out "$RUN_DIR/validation.json" \
  2>&1 | tee "$RUN_DIR/validation.txt"
VALIDATION_STATUS="${PIPESTATUS[0]}"
set -e

echo '=== D435I USB枚举（录制后） ==='
set +e
lsusb -d 8086:0b3a | tee "$RUN_DIR/lsusb_after.txt"
USB_ENUM_STATUS="${PIPESTATUS[0]}"
set -e
if ! grep -qi '8086:0b3a' "$RUN_DIR/lsusb_after.txt"; then
  USB_ENUM_STATUS=1
fi

KERNEL_STATUS=2
if ((SKIP_KERNEL_CHECK)); then
  echo 'USB_KERNEL_CHECK=UNKNOWN (--skip-kernel-check)'
else
  echo '=== 录制期间内核USB检查 ==='
  set +e
  sudo journalctl \
    -k \
    -b \
    --since "$CAPTURE_START_WALL" \
    --no-pager \
    >"$RUN_DIR/kernel_capture.log" \
    2>"$RUN_DIR/kernel_capture.err"
  JOURNAL_STATUS=$?
  set -e

  if ((JOURNAL_STATUS != 0)); then
    echo "USB_KERNEL_CHECK=UNKNOWN journal_status=$JOURNAL_STATUS"
    cat "$RUN_DIR/kernel_capture.err"
    KERNEL_STATUS=2
  elif grep -Eiq \
      'xHCI host controller not responding|HC died|error -110|Failed to initialize|USB disconnect' \
      "$RUN_DIR/kernel_capture.log"; then
    echo 'USB_KERNEL_CHECK=FAIL'
    grep -Ei \
      'xHCI host controller not responding|HC died|error -110|Failed to initialize|USB disconnect' \
      "$RUN_DIR/kernel_capture.log"
    KERNEL_STATUS=1
  else
    echo 'USB_KERNEL_CHECK=PASS'
    KERNEL_STATUS=0
  fi
fi

{
  echo "VALIDATION_STATUS=$VALIDATION_STATUS"
  echo "USB_ENUM_STATUS=$USB_ENUM_STATUS"
  echo "KERNEL_STATUS=$KERNEL_STATUS"
} >>"$CAPTURE_ENV"

echo '=== 空间占用 ==='
du -sh "$RUN_DIR"/*

echo
echo "完成：$RUN_DIR"
echo '注意：偶发 error 11 仅记为警告；持续刷屏并伴随停流才需要处理。'

KNOWN_FAILURE=0
if ((
  IMU_STATUS != 124 ||
  LIDAR_STATUS != 124 ||
  CAMERA_STATUS != 124 ||
  VALIDATION_STATUS != 0 ||
  USB_ENUM_STATUS != 0 ||
  KERNEL_STATUS == 1
)); then
  KNOWN_FAILURE=1
fi

if ((KNOWN_FAILURE)); then
  echo 'OVERALL=FAIL'
  exit 1
elif ((KERNEL_STATUS == 2)); then
  echo 'OVERALL=UNKNOWN（内核USB检查未完成）'
  exit 2
fi

echo 'OVERALL=PASS'
exit 0
