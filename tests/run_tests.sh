#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Run the test suite against a Blender build that supports Python compositor nodes.
#
# Usage: tests/run_tests.sh /path/to/Blender      (or set $BLENDER)
#
# The GUI tests (property_updates, thread_stress) briefly open Blender windows.
set -u
cd "$(dirname "$0")"
BLENDER="${1:-${BLENDER:-}}"
if [ -z "$BLENDER" ] || [ ! -x "$BLENDER" ]; then
  echo "usage: $0 /path/to/Blender  (or set BLENDER)" >&2
  exit 2
fi
LOG_DIR="$(mktemp -d)"
failures=0

# `timeout` is not part of macOS, fall back to perl (which is).
with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout > /dev/null; then
    timeout "$seconds" "$@"
  else
    perl -e 'alarm shift; exec @ARGV or die "exec failed: $!"' "$seconds" "$@"
  fi
}

run() {
  local name="$1"
  shift
  if with_timeout 600 "$BLENDER" "$@" > "$LOG_DIR/$name.log" 2>&1 &&
     ! grep -q "internal state bug" "$LOG_DIR/$name.log"; then
    echo "PASS  $name"
  else
    echo "FAIL  $name (log: $LOG_DIR/$name.log)"
    failures=$((failures + 1))
  fi
}

# Install into a throwaway user config so nothing besides the add-on file is visible.
run_isolated() {
  local name="$1"
  shift
  local resources
  resources="$(mktemp -d)"
  BLENDER_USER_RESOURCES="$resources" run "$name" "$@"
  rm -rf "$resources"
}

ARGS=(--factory-startup --python-exit-code 1 --python)
run_isolated install    -b "${ARGS[@]}" test_install.py
run reference           -b "${ARGS[@]}" test_reference.py
run node_cpu            -b "${ARGS[@]}" test_pixel_sort_node.py
run node_gpu            -b "${ARGS[@]}" test_pixel_sort_node.py -- --gpu
run property_updates       "${ARGS[@]}" test_property_updates.py
run thread_stress          "${ARGS[@]}" test_thread_stress.py

echo "$failures failure(s)"
exit $((failures > 0))
