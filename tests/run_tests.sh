#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Run the test suite against a Blender build that supports Python compositor nodes.
#
# Usage: tests/run_tests.sh /path/to/Blender [PATTERN]   (or set $BLENDER)
#
# PATTERN (a bash glob, default *) selects suites by name, e.g. 'node_*' or 'lab_noise'.
#
# The GUI tests (property_updates, thread_stress) briefly open Blender windows.
set -u
cd "$(dirname "$0")"
BLENDER="${1:-${BLENDER:-}}"
PATTERN="${2:-*}"
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
  # shellcheck disable=SC2053
  [[ "$name" == $PATTERN ]] || return 0
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
# Auto-discovered background suites: framework/test_x.py -> "framework_x", lab/test_x.py -> "lab_x".
for test_file in framework/test_*.py lab/test_*.py; do
  [ -e "$test_file" ] || continue
  suite="$(dirname "$test_file")_$(basename "$test_file" .py)"
  run "${suite/_test_/_}"  -b "${ARGS[@]}" "$test_file"
done
run property_updates       "${ARGS[@]}" test_property_updates.py
run eval_kind_gui          "${ARGS[@]}" test_eval_kind_gui.py
run thread_stress          "${ARGS[@]}" test_thread_stress.py

echo "$failures failure(s)"
exit $((failures > 0))
