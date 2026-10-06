#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sean Wilson
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Run the test suite on another Mac over SSH, so the GUI tests don't take over this machine.
#
# Usage: tests/run_remote.sh HOST /path/to/Blender.app [REMOTE_DIR]
#
# Syncs the Blender app bundle and this repository's working tree (including uncommitted
# changes) to HOST:REMOTE_DIR (default: blender-test), then runs tests/run_tests.sh there.
# Requires key-based SSH, and a user logged in to HOST's desktop for the GUI tests.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 HOST /path/to/Blender.app [REMOTE_DIR]" >&2
  exit 2
fi
HOST="$1"
APP="${2%/}"
REMOTE_DIR="${3:-blender-test}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -x "$APP/Contents/MacOS/Blender" ]; then
  echo "not a Blender app bundle: $APP" >&2
  exit 2
fi

SSH=(ssh -o BatchMode=yes "$HOST")
"${SSH[@]}" "mkdir -p '$REMOTE_DIR'"

echo "Syncing $(basename "$APP") to $HOST:$REMOTE_DIR ..."
rsync -a --delete "$APP" "$HOST:$REMOTE_DIR/"

echo "Syncing repository ..."
rsync -a --delete --exclude .git --exclude __pycache__ --exclude .DS_Store \
  "$REPO/" "$HOST:$REMOTE_DIR/blender-plugins/"

APP_NAME="$(basename "$APP")"
"${SSH[@]}" "cd '$REMOTE_DIR/blender-plugins' && tests/run_tests.sh \"\$HOME/$REMOTE_DIR/$APP_NAME/Contents/MacOS/Blender\""
