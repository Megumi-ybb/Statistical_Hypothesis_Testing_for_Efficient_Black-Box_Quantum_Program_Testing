#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
cd "$HERE/.."
overall=0
for f in harness/tests/test_*.py; do
  echo "== $f"
  "$PY" -m pytest "$f" -q -m "not slow" "$@" || overall=1
done
exit $overall
